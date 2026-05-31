#!/usr/bin/env python3
"""
Eval script for **GPQA (diamond/main)** that reads a **local Arrow dataset**
(saved-to-disk directory with dataset_info.json OR a single .arrow shard).

This is a focused rewrite of your general eval to only target GPQA, using your
`GPQADiamondArrowDataset` (from gpqa_diamond.py) and keeping the robust
streaming/resume + low-memory fallback behaviors you already rely on.

Key differences vs your previous generic eval:
  - Drops the dataset registry; only GPQA is supported.
  - Requires `--eval_arrow` path (dir with dataset_info.json OR a `.arrow` file).
  - Keeps JSONL streaming (`generations.rank{rank}.jsonl`) and a final bundled
    JSON (
    `gpqa_{modelname}_{gen_length}_{diffusion_steps}_{rank}_generations.json`).
  - Preserves OOM fallbacks: fp16, 4-bit reload (if bitsandbytes installed),
    micro-batching, and cache-clearing knobs.

Usage example (single GPU):

python eval_gpqa_arrow.py \
  --model_path GSAI-ML/LLaDA-8B-Instruct \
  --checkpoint_path /scratch/ssinha78/.../llada_mix_temp07_ckpt942 \
  --eval_arrow /scratch/ssinha78/main-project/d1/dataset/mix_temp07/diamond_gpqa_test \
  --batch_size 1 --gen_length 128 --diffusion_steps 16 \
  --enable_low_mem_fallback --fp16_fallback --clear_cache_every 1 \
  --output_dir results

Notes:
  - If you only have 1 GPU (A100), run as above (no DDP vars set).
  - `--train_jsonl` (optional) decontaminates against your SFT train.
  - If resuming, the script will stitch from JSONL + state.json as before.
"""

import argparse
import contextlib
import gc
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, Subset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# Optional import (used only if you enable 4-bit fallback)
try:
    from transformers import BitsAndBytesConfig
    BNB_AVAILABLE = True
except Exception:
    BNB_AVAILABLE = False

# Your diffusion-style generator
from generate import generate

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def init_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def ddp_is_available():
    return "LOCAL_RANK" in os.environ


def setup_ddp_if_needed():
    if ddp_is_available():
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return local_rank
    else:
        return 0


def cleanup_ddp_if_needed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def get_rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size():
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def clear_cuda_cache(_tag: str = ""):
    if not torch.cuda.is_available():
        return
    with contextlib.suppress(Exception):
        torch.cuda.synchronize()
    with contextlib.suppress(Exception):
        torch.cuda.empty_cache()
    with contextlib.suppress(Exception):
        torch.cuda.ipc_collect()
    with contextlib.suppress(Exception):
        torch.cuda.synchronize()


# -----------------------------------------------------------------------------
# JSONL helpers (streaming)
# -----------------------------------------------------------------------------

def jsonl_count_valid_lines(path: str) -> int:
    if not os.path.exists(path):
        return 0
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
                n += 1
            except Exception:
                break
    return n


def jsonl_append_many(path: str, records):
    with open(path, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def jsonl_load_all(path: str):
    out = []
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                break
    return out


# -----------------------------------------------------------------------------
# Evaluation core
# -----------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model,
    tokenizer,
    dataloader,
    gen_length=128,
    temperature=0.0,
    cfg_scale=0.0,
    steps=64,
    block_length=32,
    # resume controls:
    skip_examples=0,      # how many examples to skip inside THIS dataloader
    state_base=0,         # absolute index offset to write into state.json
    state_path=None,
    jsonl_path=None,
    # ---- fallback knobs (opt-in only) ----
    enable_low_mem_fallback=False,
    fallback_batch_size=1,
    fallback_gen_length=256,
    fallback_diffusion_steps=256,
    fp16_fallback=False,
    bnb_4bit_fallback=False,
    bnb_4bit_compute_dtype="float16",
    alloc_conf_fallback="expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.6",
    # ---- cache clear knobs (opt-in only) ----
    clear_cache_every=0,
    print_mem_stats=False,
):
    model.eval()
    device = model.device
    processed = 0
    wall_times = []

    # we don't need to re-derive already-streamed here; main() sets skip_examples/state_base

    skipped = 0
    for bidx, batch in enumerate(tqdm(dataloader, disable=(get_rank() != 0),
                                      desc=f"Eval (len={gen_length}, steps={steps})")):
        # skip whole batches until we've covered skip_examples
        if skipped < skip_examples:
            bsz = len(batch["questions"])
            if skipped + bsz <= skip_examples:
                skipped += bsz
                continue

        if print_mem_stats and torch.cuda.is_available():
            with contextlib.suppress(Exception):
                torch.cuda.reset_peak_memory_stats()

        start_time = time.time()
        input_ids = batch["input_ids"].to(device)
        gt_answers = batch["answers"]
        questions = batch["questions"]
        prompts = batch["prompts"]

        used_gen_length = gen_length
        tried_bnb_reload = False

        while True:
            try:
                out = generate(
                    model,
                    input_ids,
                    tokenizer,
                    steps=steps,
                    gen_length=gen_length,
                    block_length=block_length,
                    temperature=temperature,
                    cfg_scale=cfg_scale,
                    remasking="low_confidence",
                )
                break  # success
            except torch.cuda.OutOfMemoryError:
                if not enable_low_mem_fallback:
                    raise
                if get_rank() == 0:
                    print("[fallback] CUDA OOM caught. Retrying with low-memory settings...")

                with contextlib.suppress(Exception):
                    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", alloc_conf_fallback)
                clear_cuda_cache("[fallback-pre]")

                if fp16_fallback:
                    with contextlib.suppress(Exception):
                        model.half().to(device)

                if bnb_4bit_fallback and not tried_bnb_reload:
                    if not BNB_AVAILABLE:
                        if get_rank() == 0:
                            print("[fallback] bitsandbytes not available; skipping 4-bit reload.")
                    else:
                        if get_rank() == 0:
                            print("[fallback] Re-loading model in 4-bit quantization...")
                        tried_bnb_reload = True
                        try:
                            del model
                        except Exception:
                            pass
                        gc.collect()
                        clear_cuda_cache("[fallback-reload-pre]")

                        compute_dtype = {
                            "float16": torch.float16,
                            "bfloat16": torch.bfloat16,
                            "float32": torch.float32,
                        }.get(bnb_4bit_compute_dtype.lower(), torch.float16)

                        bnb_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=compute_dtype,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_use_double_quant=True,
                        )
                        mdl_name = tokenizer.name_or_path
                        _ = AutoTokenizer.from_pretrained(mdl_name, trust_remote_code=True)
                        model = AutoModel.from_pretrained(
                            mdl_name,
                            trust_remote_code=True,
                            quantization_config=bnb_config,
                            device_map={"": device.index if hasattr(device, "index") else 0},
                        )
                        clear_cuda_cache("[fallback-reload-post]")

                # micro-batch fallback with altered gen params
                new_len = fallback_gen_length
                new_steps = fallback_diffusion_steps
                used_gen_length = new_len

                outs = []
                with torch.inference_mode():
                    if input_ids.size(0) <= fallback_batch_size:
                        o = generate(
                            model, input_ids, tokenizer,
                            steps=new_steps, gen_length=new_len,
                            block_length=block_length,
                            temperature=temperature, cfg_scale=cfg_scale,
                            remasking="low_confidence",
                        )
                        outs.append(o.detach().cpu() if isinstance(o, torch.Tensor) else o)
                    else:
                        for i in range(0, input_ids.size(0), fallback_batch_size):
                            micro = input_ids[i:i+fallback_batch_size]
                            o = generate(
                                model, micro, tokenizer,
                                steps=new_steps, gen_length=new_len,
                                block_length=block_length,
                                temperature=temperature, cfg_scale=cfg_scale,
                                remasking="low_confidence",
                            )
                            outs.append(o.detach().cpu() if isinstance(o, torch.Tensor) else o)
                            clear_cuda_cache("[fallback-micro]")
                    out = torch.cat(outs, dim=0) if isinstance(outs[0], torch.Tensor) else outs

                with contextlib.suppress(Exception):
                    del input_ids
                clear_cuda_cache("[fallback-post]")
                break  # leave retry loop with fallback result

        # decode
        if isinstance(out, torch.Tensor):
            texts = tokenizer.batch_decode(
                out[:, -used_gen_length:], skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
        else:
            texts = out

        def _trim_eot(txt: str):
            for stop in ("<|eot_id|>", "<|endoftext|>"):
                i = txt.find(stop)
                if i >= 0:
                    return txt[:i]
            return txt

        texts = [_trim_eot(t) for t in texts]

        example_result = [
            {
                "question": questions[j],
                "prompt_input": prompts[j],
                "generations": texts[j],
                "ground_truth": gt_answers[j],
            }
            for j in range(len(gt_answers))
        ]

        if jsonl_path:
            jsonl_append_many(jsonl_path, example_result)

        processed += len(texts)
        wall_times.append(time.time() - start_time)

        # Optional: clear cache after each N batches
        if clear_cache_every > 0 and ((bidx + 1) % clear_cache_every == 0):
            with contextlib.suppress(Exception):
                del out
            with contextlib.suppress(Exception):
                del input_ids
            gc.collect()
            clear_cuda_cache(f"[batch-clear-{bidx+1}]")
            if print_mem_stats and torch.cuda.is_available():
                with contextlib.suppress(Exception):
                    alloc = torch.cuda.memory_allocated() / (1024**3)
                    peak = torch.cuda.max_memory_allocated() / (1024**3)
                    print(f"[mem] after clear: alloc={alloc:.2f} GiB  peak={peak:.2f} GiB")

        if get_rank() == 0:
            idx = random.randint(0, len(questions) - 1)
            print(f"Question: {questions[idx]}")
            print("-" * 50)
            print("Generation:")
            print(texts[idx])
            print("-" * 50)
            print(f"Ground truth: {gt_answers[idx]}")

            # update state.json for human-readable progress; JSONL is source of truth
            if state_path and get_world_size() == 1:
                next_index = state_base + processed
                with contextlib.suppress(Exception):
                    with open(state_path, "w") as f:
                        json.dump({"next_index": int(next_index)}, f)

    avg_wall_time = sum(wall_times) / max(1, len(wall_times))
    return {"wall_time": avg_wall_time, "total_processed": processed}


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    init_seed(42)
    local_rank = setup_ddp_if_needed()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--checkpoint_path", type=str, default="", help="Optional PEFT adapter checkpoint")

    parser.add_argument("--eval_arrow", type=str, required=True,
                        help="Path to Arrow eval set: directory (with dataset_info.json) or .arrow file")
    parser.add_argument("--train_jsonl", type=str, default=None, help="Optional SFT train.jsonl for decontamination")
    parser.add_argument("--few_shot", type=int, default=0)
    parser.add_argument("--subsample", type=int, default=-1)

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gen_length", type=int, default=128)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--diffusion_steps", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cfg_scale", type=float, default=0.0)
    parser.add_argument("--add_reasoning", action="store_true")
    parser.add_argument("--dont_save", action="store_true")
    parser.add_argument("--output_dir", type=str, default="results/")

    # resume controls
    parser.add_argument("--resume_state", type=str, default=None)
    parser.add_argument("--start_index", type=int, default=-1, help="Hard override: resume from this example index.")
    parser.add_argument("--no_finalize", action="store_true", help="Do not write the final bundled JSON")

    # fallback/cache knobs (opt-in)
    parser.add_argument("--enable_low_mem_fallback", action="store_true")
    parser.add_argument("--fallback_batch_size", type=int, default=1)
    parser.add_argument("--fallback_gen_length", type=int, default=256)
    parser.add_argument("--fallback_diffusion_steps", type=int, default=256)
    parser.add_argument("--fp16_fallback", action="store_true")
    parser.add_argument("--bnb_4bit_fallback", action="store_true")
    parser.add_argument("--bnb_4bit_compute_dtype", type=str, default="float16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--alloc_conf_fallback", type=str,
                        default="expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.6")
    parser.add_argument("--clear_cache_every", type=int, default=0)
    parser.add_argument("--print_mem_stats", action="store_true")

    args = parser.parse_args()
    if args.diffusion_steps is None:
        args.diffusion_steps = max(1, args.gen_length // 2)

    os.makedirs(args.output_dir, exist_ok=True)
    if args.resume_state is None:
        args.resume_state = os.path.join(args.output_dir, "state.json")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(local_rank)

    # Optional: PEFT adapter
    if args.checkpoint_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.checkpoint_path, torch_dtype=torch.bfloat16).to(local_rank)
        if get_world_size() > 1:
            dist.barrier()
            for p in model.parameters():
                dist.broadcast(p.data, src=0)

    # ---- Dataset (GPQA only) ----
    from gpqa_diamond_2 import GPQADiamondArrowDataset
    dataset = GPQADiamondArrowDataset(
        tokenizer,
        num_examples=args.few_shot,
        add_reasoning=True if args.add_reasoning else True,  # keep default behavior used before
        train_jsonl=args.train_jsonl,
        subsample=args.subsample,
        custom_eval_arrow=args.eval_arrow,
    )

    total_len = len(dataset)
    rank = get_rank()
    if rank == 0:
        print(f"evaluating {total_len} GPQA examples from {args.eval_arrow}")

    # ---- resume math (JSONL + state + explicit override) ----
    jsonl_path = os.path.join(args.output_dir, f"generations.rank{rank}.jsonl")
    jsonl_lines = jsonl_count_valid_lines(jsonl_path)
    state_next = 0
    if os.path.exists(args.resume_state):
        try:
            state_next = int(json.load(open(args.resume_state))["next_index"])
        except Exception:
            state_next = 0
    start_idx = max(0, jsonl_lines, state_next, args.start_index if args.start_index >= 0 else 0)
    start_idx = min(start_idx, total_len)

    remaining = total_len - start_idx
    bs = max(1, int(args.batch_size))
    expected_batches = (remaining + bs - 1) // bs
    if rank == 0:
        print(f"[resume] continuing GPQA from index {start_idx} "
              f"(jsonl_lines={jsonl_lines}, state={state_next}, override={args.start_index})")
        print(f"[resume] remaining={remaining} examples, batch_size={bs}, expected_batches≈{expected_batches}")

    use_ddp = (get_world_size() > 1)
    if not use_ddp and start_idx > 0:
        dataset = Subset(dataset, range(start_idx, total_len))
        sampler = None
        skip_examples = 0       # because we sliced
        state_base   = start_idx
    else:
        sampler = DistributedSampler(dataset, shuffle=False) if use_ddp else None
        skip_examples = start_idx
        state_base    = start_idx

    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, collate_fn=dataset.collate_fn)

    # name outputs
    if args.checkpoint_path:
        tail = Path(args.checkpoint_path.rstrip("/")).name
        parent = Path(args.checkpoint_path.rstrip("/")).parent.name
        model_name = f"{parent}_{tail}"
    else:
        model_name = "instruct" if "Instruct" in args.model_path else "base"
    if args.few_shot > 0:
        model_name += f"_fs{args.few_shot}"

    out_json = f"{args.output_dir}/gpqa_{model_name}_{args.gen_length}_{args.diffusion_steps}_{rank}_generations.json"
    if rank == 0:
        print(f"Streaming JSONL to {jsonl_path}")
        print(f"Final bundle will be saved to {out_json}")

    # run eval
    metrics = evaluate(
        model, tokenizer, dataloader,
        gen_length=args.gen_length, block_length=args.block_length,
        steps=args.diffusion_steps, temperature=args.temperature,
        cfg_scale=args.cfg_scale,
        skip_examples=skip_examples,
        state_base=state_base,
        state_path=None if use_ddp else args.resume_state,
        jsonl_path=jsonl_path,
        # fallback (opt-in)
        enable_low_mem_fallback=args.enable_low_mem_fallback,
        fallback_batch_size=args.fallback_batch_size,
        fallback_gen_length=args.fallback_gen_length,
        fallback_diffusion_steps=args.fallback_diffusion_steps,
        fp16_fallback=args.fp16_fallback,
        bnb_4bit_fallback=args.bnb_4bit_fallback,
        bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
        alloc_conf_fallback=args.alloc_conf_fallback,
        # cache clear (opt-in)
        clear_cache_every=args.clear_cache_every,
        print_mem_stats=args.print_mem_stats,
    )

    # write final JSON bundle ONLY if we reached end and not suppressed
    if rank == 0 and not args.dont_save and not args.no_finalize:
        gens = jsonl_load_all(jsonl_path)
        with open(out_json, "w") as f:
            json.dump(
                {
                    "generations": gens,
                    "metrics": {
                        "wall_time": metrics["wall_time"],
                        "total_processed": len(gens),
                    },
                    "model_path": args.model_path,
                    "checkpoint_path": args.checkpoint_path,
                    "gen_length": args.gen_length,
                    "diffusion_steps": args.diffusion_steps,
                    "block_length": args.block_length,
                    "source_eval": args.eval_arrow,
                },
                f,
                indent=2,
            )

    cleanup_ddp_if_needed()


if __name__ == "__main__":
    main()
