#!/usr/bin/env python3
"""
run_llada_metrics_on_jsonl.py  — LLaDA-only, with AUTO-RESUME

What it does
- Loads val.jsonl-like data (prompt/completion/source/meta), optional --filter_source
- Runs LLaDA for each steps in --steps_list (default: 16,32,64,256)
- **Auto-resume**: reads telemetry.rank0.jsonl in --output_dir and SKIPS already-done (example_key,steps)
- Appends new telemetry lines and then rebuilds per-example & aggregated CSVs from ALL telemetry
- No accuracy; logs latency, gpu_seconds, peak_mem_gb, tokens, etc.

Usage (example):
  python eval/run_llada_metrics_on_jsonl.py \
    --dataset_jsonl "/scratch/ssinha78/main-project/d1/dataset/mix_temp07/val.jsonl" \
    --filter_source logiqa \
    --model_path "GSAI-ML/LLaDA-8B-Instruct" \
    --checkpoint_path "/scratch/ssinha78/main-project/d1/eval/sft_checkpoints/llada_mix_temp07/checkpoint-942" \
    --gen_length 256 --cfg_scale 0.0 \
    --steps_list "16,32,64,256" \
    --output_dir "/scratch/ssinha78/main-project/d1/eval/results/llada_jsonl_logiqa_allsteps"

Tip:
- To force re-run from scratch, delete telemetry.rank0.jsonl OR pass --force 1
"""

import argparse
import contextlib
import gc
import json
import logging
import os
import random
import sys
import time
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# import your repo's diffusion generate()
try:
    from generate import generate  # positional args: (model, input_ids, tokenizer, ...)
except Exception:
    print("[error] Could not import `generate`. Run from repo root or fix PYTHONPATH.")
    raise

# ---------- logging ----------
def setup_logger(outdir: str, level: str = "INFO") -> logging.Logger:
    os.makedirs(outdir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(outdir, f"llada_run_{ts}.log")
    logger = logging.getLogger("llada_jsonl_runner")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    fmt = logging.Formatter("%(asctime)s | %(levelname)-5s | %(message)s")
    fh = logging.FileHandler(log_path, "w", encoding="utf-8"); fh.setFormatter(fmt)
    ch = logging.StreamHandler(); ch.setFormatter(fmt)
    logger.handlers = []; logger.addHandler(fh); logger.addHandler(ch)
    logger.info("Log file: %s", log_path)
    return logger

# ---------- utils ----------
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)

def reset_peak_mem():
    if torch.cuda.is_available():
        with contextlib.suppress(Exception): torch.cuda.reset_peak_memory_stats()

def get_peak_mem_gb() -> float:
    if torch.cuda.is_available():
        with contextlib.suppress(Exception):
            return torch.cuda.max_memory_allocated() / (1024 ** 3)
    return 0.0

def clear_cuda_cache(_tag: str = ""):
    if not torch.cuda.is_available(): return
    with contextlib.suppress(Exception): torch.cuda.synchronize()
    with contextlib.suppress(Exception): torch.cuda.empty_cache()
    with contextlib.suppress(Exception): torch.cuda.ipc_collect()
    with contextlib.suppress(Exception): torch.cuda.synchronize()

def jsonl_append(path: str, rec: Dict[str, Any]):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush(); os.fsync(f.fileno())

def safe_load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    if not os.path.isfile(path): return rows
    with open(path, "r", encoding="utf-8") as f:
        for li, line in enumerate(f, 1):
            line = line.strip()
            if not line: continue
            try:
                rows.append(json.loads(line))
            except Exception:
                # tolerate truncated last line
                break
    return rows

def trim_eot(txt: str) -> str:
    for stop in ("<|eot_id|>", "<|endoftext|>"):
        i = txt.find(stop)
        if i >= 0: return txt[:i]
    return txt

def parse_steps_list(s: str) -> List[int]:
    if not s: return [16, 32, 64, 256]
    vals = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok: continue
        try: vals.append(int(tok))
        except: pass
    if not vals: vals = [16, 32, 64, 256]
    keep = []
    seen = set()
    for v in vals:
        if v in (16,32,64,256) and v not in seen:
            keep.append(v); seen.add(v)
    return keep or [16,32,64,256]

def stable_example_key(prompt: str, meta: Optional[dict]) -> str:
    """Prefer meta.id if present; else SHA1 of the prompt (normalized)."""
    mid = None
    if isinstance(meta, dict):
        mid = meta.get("id", None)
    if mid is not None:
        return f"id::{str(mid)}"
    # normalize prompt whitespace to be robust
    norm = " ".join(str(prompt).split())
    return "sha1::" + hashlib.sha1(norm.encode("utf-8")).hexdigest()

# ---------- dataset I/O ----------
def load_jsonl_dataset(path: str, filter_source: Optional[str], logger: logging.Logger) -> List[Dict[str, Any]]:
    if not os.path.isfile(path):
        logger.error("Dataset JSONL not found: %s", path); sys.exit(1)
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for li, line in enumerate(f, 1):
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
            except Exception as e:
                logger.warning("Skipping malformed JSON at line %d: %s", li, e)
                continue
            if filter_source:
                src = str(obj.get("source", "")).strip().lower()
                if src != filter_source.lower(): continue
            prompt = obj.get("prompt", "")
            meta = obj.get("meta", {})
            rows.append({
                "example_index": len(rows),  # index within the filtered slice (not used for resume)
                "example_key": stable_example_key(prompt, meta),
                "prompt": prompt,
                "completion": obj.get("completion", ""),
                "source": obj.get("source", None),
                "meta": meta,
                "_raw": obj,
            })
    logger.info("Loaded %d example(s) from %s (filter_source=%s)", len(rows), path, filter_source)
    return rows

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_jsonl", type=str,
        default="/scratch/ssinha78/main-project/d1/dataset/mix_temp07/val.jsonl")
    ap.add_argument("--filter_source", type=str, default="", help="Filter by `source` (e.g., logiqa, gsm8k).")
    ap.add_argument("--model_path", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--checkpoint_path", type=str, default="")
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--gen_length", type=int, default=256)
    ap.add_argument("--block_length", type=int, default=32)
    ap.add_argument("--cfg_scale", type=float, default=0.0)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--steps_list", type=str, default="16,32,64,256")
    ap.add_argument("--prefer_bfloat16_if_supported", action="store_true")
    ap.add_argument("--force", type=int, default=0, help="Set to 1 to ignore resume and re-run all pairs.")
    ap.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(args.output_dir, args.log_level)
    set_seed(args.seed)
    steps_values = parse_steps_list(args.steps_list)
    logger.info("steps_list=%s | force=%s", steps_values, args.force)

    # env
    logger.info("CUDA available: %s | devices: %d", torch.cuda.is_available(), torch.cuda.device_count())
    if torch.cuda.is_available(): logger.info("GPU: %s", torch.cuda.get_device_name(0))

    # data
    rows = load_jsonl_dataset(args.dataset_jsonl, args.filter_source or None, logger)
    if not rows:
        logger.error("No rows to process."); sys.exit(1)

    # model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.prefer_bfloat16_if_supported else torch.float16
    logger.info("Loading tok: %s", args.model_path)
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    logger.info("Loading model: %s (dtype=%s)", args.model_path, dtype)
    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=dtype).to(device)
    if args.checkpoint_path:
        logger.info("Applying PEFT: %s", args.checkpoint_path)
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.checkpoint_path, torch_dtype=dtype).to(device)

    # paths
    telemetry_path = os.path.join(args.output_dir, "telemetry.rank0.jsonl")
    per_example_csv = os.path.join(args.output_dir, "llada_per_example.csv")
    aggregated_csv  = os.path.join(args.output_dir, "llada_aggregated.csv")
    summary_json    = os.path.join(args.output_dir, "llada_summary.json")

    # --- RESUME: read done pairs from telemetry ---
    done_pairs: Set[Tuple[str,int]] = set()
    if not args.force and os.path.exists(telemetry_path):
        old = safe_load_jsonl(telemetry_path)
        # Only consider rows matching current dataset filter (if provided)
        flt = [r for r in old if (not args.filter_source or str(r.get("dataset","")).lower()==args.filter_source.lower())]
        for r in flt:
            # prefer example_key if present; else reconstruct from prompt hash or legacy fields
            ek = r.get("example_key")
            if not ek:
                p = r.get("prompt", "")
                ek = stable_example_key(p, r.get("meta", {}))
            st = int(r.get("diffusion_steps", _infer_steps_from_setting(r.get("setting",""))))
            if st in (16,32,64,256):
                done_pairs.add((ek, st))
        logger.info("[resume] Found %d done (example_key,steps) pairs in telemetry.", len(done_pairs))
    elif args.force:
        logger.info("[resume] FORCE=1 — ignoring existing telemetry; will re-run all pairs.")

    # run
    total_start = time.time()
    new_pairs = 0
    total_pairs = len(rows) * len(steps_values)
    logger.info("Planned pairs: %d examples × %d steps = %d", len(rows), len(steps_values), total_pairs)

    for ex in rows:
        prompt = ex["prompt"]; src = ex["source"]; ek = ex["example_key"]

        # tokenize once
        enc = tok(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        tokens_in = int(input_ids.numel())

        for steps in steps_values:
            if not args.force and (ek, steps) in done_pairs:
                # already done — skip
                continue

            reset_peak_mem()
            if torch.cuda.is_available():
                with contextlib.suppress(Exception): torch.cuda.synchronize()
            t0 = time.time()
            try:
                # IMPORTANT: positional args (model, input_ids, tokenizer)
                out = generate(
                    model,
                    input_ids,
                    tok,
                    steps=int(steps),
                    gen_length=args.gen_length,
                    block_length=args.block_length,
                    temperature=args.temperature,
                    cfg_scale=args.cfg_scale,
                    remasking="low_confidence",
                )
            except torch.cuda.OutOfMemoryError:
                logger.error("[OOM] ek=%s steps=%d — skipping", ek, steps)
                clear_cuda_cache("[oom]"); continue
            except Exception as e:
                logger.exception("Generation error ek=%s steps=%d: %s", ek, steps, e); continue

            if torch.cuda.is_available():
                with contextlib.suppress(Exception): torch.cuda.synchronize()
            t1 = time.time()
            latency_ms_total = (t1 - t0) * 1000.0
            peak_mem_gb = get_peak_mem_gb()
            gpus = max(torch.cuda.device_count() if torch.cuda.is_available() else 1, 1)
            gpu_seconds = (t1 - t0) * gpus

            # decode + tokens_out
            if isinstance(out, torch.Tensor):
                new_tokens = out[:, -args.gen_length:]
                tokens_out = int(new_tokens.numel())
                texts = tok.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            else:
                texts = out
                tokens_out = args.gen_length
            text = trim_eot(texts[0] if isinstance(texts, list) else texts)

            setting = f"steps{int(steps)}_cfg{args.cfg_scale}_g{args.gen_length}"
            tele = {
                "family": "LLaDA",
                "setting": setting,
                "dataset": str(src or "unknown"),
                "example_key": ek,     # <- stable across runs
                "latency_ms_total": float(latency_ms_total),
                "gpu_seconds": float(gpu_seconds),
                "peak_mem_gb": float(peak_mem_gb),
                "tokens_in": int(tokens_in),
                "tokens_out": int(tokens_out),
                "diffusion_steps": int(steps),
                "gen_length": int(args.gen_length),
                "block_length": int(args.block_length),
                "temperature": float(args.temperature),
                "cfg_scale": float(args.cfg_scale),
                "prompt": prompt,
                "generation": text,
                "meta": ex.get("meta", {}),
            }
            jsonl_append(telemetry_path, tele)
            new_pairs += 1

            if new_pairs % 20 == 0 or new_pairs == 1:
                logger.info("New pairs=%d | ek=%s steps=%d | latency=%.1f ms | peak=%.2f GiB | tok_in=%d tok_out=%d",
                            new_pairs, ek, steps, latency_ms_total, peak_mem_gb, tokens_in, tokens_out)

            with contextlib.suppress(Exception): del out
            gc.collect(); clear_cuda_cache("[pair]")

    wall = time.time() - total_start
    logger.info("Run complete: new_pairs=%d, wall=%.2fs (%.2f pairs/sec)",
                new_pairs, wall, (new_pairs / wall) if wall > 0 else 0.0)

    # ---------- Rebuild CSVs from ALL telemetry (respecting filter_source) ----------
    import pandas as pd
    all_rows = safe_load_jsonl(telemetry_path)
    if args.filter_source:
        all_rows = [r for r in all_rows if str(r.get("dataset","")).lower()==args.filter_source.lower()]

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df.to_csv(per_example_csv, index=False)
        logger.info("Wrote per-example CSV -> %s (rows=%d)", per_example_csv, len(df))

        # aggregation
        for col in ["latency_ms_total","gpu_seconds","peak_mem_gb","tokens_in","tokens_out"]:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
        agg = (df.groupby(["setting","dataset"], as_index=False)
                 .agg(n_examples=("latency_ms_total","size"),
                      med_gpu_seconds=("gpu_seconds", lambda x: float(np.nanmedian(x))),
                      p50_latency_ms=("latency_ms_total", lambda x: float(np.nanmedian(x))),
                      p95_latency_ms=("latency_ms_total", lambda x: float(np.nanpercentile(x,95))),
                      med_peak_mem_gb=("peak_mem_gb", lambda x: float(np.nanmedian(x))),
                      mean_tokens_in=("tokens_in","mean"),
                      mean_tokens_out=("tokens_out","mean")))
        agg.to_csv(aggregated_csv, index=False)
        logger.info("Wrote aggregated CSV -> %s", aggregated_csv)
    else:
        pd.DataFrame(columns=["setting","dataset","n_examples","med_gpu_seconds","p50_latency_ms",
                              "p95_latency_ms","med_peak_mem_gb","mean_tokens_in","mean_tokens_out"]).to_csv(aggregated_csv, index=False)
        logger.warning("No telemetry rows to aggregate; wrote headers only -> %s", aggregated_csv)

    # summary
    summary = {
        "new_pairs": int(new_pairs),
        "wall_seconds": float(wall),
        "steps_list": steps_values,
        "telemetry_jsonl": telemetry_path,
        "per_example_csv": per_example_csv,
        "aggregated_csv": aggregated_csv,
        "model_path": args.model_path,
        "checkpoint_path": args.checkpoint_path,
        "gen_length": args.gen_length,
        "block_length": args.block_length,
        "cfg_scale": args.cfg_scale,
        "dataset_jsonl": args.dataset_jsonl,
        "filter_source": args.filter_source or "",
        "force": int(args.force),
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Wrote summary JSON -> %s", summary_json)
    logger.info("Done.")

# helper: infer steps from legacy 'setting' if needed
def _infer_steps_from_setting(setting: str) -> int:
    # expects things like "steps256_cfg0.0_g256"
    try:
        if not setting: return -1
        for part in setting.split("_"):
            if part.startswith("steps"):
                return int(part.replace("steps",""))
    except Exception:
        return -1
    return -1

if __name__ == "__main__":
    main()