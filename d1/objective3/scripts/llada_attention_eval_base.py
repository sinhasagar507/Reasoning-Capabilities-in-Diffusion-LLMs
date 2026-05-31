#!/usr/bin/env python3
import os
import sys

# ── Make sure repository root is on sys.path ────────────────────────────────
THIS_DIR = os.path.dirname(os.path.abspath(__file__))         # .../objective3/scripts
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))  # .../NLP_Project_d1_reasoning

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from objective3.probes import (
    compute_bidirectionality_index,
    compute_attention_entropy,
    compute_attention_localization_index,
    get_llada_layer_attn,
    build_full_input_from_ids_and_answer,
)

# TODO: plug your actual dataset classes here.
# These names are placeholders – replace them with the real ones from your repo.
from gsm8k_test_dataset import GSM8KTestDataset          # <-- you define/import this
from logiqa_extended_dataset import LogiQAExtendedDataset
from gpqa_test_dataset import GPQATestDataset
from aime2025_dataset import AIME2025Dataset

DATASET_MAP = {
    "gsm8k_test": GSM8KTestDataset,
    "logiqa_ext": LogiQAExtendedDataset,
    "gpqa_test":  GPQATestDataset,
    "aime2025":   AIME2025Dataset,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True,
                    help="Base LLaDA model (e.g., GSAI-ML/LLaDA-8B-Instruct)")
    ap.add_argument("--checkpoint_path", default="",
                    help="Optional SFT/LoRA checkpoint; leave empty for base")
    ap.add_argument("--dataset", required=True, choices=list(DATASET_MAP.keys()))
    ap.add_argument("--layers", type=str, default="4,16,24,31",
                    help="Comma-separated layer indices, e.g. '4,16,24,31'")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--max_examples", type=int, default=128,
                    help="Subsample up to this many examples")
    ap.add_argument("--output_path", type=str, default="llada_attn_base.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device: {device}")

    # ----- Tokenizer + model -----
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    if args.checkpoint_path:
        model = PeftModel.from_pretrained(model, args.checkpoint_path, torch_dtype=torch.bfloat16)
    model.to(device)
    model.eval()

    # ----- Dataset -----
    DatasetCls = DATASET_MAP[args.dataset]
    # Match *your* dataset constructor – this is a typical pattern:
    dataset = DatasetCls(tok, split="test", subsample=args.max_examples)
    dl = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=dataset.collate_fn)

    layers = [int(x) for x in args.layers.split(",") if x.strip() != ""]
    print(f"[info] probing layers: {layers}")

    # accumulators
    bi_per_layer = {l: [] for l in layers}
    ent_per_layer = {l: [] for l in layers}
    ali_per_layer = {l: [] for l in layers}

    n_examples = 0

    with torch.no_grad():
        for batch in dl:
            input_ids = batch["input_ids"].to(device)  # [B, L_prompt]
            answers = batch["answers"]                # list of GT answers (str or other)
            B = input_ids.size(0)

            for b in range(B):
                prompt_ids = input_ids[b]
                gt_answer = answers[b]

                full_ids, prompt_len, gen_length = build_full_input_from_ids_and_answer(
                    tok, prompt_ids, gt_answer, device
                )

                for l in layers:
                    attn_layer = get_llada_layer_attn(model, full_ids, layer_idx=l)

                    bi  = compute_bidirectionality_index(attn_layer, prompt_len, gen_length).mean().item()
                    ent = compute_attention_entropy(attn_layer, prompt_len, gen_length).mean().item()
                    ali = compute_attention_localization_index(attn_layer, prompt_len, gen_length).mean().item()

                    bi_per_layer[l].append(bi)
                    ent_per_layer[l].append(ent)
                    ali_per_layer[l].append(ali)

                n_examples += 1
                if n_examples >= args.max_examples:
                    break

            if n_examples >= args.max_examples:
                break

    # ----- aggregate -----
    stats = {}
    for l in layers:
        bi_arr = np.array(bi_per_layer[l], dtype=float)
        ent_arr = np.array(ent_per_layer[l], dtype=float)
        ali_arr = np.array(ali_per_layer[l], dtype=float)

        stats[l] = {
            "bi_mean": float(bi_arr.mean()) if bi_arr.size else 0.0,
            "bi_std":  float(bi_arr.std())  if bi_arr.size else 0.0,
            "ent_mean": float(ent_arr.mean()) if ent_arr.size else 0.0,
            "ent_std":  float(ent_arr.std())  if ent_arr.size else 0.0,
            "ali_mean": float(ali_arr.mean()) if ali_arr.size else 0.0,
            "ali_std":  float(ali_arr.std())  if ali_arr.size else 0.0,
            "num_examples": int(bi_arr.size),
        }

    out = {
        "model_path": args.model_path,
        "checkpoint_path": args.checkpoint_path,
        "dataset": args.dataset,
        "layers": layers,
        "stats": stats,
        "max_examples": args.max_examples,
    }

    with open(args.output_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] {args.output_path}")
    print(f"[info] total examples: {n_examples}")


if __name__ == "__main__":
    main()
