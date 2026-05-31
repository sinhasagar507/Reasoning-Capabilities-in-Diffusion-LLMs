#!/usr/bin/env python3
"""
inspect_models.py

Usage (examples):
  python objective3/scripts/inspect_models.py \
      --model_name GSAI-ML/LLaDA-8B-Instruct \
      --output llada_summary.txt

  python objective3/scripts/inspect_models.py \
      --model_name meta-llama/Meta-Llama-3.1-8B-Instruct \
      --output llama3_8b_summary.txt
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True,
                        help="HF model id or local path")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional path to save the textual summary")
    parser.add_argument("--trust_remote_code", action="store_true",
                        help="Pass through to HF loader if custom code is needed")
    args = parser.parse_args()

    print(f"[info] Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )

    # Collect info
    lines = []
    lines.append(f"=== Model: {args.model_name} ===")
    lines.append(repr(model))
    lines.append("")

    # Try to detect a 'layers' list and attention modules
    n_layers = None
    possible_layer_attrs = ["model.layers", "layers", "transformer.layers", "decoder.layers"]
    for attr in possible_layer_attrs:
        try:
            obj = eval(f"model.{attr}")
            n_layers = len(obj)
            lines.append(f"[info] Detected layer stack at model.{attr} with {n_layers} layers")
            break
        except Exception:
            continue

    # List attention-like submodules
    lines.append("\n[info] Attention-like modules (module name -> class):")
    for name, module in model.named_modules():
        if "attn" in name.lower() or "attention" in name.lower():
            lines.append(f"  {name}: {module.__class__.__name__}")

    text = "\n".join(lines)

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(text)
        print(f"[info] Wrote summary to: {out_path}")
    else:
        print(text)


if __name__ == "__main__":
    main()
