# Objective 2 — Repairability

Does giving a model a *second chance* fix wrong answers without breaking right
ones? This objective evaluates three decoding strategies per question and
tracks **repair success** vs. **over-repair**.

## Decoding strategies

1. **Base** — one deterministic decode (`T = 0`).
2. **Self-consistency (SC)** — `k = 5` chain-of-thought samples (`T = 0.7`,
   `p = 0.9`), majority vote on the final answer.
3. **Guided retry (GR)** — show the model its own first answer and ask it to
   re-check once, then decode again deterministically.

## Metrics

For `N_total` questions, with per-method correct counts `N_base / N_sc / N_gr`:

- Raw accuracy: `N_method / N_total`
- **Repair success** = `#(base wrong & method correct) / #(base wrong)`
- **Over-repair**    = `#(base correct & method wrong) / #(base correct)`

Compute cost (generations/question): base = 1, SC = 5, GR = 2.

## What's in this folder

| File | Description |
|---|---|
| [LlaDA_repairability_sagar.ipynb](LlaDA_repairability_sagar.ipynb) | **LLaDA** repairability on GSM8K — diffusion-style masked sampling, base / SC / guided-retry decode, and the repair/over-repair metrics above. |

The notebook loads `GSAI-ML/LLaDA-8B-Instruct`, implements Gumbel-max masked
diffusion sampling, GSM8K numeric scoring, and the three decoding schemes.

> Note: cell 2 prompts for a Hugging Face token via `getpass` (no token is
> stored in the notebook). Run it interactively or set `HF_TOKEN` in the
> environment beforehand.

## Not in this repo (owned by teammates)

The **LLaMA-side** repairability pipeline (`run_llama_repairability_on_dataset`,
`llama_generate_answer`, `llama_self_consistency_answer`,
`llama_guided_retry_answer`, and `llama_repairability_results.json`) was
implemented separately and is **not** included here. See the team report for
the LLaMA repairability results (GSM8K base 0.45 → SC 0.45 → GR 0.25).
