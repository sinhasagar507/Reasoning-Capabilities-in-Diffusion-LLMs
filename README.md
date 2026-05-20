# Reasoning Capabilities in Discrete Diffusion Large Language Models

CSE 576 — Topics in Natural Language Processing, Arizona State University.
Team **EchoAgents**: Ahmad Karimi, Sagar Sinha, Abhishek Subhash Yadav,
Daniyal Ahmed Khan, Mohammad Hamza Choudhry. Mentor: Shri Kumbhar.

This repository holds the **LLaDA (discrete diffusion) evaluation and
supervised-fine-tuning pipeline** used in the project. The autoregressive
(LLaMA-side) attention and repairability code, the LLaDA-side attention
probes, and the mechanistic-probing scripts (entropy / FLI / linear probes /
causal reinit) live in other team members' repositories and are **not**
included here.

## What the project studies

We compare a discrete-diffusion LLM (`LLaDA-8B-Instruct`) against an
autoregressive LLM (`Llama-3-8B-Instruct`) on math and logical-reasoning
benchmarks under **matched decoding compute**. Three objectives:

1. **Compute-parity benchmarking** — Accuracy@1, latency, and VRAM for Base
   and SFT variants of both model families on GSM8K, LogiQA, GPQA, AIME 2025
   (and auxiliary tasks: MATH-500, Countdown, Sudoku).
2. **Repairability** — base decode vs. self-consistency (k=5, T=0.7, p=0.9)
   vs. guided retry, plus repair-success and over-repair metrics.
3. **Mechanistic probing** — diffusion trajectories (entropy, token flips,
   stability, Flip Localization Index) and attention analysis
   (bidirectionality, entropy, ALI, quadrant flows, numeric tokens).

## Scope of this repository

This repo covers the **LLaDA evaluation harness, SFT artifacts, and
diffusion-side step-budget sweeps**. Specifically:

- Unified LLaDA inference + scoring pipeline across datasets.
- SFT LoRA training output for `LLaDA-8B-Instruct` (`llada_mix_temp07`,
  checkpoint-942) used by the rest of the team for downstream analyses.
- Step-budget sweep results (16 / 32 / 64 / 128 / 256 diffusion steps) on
  GSM8K, LogiQA, GPQA, AIME 2025.

The complementary work is held by other team members:
- LLaMA-side repairability and attention probes (Abhishek S. Yadav).
- LLaDA-side attention probes & repairability (Daniyal Ahmed Khan).
- Mechanistic probing (entropy / FLI / linear probes / interventions) on
  top of the `d1` codebase (Mohammad Hamza Choudhry).

## Repository layout

```
.
├── eval.py                          # Original LLaDA eval entry (d1-style)
├── eval_sft_task2.py                # Eval harness used by run_eval_all.sh (SFT + resume)
├── eval_gpqa_arrow.py               # GPQA eval reading local Arrow shards
├── generate.py                      # LLaDA denoising / generation helpers
├── gsm8k.py / math500.py /          # Dataset wrappers (prompt + gold parsing)
│   aime2025.py / logiqa.py /
│   gpqa_diamond.py / gpqa_diamond_2.py /
│   countdown.py / sudoku.py
├── parsers.py / parser_helper.py /  # Answer extraction & scoring utilities
│   parser_json.py / parse_and_get_acc.py /
│   parse_boxed_accuracy.py
├── llada_parity_metrics_verbose.py  # Telemetry-based compute-parity metrics
├── run_llada_metrics_on_jsonl.py    # LLaDA-only metrics over streamed JSONL (auto-resume)
├── run_eval.sh                      # Single-config DDP eval driver
├── run_eval_all.sh                  # Sweep: datasets × step budgets, with resume + retry
├── run_eval_all_gpqa.sh             # GPQA-specific sweep
├── eval_baselines/                  # LLaDA Base generations (reference outputs, gitignored)
├── results/                         # Sweep outputs (gitignored)
└── sft_checkpoints/                 # LoRA adapters (gitignored — distribute out-of-band)
```

## Models

- **Base diffusion model:** `GSAI-ML/LLaDA-8B-Instruct`.
- **SFT variant:** LoRA adapter trained on a custom dataset mixture
  (`llada_mix_temp07`), final checkpoint `checkpoint-942`. Adapter config
  and `adapter_model.safetensors` live under
  `sft_checkpoints/llada_mix_temp07/checkpoint-942/` and are git-ignored
  due to size; obtain via the `.tgz` bundle shared separately.

## Setup

Tested with Python 3.10 / CUDA. No `requirements.txt` is shipped; install
the dependencies the scripts import:

```bash
pip install torch transformers peft datasets accelerate tqdm numpy
```

For GPQA Arrow evaluation, also install `pyarrow`. For multi-GPU runs,
`torchrun` (bundled with PyTorch) is used.

## Running an evaluation

**Single config (DDP across visible GPUs):**

```bash
bash run_eval.sh 0 1 2 3                  # GPU ids
```

This iterates `TASKS=(countdown sudoku math gsm8k)` × `GEN_LENGTHS=(128 256)`
through `eval.py`, using `LLaDA-8B-Instruct` from the path set in the
script.

**Step-budget sweep on a single dataset (resume-safe):**

Edit the user settings at the top of `run_eval_all.sh`:

```bash
DATASETS=("logiqa")            # or gsm8k / aime2025 / gpqa
STEPS=(16 32 64 256)
GENLEN=256
SFT_CKPT=".../sft_checkpoints/llada_mix_temp07/checkpoint-942"
```

then:

```bash
bash run_eval_all.sh           # resumes from JSONL line count / DONE markers
FORCE=1 bash run_eval_all.sh   # ignore DONE markers and re-run
```

Outputs are written under `results/eval_sweep_4datasets/{base,sft}/<dataset>/g<GENLEN>_s<STEPS>/`
as `generations.rank0.jsonl` (streamed) and a final
`*_generations.json` bundle, plus a `DONE` file on success.

**GPQA-specific sweep:** `bash run_eval_all_gpqa.sh` (uses
`eval_gpqa_arrow.py` against a local Arrow dataset directory).

**Compute-parity metrics from telemetry:**

```bash
python llada_parity_metrics_verbose.py --telemetry_dir <dir>
python run_llada_metrics_on_jsonl.py --jsonl <results/.../generations.rank0.jsonl>
```

## Key results held in this repo's experiments

From the team report (full numbers in the project write-up):

| Setting                  | GSM8K  | LogiQA | GPQA-diamond | AIME 2025 |
|--------------------------|:------:|:------:|:------------:|:---------:|
| LLaDA-SFT, 16 steps      | 24.6%  | 39.3%  | 22.6%        | 0.0%      |
| LLaDA-SFT, 32 steps      | 34.3%  | 41.9%  | 24.7%        | 0.0%      |
| LLaDA-SFT, 64 steps      | 56.1%  | 43.5%  | 24.8%        | 0.0%      |
| LLaDA-SFT, 256 steps     | 75.3%  | 47.4%  | 28.0%        | 0.0%      |
| LLaMA-SFT (greedy g128)  | 56%    | 40%    | —            | —         |

Accuracy on LLaDA scales monotonically with the diffusion step budget on
all three solvable benchmarks. AIME 2025 remains at 0% for every
configuration we ran (8B-scale headroom).

## Notes for collaborators

- Heavy artifacts (LoRA checkpoints, generation dumps, telemetry JSONL,
  archives) are `.gitignore`-d. Share these via the project's external
  storage / `.tgz` bundles rather than committing them.
- `results/`, `eval_baselines/`, and `eval_results/` are output
  directories — recreate them by running the scripts above. Do **not**
  commit their contents.
- The folder structure has **not** been reorganized yet; module-level
  refactor is pending.

## References

Core works the project builds on: LLaDA (Nie et al., 2025), GSM8K (Cobbe
et al., 2021), LogiQA (Liu et al., 2020), GPQA (Rein et al., 2023),
self-consistency CoT (Wang et al., 2023), chain-of-thought prompting (Wei
et al., 2022). See the project report for full citations.
