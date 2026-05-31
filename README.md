# Reasoning Capabilities in Discrete Diffusion Large Language Models

CSE 576 — Topics in Natural Language Processing, Arizona State University.
Team **EchoAgents**: Ahmad Karimi, Sagar Sinha, Abhishek Subhash Yadav,
Daniyal Ahmed Khan, Mohammad Hamza Choudhry. Mentor: Shri Kumbhar.

This repository studies the reasoning behaviour of a discrete-diffusion LLM
(`LLaDA-8B-Instruct`) against an autoregressive baseline (`Llama-3-8B-Instruct`)
under matched decoding compute. It is built on top of the
[dllm-reasoning/d1](https://github.com/dllm-reasoning/d1) codebase (vendored
under [`d1/`](d1/)) and is organized around the project's **three objectives**.

## What the project studies

We compare LLaDA (diffusion) and LLaMA (autoregressive) on math and
logical-reasoning benchmarks. Three objectives:

1. **Compute-parity benchmarking** — Accuracy@1, latency, and VRAM for Base
   and SFT variants of both model families on GSM8K, LogiQA, GPQA, AIME 2025
   (and auxiliary tasks: MATH-500, Countdown, Sudoku).
2. **Repairability** — base decode vs. self-consistency (k=5, T=0.7, p=0.9)
   vs. guided retry, plus repair-success and over-repair metrics.
3. **Mechanistic probing** — diffusion trajectories (entropy, token flips,
   stability, Flip Localization Index) and attention analysis
   (bidirectionality, entropy, ALI, quadrant flows).

## Repository layout

```text
.
├── objective1_benchmarking/        # Obj 1 — compute-parity evaluation (team's eval code; shared libs imported from d1)
│   ├── eval_sft_task2.py           #   main LLaDA eval harness (SFT + resume), used by run_eval_all.sh
│   ├── eval.py                     #   d1-style LLaDA eval entry (uses local generate.py)
│   ├── eval_gpqa_arrow.py          #   GPQA eval reading local Arrow shards
│   ├── generate.py                 #   LLaDA diffusion sampler — modified locally (NOT from d1)
│   ├── logiqa.py aime2025.py gpqa_diamond.py gpqa_diamond_2.py   #   dataset wrappers added by the team
│   ├── parse_boxed_accuracy.py     #   boxed-answer scoring (team)
│   ├── llada_parity_metrics_verbose.py run_llada_metrics_on_jsonl.py
│   │                               #   compute-parity / telemetry metrics
│   └── run_eval.sh run_eval_all.sh run_eval_all_gpqa.sh   #   drivers (paths resolve to repo root)
│   # gsm8k/countdown/math500/sudoku + parsers/parser_helper/parser_json/parse_and_get_acc
│   # are NOT duplicated here — imported from d1/eval/ via a sys.path shim in the entry scripts.
│
├── objective2_repairability/       # Obj 2 — second-chance decoding (self-contained notebook)
│   ├── LlaDA_repairability_sagar.ipynb   #   LLaDA base / self-consistency / guided-retry on GSM8K
│   └── README.md
│
├── objective3_probing/             # Obj 3 — mechanistic & attention probing
│   ├── attention_probes.py         #   wires imports of the probe package from d1/objective3/
│   └── README.md                   #   attention probes imported from d1/objective3/; mechanistic scripts pending
│
├── d1/                             # vendored upstream dllm-reasoning/d1 (SFT/, diffu-grpo/, eval/, objective3/)
│
├── results/                        # sweep outputs (gitignored)
├── eval_baselines/                 # LLaDA Base reference generations (gitignored)
└── sft_checkpoints/                # LoRA adapters (gitignored — distribute out-of-band)
```

**Shared code lives in `d1/`, not duplicated.** `d1/` is the vendored upstream
and is kept untouched. The objective folders import what they need from it:
`objective1_benchmarking/` keeps only the team's modified/added files
(`generate.py`, the eval harnesses, the extra dataset wrappers) and imports the
unchanged datasets/parsers from `d1/eval/`; `objective3_probing/` imports the
attention-probe package from `d1/objective3/`. Each entry script appends the
relevant `d1/` subdir to `sys.path`, and because the script's own directory is
`sys.path[0]`, local files (e.g. the modified `generate.py`) always take
precedence over their d1 counterparts.

## Objectives at a glance

| Objective | Folder | Status |
|---|---|---|
| 1 — Compute-parity benchmarking | [objective1_benchmarking/](objective1_benchmarking/) | Complete (LLaDA side) — runnable |
| 2 — Repairability | [objective2_repairability/](objective2_repairability/) | LLaDA notebook present; LLaMA repairability owned by a teammate (not here) |
| 3 — Mechanistic & attention probing | [objective3_probing/](objective3_probing/) | Attention probes in [d1/objective3/](d1/objective3/); mechanistic scripts owned by a teammate (not here) |

## Models

- **Base diffusion model:** `GSAI-ML/LLaDA-8B-Instruct`.
- **SFT variant:** LoRA adapter trained on a custom dataset mixture
  (`llada_mix_temp07`), final `checkpoint-942`. Lives under
  `sft_checkpoints/llada_mix_temp07/checkpoint-942/` (git-ignored due to size;
  obtain via the `.tgz` bundle shared separately). The SFT training code is in
  [d1/SFT/](d1/SFT/).

## Setup

Tested with Python 3.10 / CUDA. No `requirements.txt` is shipped; install the
dependencies the scripts import (or use d1's [env.yml](d1/env.yml)):

```bash
pip install torch transformers peft datasets accelerate tqdm numpy
# GPQA Arrow eval additionally needs: pyarrow
```

Multi-GPU runs use `torchrun` (bundled with PyTorch).

## Running — Objective 1 (compute-parity evaluation)

All drivers live in `objective1_benchmarking/` and resolve artifact paths to the
repo root (`results/`, `sft_checkpoints/`).

```bash
cd objective1_benchmarking

# Single-config DDP sweep over TASKS × GEN_LENGTHS (edit GPU ids):
bash run_eval.sh 0 1 2 3

# Step-budget sweep on one dataset (resume-safe). Edit the user settings
# block at the top of the script (DATASETS, STEPS, GENLEN, SFT_CKPT):
bash run_eval_all.sh            # resumes from JSONL line count / DONE markers
FORCE=1 bash run_eval_all.sh    # ignore DONE markers and re-run

# GPQA sweep (reads a local Arrow dataset directory):
bash run_eval_all_gpqa.sh
```

Outputs are written to `../results/eval_sweep_4datasets/{base,sft}/<dataset>/g<GENLEN>_s<STEPS>/`
as streamed `generations.rank0.jsonl` + a final `*_generations.json` bundle,
plus a `DONE` marker on success.

Compute-parity / telemetry metrics:

```bash
python llada_parity_metrics_verbose.py --root <dir-with-telemetry.rank*.jsonl> --out_dir <out>
python run_llada_metrics_on_jsonl.py --dataset_jsonl <val.jsonl> --output_dir <out>
```

## Running — Objective 2 (repairability)

Open [objective2_repairability/LlaDA_repairability_sagar.ipynb](objective2_repairability/LlaDA_repairability_sagar.ipynb).
It loads LLaDA, runs base / self-consistency / guided-retry decoding on GSM8K,
and computes repair-success and over-repair. See that folder's README for the
metric definitions.

## Objective 3 (probing)

Attention probes (Bidirectionality Index, entropy, ALI, quadrant flows) are a
package under [d1/objective3/](d1/objective3/); see
[objective3_probing/README.md](objective3_probing/README.md) for the file map
and findings. The diffusion-trajectory mechanistic scripts (`entropy_eval.py`,
`fli_eval.py`, `probe_eval.py`, `intervention_eval.py`) are owned by a teammate
and are not in this repo.

## Key results

From the team report (full numbers in the project write-up):

| Setting                  | GSM8K  | LogiQA | GPQA-diamond | AIME 2025 |
|--------------------------|:------:|:------:|:------------:|:---------:|
| LLaDA-SFT, 16 steps      | 24.6%  | 39.3%  | 22.6%        | 0.0%      |
| LLaDA-SFT, 32 steps      | 34.3%  | 41.9%  | 24.7%        | 0.0%      |
| LLaDA-SFT, 64 steps      | 56.1%  | 43.5%  | 24.8%        | 0.0%      |
| LLaDA-SFT, 256 steps     | 75.3%  | 47.4%  | 28.0%        | 0.0%      |
| LLaMA-SFT (greedy g128)  | 56%    | 40%    | —            | —         |

LLaDA accuracy scales monotonically with the diffusion step budget on all three
solvable benchmarks. AIME 2025 stays at 0% for every configuration (8B-scale
headroom).

## Notes for collaborators

- Heavy artifacts (LoRA checkpoints, generation dumps, telemetry JSONL,
  archives) are `.gitignore`-d — share via external storage / `.tgz` bundles.
- `results/`, `eval_baselines/`, `eval_results/` are output directories;
  recreate them by running the scripts. Do **not** commit their contents.
- `d1/` is a verbatim vendored copy of upstream d1 (its nested `.git` was
  removed so the parent repo tracks its files). Treat it as read-only upstream.

## References

LLaDA (Nie et al., 2025), GSM8K (Cobbe et al., 2021), LogiQA (Liu et al.,
2020), GPQA (Rein et al., 2023), self-consistency CoT (Wang et al., 2023),
chain-of-thought prompting (Wei et al., 2022). Full citations in the project
report.
