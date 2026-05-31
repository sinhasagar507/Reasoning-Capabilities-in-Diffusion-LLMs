# Objective 3 — Mechanistic & Attention Probing

This objective opens up *how* the diffusion model reasons, rather than just its
final accuracy. It has two halves: **diffusion-trajectory probing** (entropy,
flips, FLI, stability, linear probes, causal interventions) and **attention
probing** (bidirectionality, entropy, localization, quadrant flows).

## Attention probing — imported from [`../d1/objective3/`](../d1/objective3/)

The attention-map study is implemented as a package inside the vendored d1
checkout (it depends on d1's diffusion sampler in `d1/eval/generate.py`), so it
is kept there **untouched** and **imported** rather than duplicated.

[`attention_probes.py`](attention_probes.py) in this folder wires the import
paths (adds `d1/` and `d1/eval/` to `sys.path`) so the d1 probe package is
usable from here:

```python
import attention_probes                       # sets up sys.path
from objective3.probes import attention_metrics, llada_attention_utils
# or run the d1 attention eval directly:
#   python ../d1/objective3/scripts/llada_attention_eval_base.py
```

The probe modules it exposes:

| Path | Description |
|---|---|
| [../d1/objective3/README_attention_maps.md](../d1/objective3/README_attention_maps.md) | Design notes for the attention probes (BI, entropy, ALI). |
| [../d1/objective3/probes/attention_metrics.py](../d1/objective3/probes/attention_metrics.py) | Bidirectionality Index, attention entropy, ALI, quadrant flows. |
| [../d1/objective3/probes/llada_attention_utils.py](../d1/objective3/probes/llada_attention_utils.py) | Attention extraction / segment-mask helpers for LLaDA. |
| [../d1/objective3/sampling/attention_generate.py_](../d1/objective3/sampling/) | `generate_with_attention_probes` — diffusion sampler with `output_attentions`. |
| [../d1/objective3/scripts/llada_attention_eval_base.py](../d1/objective3/scripts/llada_attention_eval_base.py) | Dataset-level attention eval for LLaDA (base). |
| [../d1/objective3/scripts/inspect_models.py](../d1/objective3/scripts/inspect_models.py) | Dumps LLaDA / LLaMA module structure (`llada_summary.txt`, `llama_summary.txt`). |
| [../d1/objective3/notebooks/attention_map_study.ipynb](../d1/objective3/notebooks/attention_map_study.ipynb) | Attention heatmaps and per-layer metric plots. |

**Key findings (from the report):** LLaDA shows genuinely bidirectional
attention (BI ≈ 0.3–0.4) with a mid-layer prompt↔answer "reasoning corridor",
while LLaMA stays prompt-anchored (BI ≈ 0).

## Diffusion-trajectory probing — not in this repo (owned by teammate)

The mechanistic-probing scripts described in the report — `entropy_eval.py`,
`fli_eval.py`, `probe_eval.py`, `intervention_eval.py`,
`plot_probe_results.py` (built on top of the d1 codebase) — are **not**
included in this repository. They cover:

- **Entropy / flips** — mean predictive entropy and token flips per diffusion step.
- **FLI / stability** — Flip Localization Index and last-flip stability curves.
- **Linear probes** — logistic-regression correctness probes over (layer × step).
- **Causal interventions** — reinitializing hidden channels at steps {8, 16, 32}.

Drop those scripts into this folder when available; they are designed to run
against the same LLaDA decoder and GSM8K subset used in Objective 1.
