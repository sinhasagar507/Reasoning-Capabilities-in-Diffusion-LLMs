
# Objective 3 — Attention Map Study (LLaDA vs LLaMA)

This sub-folder contains code to study **attention dynamics** in diffusion-based LLaDA
vs autoregressive LLaMA.

We build directly on the mechanistic probing probes already implemented for diffusion
trajectories (entropy, flips, FLI, stability, flip heatmaps). Those probes operate on
token trajectories x^(s); here we add probes that operate on **attention maps** A^(s).

## Goals

1. Inspect the model architectures for:
   - LLaDA-8B-Instruct (bidirectional diffusion)
   - LLaMA-3(.1)-8B-Instruct (causal autoregressive)

2. Define lightweight attention probes:
   - Bidirectionality Index (BI): how much attention flows to future tokens.
   - Attention entropy: how focused vs diffuse attention is over positions.
   - Attention Localization Index (ALI): how spatially localized attention mass is over
     the generated answer region (FLI-analogue).

3. Add a `generate_with_attention_probes` sampler that:
   - Reuses the existing diffusion sampler logic (`eval/generate.py`).
   - At each diffusion step, calls the model with `output_attentions=True`.
   - Computes BI / entropy / ALI on one or more layers.
   - Logs trajectories over diffusion steps: [S, B] tensors.

Later steps (not implemented yet here) will:
- Add dataset-level evaluation scripts for LLaDA and LLaMA.
- Plot trajectories, heatmaps, and compare BASE vs SFT models.

# Objective 3 — Attention Map Study (LLaDA vs LLaMA)

This sub-folder contains code to study **attention dynamics** in diffusion-based LLaDA
vs autoregressive LLaMA.

We build directly on the mechanistic probing probes already implemented for diffusion
trajectories (entropy, flips, FLI, stability, flip heatmaps). Those probes operate on
token trajectories x^(s); here we add probes that operate on **attention maps** A^(s).

## Goals

1. Inspect the model architectures for:
   - LLaDA-8B-Instruct (bidirectional diffusion)
   - LLaMA-3(.1)-8B-Instruct (causal autoregressive)

2. Define lightweight attention probes:
   - Bidirectionality Index (BI): how much attention flows to future tokens.
   - Attention entropy: how focused vs diffuse attention is over positions.
   - Attention Localization Index (ALI): how spatially localized attention mass is over
     the generated answer region (FLI-analogue).

3. Add a `generate_with_attention_probes` sampler that:
   - Reuses the existing diffusion sampler logic (`eval/generate.py`).
   - At each diffusion step, calls the model with `output_attentions=True`.
   - Computes BI / entropy / ALI on one or more layers.
   - Logs trajectories over diffusion steps: [S, B] tensors.

Later steps (not implemented yet here) will:
- Add dataset-level evaluation scripts for LLaDA and LLaMA.
- Plot trajectories, heatmaps, and compare BASE vs SFT models.
