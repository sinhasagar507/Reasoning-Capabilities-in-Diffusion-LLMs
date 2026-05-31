"""Objective 3 — attention probing entry point.

The probe implementations (Bidirectionality Index, attention entropy, ALI,
quadrant flows, the attention-logging diffusion sampler) live in the vendored
upstream at ``d1/objective3/`` and are **imported** here rather than
duplicated. This module only wires the import paths so the d1 probe package is
usable from ``objective3_probing/``.

Usage (from the repo root or this folder)::

    import attention_probes                      # sets up sys.path
    from objective3.probes import attention_metrics, llada_attention_utils
    # ... or run the d1 attention eval directly:
    #   python ../d1/objective3/scripts/llada_attention_eval_base.py
"""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# d1/ on path so the `objective3` probe package (d1/objective3) is importable;
# d1/eval on path for the shared diffusion sampler the probes reuse.
for _p in (os.path.join(_REPO_ROOT, "d1"), os.path.join(_REPO_ROOT, "d1", "eval")):
    if _p not in sys.path:
        sys.path.append(_p)

# Re-export the probe modules so callers can `from attention_probes import ...`.
from objective3.probes import attention_metrics, llada_attention_utils  # noqa: E402,F401

__all__ = ["attention_metrics", "llada_attention_utils"]
