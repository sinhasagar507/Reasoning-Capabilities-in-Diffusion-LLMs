#!/usr/bin/env python3
"""
llada_parity_metrics_verbose.py

Loads telemetry JSONL files (telemetry.rank*.jsonl), validates them,
computes per-example and per-(family,setting,dataset) aggregates,
computes compute-parity and inference-parity pairings, and logs every
calculation step to a detailed run log.

Outputs:
 - <out_dir>/per_example.csv
 - <out_dir>/aggregated_by_family_setting_dataset.csv
 - <out_dir>/compute_parity_pairs.csv
 - <out_dir>/inference_parity_pairs.csv
 - <out_dir>/pairs_verbose.csv (detailed diagnostics for every AR->LLaDA candidate pair)
 - <out_dir>/summary.json

Usage example:
  python eval/llada_parity_metrics_verbose.py --root /path/to/eval/results \
      --out_dir /path/to/outdir --log_level INFO

Notes:
 - Requires pandas, numpy (standard in your environment)
 - Telemetry JSONL lines should include at least:
     family (str), setting (str), dataset (str),
     latency_ms_total (float), gpu_seconds (float), peak_mem_gb (float)
   tokens_in / tokens_out optional but included if present.
"""
import argparse
import glob
import json
import logging
import os
from datetime import datetime
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd

# -------------------------
# Helpers
# -------------------------
def setup_logger(logpath: str, level: str = "INFO"):
    os.makedirs(os.path.dirname(logpath), exist_ok=True)
    logger = logging.getLogger("llada_parity_verbose")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    # file handler
    fh = logging.FileHandler(logpath, mode="w", encoding="utf-8")
    fh.setLevel(getattr(logging, level.upper(), logging.INFO))
    fmt = logging.Formatter("%(asctime)s | %(levelname)-5s | %(message)s")
    fh.setFormatter(fmt)
    logger.handlers = []  # reset
    logger.addHandler(fh)
    # console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger

def find_telemetry_files(root: str) -> List[str]:
    patterns = [
        os.path.join(root, "**", "telemetry.rank*.jsonl"),
        os.path.join(root, "telemetry.rank*.jsonl"),
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p, recursive=True))
    files = sorted(set(files))
    return files

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                # tolerate malformed trailing line but log
                logger.warning("Skipping malformed JSON line %d in %s (%s)", i+1, path, str(e))
                continue
    return rows

def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")

def ensure_columns(df: pd.DataFrame, cols: List[str], fill=None) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = fill
    return df

# -------------------------
# Aggregation & Pairing logic
# -------------------------
def aggregate_df(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    numeric_cols = ["latency_ms_total", "gpu_seconds", "peak_mem_gb", "tokens_in", "tokens_out"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = safe_numeric(df[c])

    # Fill missing dataset with "unknown"
    df["dataset"] = df.get("dataset", pd.Series(["unknown"] * len(df), index=df.index))
    df["family"] = df.get("family", pd.Series(["unknown"] * len(df), index=df.index))
    df["setting"] = df.get("setting", pd.Series(["unknown"] * len(df), index=df.index))

    group_cols = ["family", "setting", "dataset"]
    logger.info("Aggregating telemetry by %s", group_cols)
    agg = df.groupby(group_cols).agg(
        n_examples=("latency_ms_total", "size"),
        med_gpu_seconds=("gpu_seconds", lambda x: float(np.nanmedian(x)) if x.dropna().size>0 else float("nan")),
        mean_gpu_seconds=("gpu_seconds", lambda x: float(np.nanmean(x)) if x.dropna().size>0 else float("nan")),
        p50_latency_ms=("latency_ms_total", lambda x: float(np.nanmedian(x)) if x.dropna().size>0 else float("nan")),
        p95_latency_ms=("latency_ms_total", lambda x: float(np.nanpercentile(x.dropna(), 95)) if x.dropna().size>0 else float("nan")),
        med_peak_mem_gb=("peak_mem_gb", lambda x: float(np.nanmedian(x)) if x.dropna().size>0 else float("nan")),
        mean_tokens_in=("tokens_in", lambda x: float(np.nanmean(x)) if x.dropna().size>0 else float("nan")),
        mean_tokens_out=("tokens_out", lambda x: float(np.nanmean(x)) if x.dropna().size>0 else float("nan")),
    ).reset_index()
    logger.info("Aggregation complete: %d rows", len(agg))
    return agg

def compute_pairwise_candidates(agg: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    For each AR row, compute candidate pairing metrics with every LLaDA row.
    Returns a verbose DataFrame with one row per (AR_row, LLaDA_row) pair with distances.
    """
    ar = agg[agg["family"] == "AR"].reset_index(drop=True)
    ll = agg[agg["family"] == "LLaDA"].reset_index(drop=True)
    if ar.empty:
        logger.warning("No AR rows found in aggregation.")
    if ll.empty:
        logger.warning("No LLaDA rows found in aggregation.")
    rows = []
    for _, a in ar.iterrows():
        for _, l in ll.iterrows():
            # compute difference metrics
            try:
                a_gpu = float(a["med_gpu_seconds"])
                l_gpu = float(l["med_gpu_seconds"])
            except Exception:
                a_gpu = np.nan
                l_gpu = np.nan
            try:
                a_lat = float(a["p50_latency_ms"])
                l_lat = float(l["p50_latency_ms"])
            except Exception:
                a_lat = np.nan
                l_lat = np.nan

            rows.append({
                "ar_setting": a["setting"],
                "ar_dataset": a["dataset"],
                "ar_med_gpu_s": a_gpu,
                "ar_p50_ms": a_lat,
                "llada_setting": l["setting"],
                "llada_dataset": l["dataset"],
                "llada_med_gpu_s": l_gpu,
                "llada_p50_ms": l_lat,
                "abs_diff_gpu_s": None if np.isnan(a_gpu) or np.isnan(l_gpu) else abs(a_gpu - l_gpu),
                "abs_diff_p50_ms": None if np.isnan(a_lat) or np.isnan(l_lat) else abs(a_lat - l_lat),
            })
    logger.info("Built %d candidate pairs (AR x LLaDA)", len(rows))
    return pd.DataFrame(rows)

def choose_compute_parity_pairs(verbose_pairs: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    For each unique AR setting, pick the LLaDA setting with minimal abs_diff_gpu_s.
    """
    if verbose_pairs.empty:
        logger.warning("No verbose pairs to choose compute parity from.")
        return pd.DataFrame()
    # drop pairs where gpu diff is NaN
    cand = verbose_pairs.dropna(subset=["abs_diff_gpu_s"]).copy()
    chosen = cand.loc[cand.groupby(["ar_setting"])["abs_diff_gpu_s"].idxmin()].reset_index(drop=True)
    logger.info("Selected %d compute-parity pairs (one per AR setting)", len(chosen))
    return chosen.sort_values(["ar_med_gpu_s"])

def choose_inference_parity_pairs(agg: pd.DataFrame, verbose_pairs: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    For each AR setting (used as a latency cap = AR.p50), choose LLaDA setting whose p50 <= cap.
    If many under cap, pick the one with largest med_gpu_seconds (closest to compute frontier).
    If none under cap, pick the LLaDA setting with smallest abs diff in p50 latency.
    """
    ar = agg[agg["family"] == "AR"].reset_index(drop=True)
    ll = agg[agg["family"] == "LLaDA"].reset_index(drop=True)
    if ar.empty or ll.empty:
        logger.warning("Need both AR and LLaDA rows for inference parity.")
        return pd.DataFrame()

    pairs = []
    for _, a in ar.iterrows():
        cap = float(a["p50_latency_ms"])
        under = ll[ll["p50_latency_ms"].astype(float) <= cap]
        if not under.empty:
            # pick highest med_gpu_seconds among those under cap (closest to compute frontier)
            chosen = under.loc[under["med_gpu_seconds"].astype(float).idxmax()]
            reason = "under_cap_max_gpu"
        else:
            # pick closest by p50 latency
            ll["lat_diff"] = np.abs(ll["p50_latency_ms"].astype(float) - cap)
            chosen = ll.loc[ll["lat_diff"].idxmin()]
            reason = "closest_by_latency"
        pairs.append({
            "ar_setting": a["setting"],
            "ar_dataset": a["dataset"],
            "ar_p50_ms": float(a["p50_latency_ms"]),
            "llada_setting": chosen["setting"],
            "llada_dataset": chosen["dataset"],
            "llada_p50_ms": float(chosen["p50_latency_ms"]),
            "llada_med_gpu_s": float(chosen["med_gpu_seconds"]),
            "selection_reason": reason,
        })
    logger.info("Selected %d inference-parity pairs (one per AR setting)", len(pairs))
    return pd.DataFrame(pairs)

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="root directory containing telemetry.rank*.jsonl")
    ap.add_argument("--out_dir", required=True, help="directory to write CSV/JSON outputs and log")
    ap.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    ap.add_argument("--show_top", type=int, default=50, help="how many aggregated rows to print")
    args = ap.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.out_dir, exist_ok=True)
    logpath = os.path.join(args.out_dir, f"llada_parity_verbose_{timestamp}.log")
    global logger
    logger = setup_logger(logpath, level=args.log_level)
    logger.info("Starting llada_parity_metrics_verbose.py")
    logger.info("Telemetry root: %s", args.root)
    # Note about the uploaded notebook (user-supplied)
    uploaded_notebook_path = "/mnt/data/llama_llada_logiqa_parity_colab_1.ipynb - Colab.pdf"
    logger.info("User notebook (uploaded) available at: %s", uploaded_notebook_path)

    files = find_telemetry_files(args.root)
    logger.info("Found %d telemetry files", len(files))
    if not files:
        logger.error("No telemetry files found under %s. Looked for telemetry.rank*.jsonl", args.root)
        return

    # load all rows
    rows = []
    for f in files:
        logger.info("Loading %s", f)
        r = load_jsonl(f)
        logger.info("  -> %d lines parsed", len(r))
        for rec in r:
            # attach source file for traceability
            rec["_source_file"] = f
            rows.append(rec)
    if not rows:
        logger.error("No telemetry rows found after loading files.")
        return

    df = pd.DataFrame(rows)
    logger.info("Total telemetry rows loaded: %d", len(df))

    # ensure minimal columns exist
    required = ["family", "setting", "dataset", "latency_ms_total", "gpu_seconds", "peak_mem_gb"]
    df = ensure_columns(df, required, fill=None)
    # save per-example table
    per_example_path = os.path.join(args.out_dir, f"per_example_{timestamp}.csv")
    df.to_csv(per_example_path, index=False)
    logger.info("Wrote per-example CSV -> %s", per_example_path)

    # aggregated
    agg = aggregate_df(df, logger)
    agg_path = os.path.join(args.out_dir, f"aggregated_by_family_setting_dataset_{timestamp}.csv")
    agg.to_csv(agg_path, index=False)
    logger.info("Wrote aggregated CSV -> %s (rows=%d)", agg_path, len(agg))

    # compute diagnostic pairs (verbose)
    verbose_pairs = compute_pairwise_candidates(agg, logger)
    verbose_pairs_path = os.path.join(args.out_dir, f"pairs_verbose_{timestamp}.csv")
    if not verbose_pairs.empty:
        verbose_pairs.to_csv(verbose_pairs_path, index=False)
        logger.info("Wrote verbose pairs CSV -> %s", verbose_pairs_path)
    else:
        logger.warning("Verbose pairs DataFrame empty -- skipping write")

    # compute parity selections
    compute_pairs = choose_compute_parity_pairs(verbose_pairs, logger)
    compute_pairs_path = os.path.join(args.out_dir, f"compute_parity_pairs_{timestamp}.csv")
    if not compute_pairs.empty:
        compute_pairs.to_csv(compute_pairs_path, index=False)
        logger.info("Wrote compute-parity CSV -> %s", compute_pairs_path)
    else:
        logger.warning("Compute parity pairs empty; check presence of AR and LLaDA in aggregated table.")

    inference_pairs = choose_inference_parity_pairs(agg, verbose_pairs, logger)
    inference_pairs_path = os.path.join(args.out_dir, f"inference_parity_pairs_{timestamp}.csv")
    if not inference_pairs.empty:
        inference_pairs.to_csv(inference_pairs_path, index=False)
        logger.info("Wrote inference-parity CSV -> %s", inference_pairs_path)
    else:
        logger.warning("Inference parity pairs empty; check presence of AR and LLaDA in aggregated table.")

    # Save JSON summary
    summary = {
        "timestamp": timestamp,
        "telemetry_files_count": len(files),
        "per_example_rows": len(df),
        "aggregated_rows": len(agg),
        "compute_pairs": len(compute_pairs),
        "inference_pairs": len(inference_pairs),
        "per_example_csv": per_example_path,
        "aggregated_csv": agg_path,
        "verbose_pairs_csv": verbose_pairs_path if not verbose_pairs.empty else None,
        "compute_pairs_csv": compute_pairs_path if not compute_pairs.empty else None,
        "inference_pairs_csv": inference_pairs_path if not inference_pairs.empty else None,
        "logfile": logpath,
        "uploaded_notebook_path": uploaded_notebook_path
    }
    summary_path = os.path.join(args.out_dir, f"summary_{timestamp}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Wrote summary JSON -> %s", summary_path)

    # print a short table for convenience
    logger.info("Top aggregated rows (sample):")
    try:
        print(agg.sort_values(["family", "med_gpu_seconds"]).head(args.show_top).to_string(index=False))
    except Exception:
        logger.info("Could not pretty-print aggregated DataFrame to console.")

    logger.info("Done. Detailed log is at %s", logpath)

if __name__ == "__main__":
    main()