#!/usr/bin/env python3
import argparse, json, os, re, glob
from typing import Optional, List, Tuple, Dict, Iterable
from collections import defaultdict

NUM_RE = re.compile(
    r"^[^\d\-\+]*"                   # allow junk before number
    r"(?P<num>[+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+\-]?\d+)?)"  # signed float / sci
    r"[^\d]*$"
)

def almost_equal(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= max(tol, 1e-3 * max(abs(a), abs(b), 1.0))

def _boxed_spans(text: str) -> List[str]:
    out = []
    i = 0
    while True:
        i = text.find(r"\boxed{", i)
        if i < 0:
            break
        j = i + len(r"\boxed{")
        depth = 1
        start = j
        while j < len(text) and depth > 0:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        if depth == 0:
            out.append(text[start:j-1])
            i = j
        else:
            k = text.find("\n", start)
            out.append(text[start:(k if k != -1 else len(text))])
            break
    return out

def _extract_signed_number(s: str) -> Optional[float]:
    if s is None:
        return None
    s = s.replace(",", "").strip()
    s = s.strip("{} \t\r\n")
    try:
        return float(s)
    except Exception:
        pass
    m = re.search(r"[+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+\-]?\d+)?", s)
    if m:
        try:
            return float(m.group(0))
        except Exception:
            return None
    cleaned = re.sub(r"[^0-9eE\+\-\.]", " ", s).strip()
    m2 = re.search(r"[+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+\-]?\d+)?", cleaned)
    if m2:
        try:
            return float(m2.group(0))
        except Exception:
            return None
    return None

def extract_last_boxed_number(generation: str) -> Optional[float]:
    boxed_contents = _boxed_spans(generation or "")
    for content in reversed(boxed_contents):
        val = _extract_signed_number(content)
        if val is not None:
            return val
    return None

# ---- new helper: extract letter answers (A-D) ----
def extract_letter_from_text(s: str) -> Optional[str]:
    """
    Robustly find an answer letter (A-D) in the generation.
    Prefers boxed forms like \boxed{A}, then <answer>...</answer>, then 'Final: A', then last-line isolated letter.
    Returns 'A'..'D' or None.
    """
    if not s:
        return None
    # 1) boxed \boxed{A}
    m = re.search(r"\\boxed\{\s*([A-D])\s*\}", s, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # 2) <answer> ... </answer>
    m = re.search(r"<answer>.*?([A-D]).*?</answer>", s, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).upper()
    # 3) Final: A  or Final: <A>
    m = re.search(r"final[:\s]*\<?\s*([A-D])\s*\>?", s, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # 4) isolated letter near end (last 4 lines)
    for line in (s.splitlines()[-4:]):
        line = line.strip()
        if len(line) == 1 and line.upper() in "ABCD":
            return line.upper()
        m = re.match(r"^([A-D])[\)\.\-\:]*\s*$", line, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()
    # 5) fallback: first standalone token A/B/C/D
    m = re.search(r"\b([A-D])\b", s, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return None

def score_file(path: str) -> Tuple[int, int, int, int, int]:
    """
    Returns (correct, total, num_scored_numeric, num_scored_mcq, num_skipped) for a single *_generations.json file.

    Handles:
      - numeric problems: compares last boxed numeric to ground-truth numeric
      - MCQ letter problems: compares extracted letter (A-D) to ground-truth letter
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    gens = data.get("generations", [])
    correct = 0
    total = 0
    scored_numeric = 0
    scored_mcq = 0
    skipped = 0

    for item in gens:
        gen_text = item.get("generations", "") or ""
        gt = item.get("ground_truth", None)

        # 1) try numeric ground-truth (prefer numeric if GT contains digits)
        gt_val = None
        try:
            if gt is not None and (isinstance(gt, (int, float)) or (isinstance(gt, str) and re.search(r"[0-9]", gt))):
                gt_val = float(gt)
        except Exception:
            gt_val = None

        if gt_val is not None:
            # numeric scoring path
            pred = extract_last_boxed_number(gen_text)
            if pred is not None and almost_equal(pred, gt_val):
                correct += 1
            total += 1
            scored_numeric += 1
            continue

        # 2) non-numeric -> try MCQ letter scoring
        gt_letter = None
        if isinstance(gt, str) and gt.strip():
            gts = gt.strip().upper()
            if len(gts) == 1 and gts in "ABCD":
                gt_letter = gts
            else:
                m = re.search(r"([A-D])", gts, flags=re.IGNORECASE)
                if m:
                    gt_letter = m.group(1).upper()
        else:
            # try alternate fields often present in MCQ JSON (check item)
            for alt in ("correct_option", "gold", "correct"):
                if alt in item and item[alt] is not None:
                    try:
                        val = int(item[alt])
                        if 0 <= val < 4:
                            gt_letter = "ABCD"[val]
                            break
                    except Exception:
                        if isinstance(item[alt], str):
                            m = re.search(r"([A-D])", item[alt], flags=re.IGNORECASE)
                            if m:
                                gt_letter = m.group(1).upper()
                                break

        if gt_letter is not None:
            pred_letter = extract_letter_from_text(gen_text)
            if pred_letter is not None and pred_letter.upper() == gt_letter.upper():
                correct += 1
            total += 1
            scored_mcq += 1
            continue

        # 3) couldn't interpret GT -> skip
        skipped += 1
        continue

    return correct, total, scored_numeric, scored_mcq, skipped

# ----- path helpers -----
def find_generation_files(root: str) -> List[str]:
    if os.path.isdir(root):
        files = glob.glob(os.path.join(root, "**", "*_generations.json"), recursive=True)
        files.sort()
        return files
    if os.path.isfile(root):
        return [root]
    return []

def detect_dataset_from_path(path: str, candidates: Optional[Iterable[str]] = None) -> Optional[str]:
    parts = [p for p in os.path.normpath(path).split(os.sep) if p]
    if candidates:
        for c in candidates:
            if c in parts:
                return c
    # common dataset tokens:
    tokens = ["gsm8k", "math", "countdown", "sudoku", "aime2024", "logiqa", "gpqa", "gpqa_ext", "gpqa_diamond"]
    for t in tokens:
        if t in parts:
            return t
    # fallback: try to extract dataset by pattern like ".../{variant}/{dataset}/..."
    for p in reversed(parts[-6:]):
        if re.match(r"^[a-zA-Z0-9_\-]+$", p) and len(p) <= 40:
            return p
    return None

def detect_variant_from_path(path: str) -> str:
    parts = [p for p in os.path.normpath(path).split(os.sep) if p]
    for v in ("base", "sft"):
        if v in parts:
            return v
    return "unknown"

# ----- main aggregator -----
def main():
    ap = argparse.ArgumentParser(description="Score numeric boxed answers across many generation JSON files and aggregate by dataset/variant.")
    ap.add_argument("--in", dest="inp", required=True,
                    help="Path to a file or a directory containing *_generations.json (recursively).")
    ap.add_argument("--datasets", dest="datasets", default=None,
                    help="Optional comma-separated list of dataset tokens to include (e.g. gsm8k,logiqa,gpqa_diamond). If omitted the script will try to auto-detect.")
    ap.add_argument("--min-files-per-dataset", type=int, default=0,
                    help="If >0, warn when a detected dataset has fewer files than this (useful to catch missing variants).")
    args = ap.parse_args()

    candidates = None
    if args.datasets:
        candidates = [s.strip() for s in args.datasets.split(",") if s.strip()]

    paths = find_generation_files(args.inp)
    if not paths:
        print(f"[warn] no *_generations.json under {args.inp}")
        return

    per_file_results: List[Tuple[str,int,int]] = []
    per_dataset_counts: Dict[str, Tuple[int,int]] = defaultdict(lambda: (0,0))
    per_variant_counts: Dict[str, Tuple[int,int]] = defaultdict(lambda: (0,0))
    per_dataset_variant: Dict[Tuple[str,str], List[Tuple[str,int,int]]] = defaultdict(list)

    grand_c = 0
    grand_n = 0

    print("===== PER-FILE RESULTS =====")
    for p in paths:
        ds = detect_dataset_from_path(p, candidates)
        if candidates and (ds not in candidates):
            continue

        try:
            c, n, n_num, n_mcq, n_skipped = score_file(p)
        except Exception as e:
            print(f"{p}: ERROR while scoring: {e}")
            continue

        per_file_results.append((p,c,n))
        variant = detect_variant_from_path(p)
        ds_key = ds or "unknown"

        # accumulate per-dataset
        prev_c, prev_n = per_dataset_counts[ds_key]
        per_dataset_counts[ds_key] = (prev_c + c, prev_n + n)

        # accumulate per-variant
        pv_c, pv_n = per_variant_counts[variant]
        per_variant_counts[variant] = (pv_c + c, pv_n + n)

        # store per-dataset+variant detailed files
        per_dataset_variant[(ds_key, variant)].append((p,c,n))

        # accumulate global
        grand_c += c
        grand_n += n

        # print per-file with numeric/mcq/skip counts
        acc = (100.0 * c / n) if n else 0.0
        print(f"{p}: {c}/{n}  ({acc:.2f}%)  [dataset={ds_key} variant={variant} numeric={n_num} mcq={n_mcq} skipped={n_skipped}]")

    # Print per-dataset summary
    print("\n===== PER-DATASET SUMMARY =====")
    for ds, (c_sum, n_sum) in sorted(per_dataset_counts.items(), key=lambda x: x[0]):
        acc = (100.0 * c_sum / n_sum) if n_sum else 0.0
        print(f"{ds}: {c_sum}/{n_sum}  ({acc:.2f}%)")

    # Print per-variant summary
    print("\n===== PER-VARIANT SUMMARY =====")
    for variant, (c_sum, n_sum) in sorted(per_variant_counts.items(), key=lambda x: x[0]):
        acc = (100.0 * c_sum / n_sum) if n_sum else 0.0
        print(f"{variant}: {c_sum}/{n_sum}  ({acc:.2f}%)")

    # Optionally warn about datasets with few files
    if args.min_files_per_dataset > 0:
        ds_file_counts: Dict[str,int] = {}
        for (ds, var), files in per_dataset_variant.items():
            ds_file_counts.setdefault(ds, 0)
            ds_file_counts[ds] += len(files)
        for ds, cnt in ds_file_counts.items():
            if cnt < args.min_files_per_dataset:
                print(f"[warn] dataset {ds} only had {cnt} generation files (less than --min-files-per-dataset)")

    # Global aggregate
    print("\n===== AGGREGATE =====")
    total_acc = (100.0 * grand_c / grand_n) if grand_n else 0.0
    print(f"TOTAL: {grand_c}/{grand_n}  ({total_acc:.2f}%)")

if __name__ == "__main__":
    main()
