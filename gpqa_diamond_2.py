#!/usr/bin/env python3
"""
gpqa_diamond.py

GPQADiamondDataset: dataset wrapper for GPQA diamond/main that mirrors the GSM8K dataset class API.

New:
 - `custom_eval_path`: use a local JSON/JSONL file as the eval set
 - `custom_eval_arrow`: use a local HF Arrow dataset (directory with dataset_info.json or a .arrow shard)
Both routes keep the HF mirror only for few-shot pool construction to avoid eval leakage.
"""
import json
import os
import random
import re
from typing import List, Optional, Set, Tuple

from datasets import load_dataset, load_from_disk, Dataset
import torch

# System prompt and helpers
GPQA_SYSTEM_PROMPT = """You are a careful scientific reasoner.
Choose the correct option (A–D). Think step by step, then put ONLY the final letter inside \\boxed{<LETTER>}.
Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
\\boxed{A|B|C|D}
</answer>"""

LETTERS = "ABCD"
NORM_RE = re.compile(r"\s+")


def norm_text(s: str) -> str:
    return NORM_RE.sub(" ", (s or "").strip()).lower()


def load_gpqa_diamond_prefer_mirrors():
    """
    Try a sequence of HF dataset ids to load a diamond split.
    Returns (dataset_split, chosen_split_name)
    """
    tries = [
        ("Idavidrein/gpqa", "diamond"),
        ("Wanfq/gpqa", "diamond"),
        ("sayantan0013/gpqa_extended", "diamond"),
        ("Idavidrein/gpqa", "main"),
        ("Wanfq/gpqa", "main"),
        ("sayantan0013/gpqa_extended", "validation"),
        ("sayantan0013/gpqa_extended", "test"),
        ("sayantan0013/gpqa_extended", "all"),
    ]
    for ds, split in tries:
        try:
            d = load_dataset(ds, split=split)
            print(f"[gpqa_loader] Loaded {ds} split={split} len={len(d)}")
            return d, f"{ds}:{split}"
        except Exception:
            continue
    # try split=None for dicts and pick diamond/main if present
    mirrors = ["Idavidrein/gpqa", "Wanfq/gpqa", "sayantan0013/gpqa_extended"]
    for ds in mirrors:
        try:
            dd = load_dataset(ds, split=None)
            if isinstance(dd, dict):
                for pref in ("diamond", "main", "test", "validation", "train"):
                    if pref in dd:
                        print(f"[gpqa_loader] Loaded {ds} split={pref} len={len(dd[pref])}")
                        return dd[pref], f"{ds}:{pref}"
                first = next(iter(dd.keys()))
                print(f"[gpqa_loader] Loaded {ds} first split={first} len={len(dd[first])}")
                return dd[first], f"{ds}:{first}"
        except Exception:
            continue
    raise RuntimeError("Could not load any GPQA diamond/main split from known mirrors. Please install HF datasets and ensure internet access.")


def collect_train_gpqa_ids_texts(train_jsonl_path: str) -> Tuple[Set[str], Set[str]]:
    ids = set()
    texts = set()
    if not train_jsonl_path or not os.path.exists(train_jsonl_path):
        return ids, texts
    with open(train_jsonl_path, "r", encoding="utf-8") as f:
        for ln in f:
            try:
                j = json.loads(ln)
            except Exception:
                continue
            src = j.get("source", "")
            # permissive checking for gpqa source tags
            if src not in ("gpqa_ext", "gpqa", "gpqa_ext_train", "gpqa_ext_train.jsonl", "gpqa_ext_train.json"):
                meta = j.get("meta") or {}
                if meta and meta.get("source") not in ("gpqa_ext", "gpqa"):
                    continue
            # collect id if present in meta
            mid = None
            meta = j.get("meta") or {}
            for k in ("id", "problem_id", "qid"):
                if meta.get(k) is not None:
                    mid = str(meta.get(k))
                    break
            if mid:
                ids.add(mid)
            # extract question from prompt if possible
            prompt = j.get("prompt", "")
            q = None
            if "Question:" in prompt:
                try:
                    q = prompt.split("Question:")[-1].split("Options:")[0]
                except Exception:
                    q = prompt
            else:
                q = prompt
            nq = norm_text(q)
            if nq:
                texts.add(nq)
    return ids, texts


def decontaminate(hf_split, train_ids: Set[str], train_texts: Set[str]):
    kept, removed = [], []
    for ex in hf_split:
        hid = None
        for k in ("id", "problem_id", "qid"):
            if ex.get(k) is not None:
                hid = str(ex.get(k))
                break
        q = ex.get("question") or ex.get("context") or ex.get("prompt") or ""
        nq = norm_text(q)
        if hid is not None and hid in train_ids:
            removed.append((ex, "id"))
        elif nq and nq in train_texts:
            removed.append((ex, "text"))
        else:
            kept.append(ex)
    return kept, removed


# -------- New loaders for custom eval sets --------
def load_custom_eval(path: str) -> list[dict]:
    """
    Load a local JSON/JSONL custom eval set.
    Expected keys (flexible): question|context|prompt, choices|options (list or dict), answer|label|correct.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    def _coerce_rec(r: dict) -> dict:
        q = r.get("question") or r.get("context") or r.get("prompt") or ""
        ch = r.get("choices") or r.get("options") or r.get("alts") or {}
        # normalize choices to list [A,B,C,D]
        if isinstance(ch, dict):
            ch = [ch.get("A", ""), ch.get("B", ""), ch.get("C", ""), ch.get("D", "")]
        else:
            ch = list(ch)[:4] + ["", "", "", ""]
            ch = ch[:4]
        lab = r.get("answer") or r.get("label") or r.get("correct") or ""
        if isinstance(lab, (int, float)):
            try:
                lab = LETTERS[int(lab)]
            except Exception:
                lab = ""
        elif isinstance(lab, str) and lab.isdigit():
            try:
                lab = LETTERS[int(lab)]
            except Exception:
                lab = lab
        lab = (lab or "").strip().upper()
        return {
            "id": str(r.get("id")) if r.get("id") is not None else None,
            "question": q,
            "choices": ch[:4],
            "answer": lab,
        }

    rows = []
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    rows.append(_coerce_rec(json.loads(ln)))
                except Exception:
                    continue
    else:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict) and "data" in obj:
            obj = obj["data"]
        if not isinstance(obj, list):
            raise ValueError("Custom eval JSON must be a list or JSONL.")
        for r in obj:
            rows.append(_coerce_rec(r))
    return rows


def load_custom_eval_arrow(path: str) -> list[dict]:
    """
    Load a local HF Arrow dataset (saved-to-disk directory with dataset_info.json OR a single .arrow shard).
    """
    if os.path.isdir(path):
        ds = load_from_disk(path)
    elif path.endswith(".arrow"):
        ds = Dataset.from_file(path)
    else:
        raise FileNotFoundError(f"Not a directory or .arrow file: {path}")

    rows = []
    for ex in ds:
        q = ex.get("question") or ex.get("context") or ex.get("prompt") or ""
        ch = ex.get("choices") or ex.get("options") or {}
        if isinstance(ch, dict):
            ch = [ch.get("A", ""), ch.get("B", ""), ch.get("C", ""), ch.get("D", "")]
        else:
            ch = list(ch)[:4] + ["", "", "", ""]
            ch = ch[:4]
        lab = ex.get("answer") or ex.get("label") or ex.get("correct")
        if isinstance(lab, (int, float)):
            lab = LETTERS[int(lab)] if 0 <= int(lab) <= 3 else ""
        elif isinstance(lab, str) and lab.isdigit():
            lab = LETTERS[int(lab)] if 0 <= int(lab) <= 3 else lab.strip().upper()
        else:
            lab = (lab or "").strip().upper()
        rows.append(
            {
                "id": str(ex.get("id")) if ex.get("id") is not None else None,
                "question": q,
                "choices": ch[:4],
                "answer": lab,
            }
        )
    return rows
# -------------------------------------------------


def build_prompt(question: str, choices: List[str], few_shot_prompt: Optional[str] = None, add_reasoning: bool = True):
    opts = "\n".join([f"{LETTERS[i]}) {choices[i]}" for i in range(min(4, len(choices)))])
    body = ""
    if few_shot_prompt:
        body += few_shot_prompt.strip() + "\n\n"
    body += f"Question:\n{question.strip()}\n\nOptions:\n{opts}\n"
    if few_shot_prompt:
        body += "Answer:\n"
    msg = GPQA_SYSTEM_PROMPT + "\n\n" + body
    return msg + "<reasoning>" if add_reasoning else msg


class GPQADiamondArrowDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper for GPQA diamond/main split.

    Parameters
    ----------
    tokenizer : tokenizer instance with .apply_chat_template(...) and callable
    num_examples : int
        number of few-shot examples to include (drawn from fewshot_pool)
    add_reasoning : bool
        whether to append "<reasoning>" generation token
    train_jsonl : Optional[str]
        path to SFT train.jsonl to decontaminate against (optional)
    subsample : int
        -1 means use all examples; otherwise pick exactly this many examples (random)
    seed : int
        RNG seed for deterministic sampling
    custom_eval_path : Optional[str]
        path to local JSON/JSONL eval set
    custom_eval_arrow : Optional[str]
        path to local Arrow dataset (dir with dataset_info.json OR a .arrow file)
    """

    def __init__(
        self,
        tokenizer,
        num_examples: int = 0,
        add_reasoning: bool = True,
        train_jsonl: Optional[str] = None,
        subsample: int = -1,
        seed: int = 23,
        custom_eval_path: Optional[str] = None,
        custom_eval_arrow: Optional[str] = None,
    ):
        self.tokenizer = tokenizer
        self.num_examples = int(num_examples)
        self.add_reasoning = bool(add_reasoning)
        self.train_jsonl = train_jsonl
        self.seed = int(seed)
        self.subsample = int(subsample)

        # Load HF GPQA split (kept for few-shot pool and for fallback)
        hf_split, chosen = load_gpqa_diamond_prefer_mirrors()
        self.chosen_split = chosen
        self.raw_hf = hf_split  # keep for debugging

        # Collect SFT train ids/texts for decontamination (optional)
        train_ids, train_texts = collect_train_gpqa_ids_texts(train_jsonl) if train_jsonl else (set(), set())
        self.train_ids = train_ids
        self.train_texts = train_texts

        # Determine evaluation rows
        if custom_eval_arrow:
            custom_rows = load_custom_eval_arrow(custom_eval_arrow)
            kept, removed = decontaminate(custom_rows, train_ids, train_texts)
            eval_rows = kept
            pool_candidates = list(hf_split)  # few-shot pool from HF to avoid leakage
            self.chosen_split = f"custom_arrow:{os.path.basename(custom_eval_arrow.rstrip('/'))}"
        elif custom_eval_path:
            custom_rows = load_custom_eval(custom_eval_path)
            kept, removed = decontaminate(custom_rows, train_ids, train_texts)
            eval_rows = kept
            pool_candidates = list(hf_split)
            self.chosen_split = f"custom:{os.path.basename(custom_eval_path)}"
        else:
            kept, removed = decontaminate(hf_split, train_ids, train_texts)
            eval_rows = kept
            pool_candidates = list(hf_split)

        self.decontaminated = kept
        self.contaminated_removed = removed

        # Few-shot pool
        rnd = random.Random(self.seed)
        rnd.shuffle(pool_candidates)
        self.fewshot_pool = pool_candidates

        # Subsample if requested
        if self.subsample != -1:
            if self.subsample > len(eval_rows):
                raise ValueError("subsample > dataset size")
            picks = random.Random(self.seed).sample(range(len(eval_rows)), k=self.subsample)
            self.eval_idx_map = picks
        else:
            self.eval_idx_map = list(range(len(eval_rows)))

        self.eval_rows = [eval_rows[i] for i in self.eval_idx_map]

        # Build few-shot prompt text (from fewshot_pool but NOT from eval_rows)
        self._build_few_shot_prompt()

        print(
            f"[GPQADataset] chosen_split={self.chosen_split} total_raw={len(self.raw_hf)} "
            f"decontaminated={len(self.decontaminated)} removed_contaminated={len(self.contaminated_removed)} "
            f"using_eval={len(self.eval_rows)}"
        )

    def __len__(self):
        return len(self.eval_rows)

    def _build_few_shot_prompt(self):
        if self.num_examples <= 0:
            self.few_shot_prompt = ""
            return
        pool = self.fewshot_pool
        if len(pool) == 0:
            self.few_shot_prompt = ""
            return
        k = min(self.num_examples, len(pool))
        rnd = random.Random(self.seed)
        ids = rnd.sample(range(len(pool)), k=k)
        chunks = []
        for i in ids:
            r = pool[i]
            q = r.get("question") or r.get("context") or ""
            choices = r.get("choices") or r.get("options") or []
            A, B, C, D = (list(choices) + ["", "", "", ""])[:4]
            lab = (r.get("answer") or r.get("correct") or r.get("label") or "").strip()
            if isinstance(lab, (int, float)):
                try:
                    lab = LETTERS[int(lab)]
                except Exception:
                    lab = ""
            chunks.append(
                f"Question:\n{q}\n\nOptions:\nA) {A}\nB) {B}\nC) {C}\nD) {D}\nAnswer:\n\\boxed{{{lab}}}"
            )
        self.few_shot_prompt = "\n\n".join(chunks)

    def create_prompt(self, question: str, choices: List[str]):
        if self.num_examples > 0 and self.few_shot_prompt:
            body = (
                f"{self.few_shot_prompt}\n\nQuestion:\n{question}\n\nOptions:\n"
                f"A) {choices[0]}\nB) {choices[1]}\nC) {choices[2]}\nD) {choices[3]}\nAnswer:\n"
            )
        else:
            body = (
                f"Question:\n{question}\n\nOptions:\n"
                f"A) {choices[0]}\nB) {choices[1]}\nC) {choices[2]}\nD) {choices[3]}"
            )
        msgs = [{"role": "user", "content": GPQA_SYSTEM_PROMPT + "\n\n" + body}]
        u = self.tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        return u + "<reasoning>" if self.add_reasoning else u

    def __getitem__(self, idx):
        r = self.eval_rows[idx]
        q = r.get("question") or r.get("context") or ""
        choices = r.get("choices") or r.get("options") or []
        if isinstance(choices, dict):
            choices = [choices.get("A", ""), choices.get("B", ""), choices.get("C", ""), choices.get("D", "")]
        A, B, C, D = (list(choices) + ["", "", "", ""])[:4]
        lab = (r.get("answer") or r.get("correct") or r.get("label") or "").strip()
        if isinstance(lab, (int, float)):
            try:
                lab = LETTERS[int(lab)]
            except Exception:
                lab = ""
        prompt = self.create_prompt(q, [A, B, C, D])
        return prompt, q, lab

    def collate_fn(self, batch):
        prompts = [b[0] for b in batch]
        questions = [b[1] for b in batch]
        answers = [b[2] for b in batch]
        input_ids = self.tokenizer(prompts, padding_side="left", return_tensors="pt", padding="longest").input_ids
        return {"input_ids": input_ids, "questions": questions, "answers": answers, "prompts": prompts}


# Standalone demo
if __name__ == "__main__":
    import argparse
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--train-jsonl", type=str, default=None)
    ap.add_argument("--num-fewshot", type=int, default=0)
    ap.add_argument("--subsample", type=int, default=-1)
    ap.add_argument("--tokenizer", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--eval-jsonl", type=str, default=None)
    ap.add_argument("--eval-arrow", type=str, default=None)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    ds = GPQADiamondArrowDataset(
        tok,
        num_examples=args.num_fewshot,
        add_reasoning=True,
        train_jsonl=args.train_jsonl,
        subsample=args.subsample,
        custom_eval_path=args.eval_jsonl,
        custom_eval_arrow=args.eval_arrow,
    )
    print("Dataset size:", len(ds))
    if len(ds) > 0:
        p, q, a = ds[0]
        print("PROMPT (truncated):\n", p[:1000])
        print("QUESTION:", q)
        print("GOLD:", a)
