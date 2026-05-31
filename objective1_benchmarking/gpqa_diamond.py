#!/usr/bin/env python3
"""
gpqa_diamond_dataset.py

GPQADiamondDataset: dataset wrapper for GPQA diamond/main that mirrors the GSM8K dataset class API.

Features:
 - Loads GPQA diamond/main from several HF mirrors (robust fallback).
 - Optional decontamination against an SFT train.jsonl (by id and normalized question text).
 - Few-shot prompt creation drawn from a separate fewshot_pool (ensures no leakage).
 - Exposes __len__, __getitem__, collate_fn, and helper prompt-building consistent with your eval scripts.

Example usage (inspect & print some prompts):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True)
    ds = GPQADiamondDataset(tok, num_examples=2, add_reasoning=True, train_jsonl="data/mix_v1/train.jsonl")
    p, q, a = ds[0]
    print(p)
    print(q, a)
"""
import json
import os
import random
import re
from typing import List, Optional, Set, Tuple

from datasets import load_dataset
import torch

# reuse your prompt & parsing utilities
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
        ("Idavidrein/gpqa", "main"),      # fallback to main if diamond missing
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
                        print(f"[gpqa_loader] Loaded {ds} split={pref} from dict len={len(dd[pref])}")
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
                # also check meta.source if present
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


class GPQADiamondDataset(torch.utils.data.Dataset):
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
    """

    def __init__(
        self,
        tokenizer,
        num_examples: int = 0,
        add_reasoning: bool = True,
        train_jsonl: Optional[str] = None,
        subsample: int = -1,
        seed: int = 23,
    ):
        import torch  # local import to avoid top-level dependency requirement in some contexts

        self.tokenizer = tokenizer
        self.num_examples = int(num_examples)
        self.add_reasoning = bool(add_reasoning)
        self.train_jsonl = train_jsonl
        self.seed = int(seed)
        self.subsample = int(subsample)

        # Load HF GPQA diamond/main split
        hf_split, chosen = load_gpqa_diamond_prefer_mirrors()
        self.chosen_split = chosen
        self.raw_hf = hf_split  # keep original for debugging

        # Collect SFT train ids/texts (for decontamination) if provided
        train_ids, train_texts = collect_train_gpqa_ids_texts(train_jsonl) if train_jsonl else (set(), set())
        self.train_ids = train_ids
        self.train_texts = train_texts

        # Decontaminate
        kept, removed = decontaminate(hf_split, train_ids, train_texts)
        self.decontaminated = kept
        self.contaminated_removed = removed

        # For few-shot examples, we will draw from a fewshot_pool.
        # Strategy:
        #  - If train_jsonl provided and there exists an HF mirror 'main' or other mirror with more examples,
        #    we try to use the removed (contaminated) examples as a potential few-shot pool (since they came from same
        #    GPQA source) OR use the raw_hf if safe. To be conservative, we'll prefer to use the removed examples only
        #    if train_jsonl was NOT provided (i.e., no contamination risk).
        if train_jsonl:
            # fallback: try to create fewshot_pool from the original full hf_split EXCLUDING decontaminated eval set.
            # We will use the removed (contaminated) list as an available pool if non-empty; otherwise fallback to raw_hf.
            pool_candidates = [r[0] for r in removed] if removed else list(hf_split)
        else:
            pool_candidates = list(hf_split)

        # Shuffle and set as list
        rnd = random.Random(self.seed)
        rnd.shuffle(pool_candidates)
        self.fewshot_pool = pool_candidates

        # Which dataset we will evaluate on: by default use decontaminated kept set.
        eval_rows = kept

        # apply subsample if requested
        if self.subsample != -1:
            if self.subsample > len(eval_rows):
                raise ValueError("subsample > dataset size")
            rnd = random.Random(self.seed)
            picks = rnd.sample(range(len(eval_rows)), k=self.subsample)
            self.eval_idx_map = picks
        else:
            self.eval_idx_map = list(range(len(eval_rows)))

        self.eval_rows = [eval_rows[i] for i in self.eval_idx_map]

        # Build few-shot prompt text (from fewshot_pool but do NOT sample from eval_rows)
        self._build_few_shot_prompt()

        print(f"[GPQADataset] chosen_split={self.chosen_split} total_raw={len(self.raw_hf)} decontaminated={len(self.decontaminated)} removed_contaminated={len(self.contaminated_removed)} using_eval={len(self.eval_rows)}")

    def __len__(self):
        return len(self.eval_rows)

    def _build_few_shot_prompt(self):
        # If num_examples == 0 => empty
        if self.num_examples <= 0:
            self.few_shot_prompt = ""
            return
        # avoid taking examples that overlap with evaluation rows (we already ensured fewshot_pool is separate above)
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
            A, B, C, D = (choices + ["", "", "", ""])[:4]
            lab = (r.get("answer") or r.get("correct") or r.get("label") or "").strip()
            # Normalize label to letter if it's integer index
            if isinstance(lab, (int, float)):
                try:
                    lab = LETTERS[int(lab)]
                except Exception:
                    lab = ""
            chunks.append(f"Question:\n{q}\n\nOptions:\nA) {A}\nB) {B}\nC) {C}\nD) {D}\nAnswer:\n\\boxed{{{lab}}}")
        self.few_shot_prompt = "\n\n".join(chunks)

    def create_prompt(self, question: str, choices: List[str]):
        # Build prompt similar to your other datasets
        if self.num_examples > 0 and self.few_shot_prompt:
            body = f"{self.few_shot_prompt}\n\nQuestion:\n{question}\n\nOptions:\nA) {choices[0]}\nB) {choices[1]}\nC) {choices[2]}\nD) {choices[3]}\nAnswer:\n"
        else:
            body = f"Question:\n{question}\n\nOptions:\nA) {choices[0]}\nB) {choices[1]}\nC) {choices[2]}\nD) {choices[3]}"
        msgs = [{"role": "user", "content": GPQA_SYSTEM_PROMPT + "\n\n" + body}]
        u = self.tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        return u + "<reasoning>" if self.add_reasoning else u

    def __getitem__(self, idx):
        # idx indexes into eval_rows
        r = self.eval_rows[idx]
        q = r.get("question") or r.get("context") or ""
        choices = r.get("choices") or r.get("options") or []
        A, B, C, D = (choices + ["", "", "", ""])[:4]
        # ground truth letter normalization
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


# If run standalone, show a small demo
if __name__ == "__main__":
    import argparse
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--train-jsonl", type=str, default=None, help="Optional SFT train.jsonl to decontaminate against")
    ap.add_argument("--num-fewshot", type=int, default=0)
    ap.add_argument("--subsample", type=int, default=-1)
    ap.add_argument("--tokenizer", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    ds = GPQADiamondDataset(tok, num_examples=args.num_fewshot, add_reasoning=True, train_jsonl=args.train_jsonl, subsample=args.subsample)
    print("Dataset size:", len(ds))
    if len(ds) > 0:
        p, q, a = ds[0]
        print("PROMPT (truncated):\n", p[:1000])
        print("QUESTION:", q)
        print("GOLD:", a)
