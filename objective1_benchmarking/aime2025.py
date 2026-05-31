# aime.py
import torch
import numpy as np
import random
from datasets import load_dataset
# Optional: keep parity with gsm8k.py imports; we won't rely on Parser though.
try:
    from parsers import Parser, is_equiv  # noqa: F401
except Exception:
    Parser = None  # not required

AIME_SYSTEM_PROMPT = """You are a math expert. Solve carefully.
Return ONLY the final integer answer inside \\boxed{...}.
Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
\\boxed{...}
</answer>"""


class AIME2025Dataset(torch.utils.data.Dataset):
    """
    AIME wrapper (prefers AIME-2025, falls back to AIME-2024).
    Expects fields: problem/question (str), answer (int-ish), solution (optional).
    Splits tried in order: test, validation, train.
    You can override the dataset id with env AIME_HF_ID.
    """

    TRY_SPLITS = ("test", "validation", "train")

    def __init__(
        self,
        tokenizer,
        num_examples=0,
        add_reasoning=True,
        system_prompt=AIME_SYSTEM_PROMPT,
        subsample=-1,
        seed=23,
    ):
        self.tokenizer = tokenizer
        self.num_examples = int(num_examples)
        self.add_reasoning = bool(add_reasoning)
        self.system_prompt = system_prompt
        self.R = random.Random(seed)

        self.load_eval_split()      # sets self.dataset
        self.create_few_shot_prompt()

        # Build index subset (like gsm8k.py)
        self.subsample = (
            np.random.choice(len(self.dataset), subsample, replace=False)
            if subsample != -1
            else np.arange(len(self.dataset))
        )
        print(f"evaluating {len(self.subsample)} examples")
        assert len(self.subsample) <= len(self.dataset), "Subsample size is greater than dataset size"

    # ---------------- core dataset API ----------------

    def __len__(self):
        return len(self.subsample)

    def __getitem__(self, idx):
        row = self.dataset[self.subsample[idx].item()]
        question = row.get("problem") or row.get("question") or ""
        gt = self._to_int(row.get("answer"))

        prompt = self.create_prompt(question)
        return prompt, question, gt

    def collate_fn(self, batch):
        prompts = [b[0] for b in batch]
        questions = [b[1] for b in batch]
        answers = [b[2] for b in batch]
        input_ids = self.tokenizer(
            prompts,
            padding_side="left",
            return_tensors="pt",
            padding="longest",
        ).input_ids
        return {"input_ids": input_ids, "questions": questions, "answers": answers, "prompts": prompts}

    # ---------------- helpers ----------------

    def load_eval_split(self):
        """Try AIME-2025 first, then fall back to AIME-2024; prefer test/validation."""
        import os
        candidates = [os.environ.get("AIME_HF_ID")] if os.environ.get("AIME_HF_ID") else []
        # Preferred → fallback order
        candidates += ["HuggingFaceH4/aime_2025", "HuggingFaceH4/aime_2024"]

        last_errs = []
        for ds_name in candidates:
            if not ds_name:
                continue
            for sp in self.TRY_SPLITS:
                try:
                    d = load_dataset(ds_name, split=sp)
                    print(f"[aime] Loaded {ds_name} split={sp} n={len(d)}")
                    self.dataset = d
                    return
                except Exception as e:
                    last_errs.append(f"{ds_name}:{sp}: {e}")
        raise RuntimeError("Could not load any AIME dataset.\n" + "\n".join(last_errs))

    def _to_int(self, x):
        try:
            return int(str(x).strip())
        except Exception:
            return None  # evaluator will skip if None

    def load_few_shot_examples(self):
        """Sample few-shot from train if available; else from current split."""
        # Try to pull a train split from the same dataset id if possible
        # If that fails, just use the current split as pool.
        pool = self.dataset
        try:
            # Attempt to detect backing HF id (best-effort; fine to fail)
            # Datasets 'Dataset' object often stores info in .info.builder_name/.info.config_name; not reliable across mirrors.
            # Simpler: re-try with known ids; if train unavailable, we'll except and keep current pool.
            for ds_name in ("HuggingFaceH4/aime_2025", "HuggingFaceH4/aime_2024"):
                try:
                    train_data = load_dataset(ds_name, split="train")
                    if len(train_data) > 0:
                        pool = train_data
                        break
                except Exception:
                    continue
        except Exception:
            pass

        k = max(0, min(self.num_examples, len(pool)))
        if k == 0:
            return []
        idxs = self.R.sample(range(len(pool)), k=k)
        return [pool[i] for i in idxs]

    def create_few_shot_prompt(self):
        """Create few-shot prompt from dataset examples (boxed integer)."""
        if self.num_examples <= 0:
            self.few_shot_prompt = ""
            return
        few_shot_examples = self.load_few_shot_examples()
        formatted = []
        for ex in few_shot_examples:
            q = ex.get("problem") or ex.get("question") or ""
            ans = self._to_int(ex.get("answer"))
            # keep it short and aligned with your eval scoring (\boxed{...})
            formatted.append(f"Question: {q}\nAnswer:\n\\boxed{{{'' if ans is None else ans}}}")
        self.few_shot_prompt = "\n\n".join(formatted)

    def create_prompt(self, input_text: str):
        # Align with gsm8k.py prompting style
        if self.num_examples > 0 and self.few_shot_prompt:
            prompt = f"{self.few_shot_prompt}\n\nQuestion: {input_text}\nAnswer:\n"
        else:
            prompt = input_text
        messages = [{"role": "user", "content": self.system_prompt + "\n\n" + prompt}]
        user_input = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return user_input + "<reasoning>" if self.add_reasoning else user_input
