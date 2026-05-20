import random, numpy as np, torch
from datasets import load_dataset

LOGIQA_SYSTEM_PROMPT = """You are a careful logical reasoner.
Choose the correct option (A–D). Think step by step, then put ONLY the final letter inside \\boxed{<LETTER>}.
Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
\\boxed{A|B|C|D}
</answer>"""

def _letter(lbl):
    if isinstance(lbl, int): return "ABCD"[lbl]
    s = str(lbl).strip().upper()
    return s[0] if s and s[0] in "ABCD" else "A"

class LogiQADataset(torch.utils.data.Dataset):
    """
    HF: lucasmccabe/logiqa
      fields: context(str), query(str), options(list[str]), correct_option(int 0..3)
    split: often 'test'; fallback to 'validation'
    """
    TRY_SPLITS = ["test"]

    def __init__(self, tokenizer, num_examples=0, add_reasoning=True,
                 system_prompt=LOGIQA_SYSTEM_PROMPT, subsample=-1):
        self.tok = tokenizer
        self.nshot = int(num_examples)
        self.add_reasoning = bool(add_reasoning)
        self.sys = system_prompt

        self.ds = self._load_any()
        self._build_fewshot()

        idxs = np.arange(len(self.ds))
        if subsample != -1:
            assert subsample <= len(self.ds), "subsample > dataset size"
            idxs = np.random.choice(len(self.ds), subsample, replace=False)
        self.idxs = idxs
        print(f"evaluating {len(self.idxs)} examples")

    def _load_any(self):
        errs = []
        for sp in self.TRY_SPLITS:
            try:
                return load_dataset("lucasmccabe/logiqa", split=sp)
            except Exception as e:
                errs.append(f"{sp}: {e}")
        raise RuntimeError("Could not load lucasmccabe/logiqa.\n" + "\n".join(errs))

    def _build_fewshot(self):
        if self.nshot <= 0:
            self.fewshot = ""
            return
        ids = random.sample(range(len(self.ds)), k=min(self.nshot, len(self.ds)))
        chunks = []
        for i in ids:
            r = self.ds[i]
            ctx = (r.get("context") or "").strip()
            q = (r.get("query") or r.get("question") or "").strip()
            opts = r.get("options") or ["", "", "", ""]
            A, B, C, D = (opts + ["", "", "", ""])[:4]
            lab = _letter(r.get("correct_option"))
            text = (ctx + "\n\n" if ctx else "") + q
            chunks.append(
                f"Question:\n{text}\n\nOptions:\nA) {A}\nB) {B}\nC) {C}\nD) {D}\nAnswer:\n\\boxed{{{lab}}}"
            )
        self.fewshot = "\n\n".join(chunks)

    def _prompt(self, question, A, B, C, D):
        if self.nshot > 0 and self.fewshot:
            body = f"{self.fewshot}\n\nQuestion:\n{question}\n\nOptions:\nA) {A}\nB) {B}\nC) {C}\nD) {D}\nAnswer:\n"
        else:
            body = f"Question:\n{question}\n\nOptions:\nA) {A}\nB) {B}\nC) {C}\nD) {D}"
        msgs = [{"role": "user", "content": self.sys + "\n\n" + body}]
        u = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        return u + "<reasoning>" if self.add_reasoning else u

    def __len__(self): return len(self.idxs)

    def __getitem__(self, i):
        r = self.ds[self.idxs[i].item()]
        ctx = (r.get("context") or "").strip()
        q = (r.get("query") or r.get("question") or "").strip()
        opts = r.get("options") or ["", "", "", ""]
        A, B, C, D = (opts + ["", "", "", ""])[:4]
        gt = _letter(r.get("correct_option"))
        prompt = self._prompt((ctx + "\n\n" if ctx else "") + q, A, B, C, D)
        return prompt, q, gt

    def collate_fn(self, batch):
        prompts = [b[0] for b in batch]
        questions = [b[1] for b in batch]
        answers = [b[2] for b in batch]
        input_ids = self.tok(prompts, padding_side="left", return_tensors="pt", padding="longest").input_ids
        return {"input_ids": input_ids, "questions": questions, "answers": answers, "prompts": prompts}
