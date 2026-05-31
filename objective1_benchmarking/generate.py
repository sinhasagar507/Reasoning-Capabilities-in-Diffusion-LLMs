# generate.py
import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch.distributed as dist

def gumbel_max_sample(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Correct Gumbel-Max sampling:
      argmax( (logits / T) + Gumbel(0,1) )
    If temperature == 0, falls back to greedy argmax(logits).
    """
    if temperature is None or temperature <= 0.0:
        return logits.argmax(dim=-1)

    logits = logits.float() / max(1e-6, float(temperature))
    # clamp to avoid log(0)
    u = torch.rand_like(logits, dtype=torch.float32).clamp_(1e-6, 1.0 - 1e-6)
    g = -torch.log(-torch.log(u))
    return (logits + g).argmax(dim=-1)

def apply_top_k_top_p(logits: torch.Tensor, top_k: int = 0, top_p: float = 0.0) -> torch.Tensor:
    """
    Filter logits with top-k and/or top-p. Returns filtered logits (float32).
    """
    logits = logits.float()
    if top_k and top_k > 0:
        v, _ = torch.topk(logits, top_k, dim=-1)
        cut = v[..., -1, None]
        logits = torch.where(logits < cut, torch.full_like(logits, -float("inf")), logits)

    if top_p and top_p > 0.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumprobs = probs.cumsum(dim=-1)
        cutoff = cumprobs > top_p
        cutoff[..., 0] = False
        sorted_logits[cutoff] = -float("inf")
        # scatter back
        logits = torch.full_like(logits, -float("inf"))
        logits.scatter_(dim=-1, index=sorted_idx, src=sorted_logits)

    return logits

def repetition_penalize_logits(logits: torch.Tensor, seq: torch.Tensor, penalty: float = 1.0) -> torch.Tensor:
    """
    Very light repetition penalty: >1.0 discourages re-using tokens already in seq.
    Implemented by dividing logits of seen tokens by penalty.
    """
    if penalty is None or penalty <= 1.0:
        return logits
    logits = logits.float()
    B, T = seq.shape
    V = logits.size(-1)
    # Build a mask of tokens present in each sequence (approx; cost O(B*T))
    seen = torch.zeros((B, V), device=seq.device, dtype=torch.bool)
    seen.scatter_(1, seq, True)
    # Normalize by penalty for seen tokens
    seen = seen.unsqueeze(1).expand(-1, logits.size(1), -1)  # [B, T, V]
    logits = torch.where(seen, logits / penalty, logits)
    return logits

@torch.no_grad()
def generate(
    model,
    prompt,
    tokenizer,
    steps=64,
    gen_length=128,
    block_length=32,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=126336,
    top_k: int = 0,
    top_p: float = 0.0,
    rep_penalty: float = 1.0,
):
    """
    Masked diffusion-style generation (blockwise):
      - Start with gen window masked with mask_id
      - Iteratively replace most-confident masked tokens
    """
    device = next(model.parameters()).device
    B = prompt.shape[0]
    T_prompt = prompt.shape[1]

    # Work in int64 ids buffer
    x = torch.full((B, T_prompt + gen_length), mask_id, dtype=torch.long, device=device)
    x[:, :T_prompt] = prompt.clone()

    prompt_index = x != mask_id
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    steps_per_block = max(1, steps // num_blocks)

    # Main loop
    for b in tqdm(range(num_blocks), disable=(dist.is_available() and dist.is_initialized() and dist.get_rank() != 0),
                  desc=f"Eval (len={gen_length}, steps={steps})"):
        start_idx = T_prompt + b * block_length
        end_idx   = T_prompt + (b + 1) * block_length

        # how many tokens to reveal each inner step per sequence (evenly split)
        block_mask_index = x[:, start_idx:end_idx] == mask_id
        mask_num = block_mask_index.sum(dim=1, keepdim=True)  # [B,1]
        base = mask_num // steps_per_block
        rem  = mask_num %  steps_per_block
        num_transfer_tokens = base.expand(-1, steps_per_block).clone()
        if (rem > 0).any():
            idx = torch.arange(steps_per_block, device=device).unsqueeze(0)
            add = (idx < rem).to(num_transfer_tokens.dtype)
            num_transfer_tokens += add

        for i in range(steps_per_block):
            mask_index = x == mask_id  # [B, T_prompt + gen_length]

            # classifier-free guidance (two passes in one forward)
            if cfg_scale and cfg_scale > 0.0:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                outputs = model(x_)
                logits, un_logits = torch.chunk(outputs.logits, 2, dim=0)
                logits = (un_logits + (cfg_scale + 1.0) * (logits - un_logits)).float()
            else:
                outputs = model(x)
                logits = outputs.logits.float()  # [B, T, V]

            # (optional) discourage repeats a bit using the *current* sequence
            logits = repetition_penalize_logits(logits, x, penalty=rep_penalty)

            # (optional) top-k / top-p filtering
            logits = apply_top_k_top_p(logits, top_k=top_k, top_p=top_p)

            # choose token ids
            x0 = gumbel_max_sample(logits, temperature=temperature)  # [B, T]

            # confidence: p(token) at chosen ids
            if remasking == "low_confidence":
                # stable softmax
                smx = torch.softmax(logits, dim=-1)
                x0_p = torch.gather(smx, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)  # [B, T]
            elif remasking == "random":
                x0_p = torch.rand_like(x0, dtype=torch.float32)
            else:
                raise NotImplementedError(remasking)

            # do not transfer outside current block
            x0_p[:, end_idx:] = -float("inf")
            # only allow masked positions to be transferred
            x0 = torch.where(mask_index, x0, x)
            conf = torch.where(mask_index, x0_p, torch.full_like(x0_p, -float("inf")))

            # pick the top num_transfer_tokens masked positions per sequence
            for j in range(B):
                k = int(num_transfer_tokens[j, i].item())
                if k <= 0:
                    continue
                vals, idxs = torch.topk(conf[j], k=k, dim=-1)
                # update those positions
                x[j, idxs] = x0[j, idxs]

    return x
