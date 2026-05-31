
import math
import torch


@torch.no_grad()
def get_llada_layer_attn(model, input_ids: torch.LongTensor, layer_idx: int):
    '''
    Return full attention probabilities [B, n_heads, T, T] for a single LLaDA layer.

    - model: LLaDAModelLM / AutoModelForCausalLM with trust_remote_code=True
    - input_ids: [B, T] LongTensor
    - layer_idx: 0-based layer index
    '''
    base = model.model           # LLaDAModel
    cfg = base.config

    assert 0 <= layer_idx < cfg.n_layers, f"layer_idx must be in [0, {cfg.n_layers-1}]"

    out = model(
        input_ids=input_ids,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )

    hidden_states = out.hidden_states  # tuple of length n_layers+1
    x = hidden_states[layer_idx]       # [B, T, d_model]

    block = base.transformer.blocks[layer_idx]
    B, T, C = x.shape

    # 1) Norm + QKV projections
    x_normed = block.attn_norm(x)
    q = block.q_proj(x_normed)
    k = block.k_proj(x_normed)
    v = block.v_proj(x_normed)

    # 2) Reshape into heads
    n_heads = cfg.n_heads
    head_dim = cfg.d_model // n_heads

    q = q.view(B, T, n_heads, head_dim).transpose(1, 2)  # [B, H, T, d]

    kv_heads = k.shape[-1] // head_dim
    k = k.view(B, T, kv_heads, head_dim).transpose(1, 2)
    v = v.view(B, T, kv_heads, head_dim).transpose(1, 2)

    # 3) Expand KV heads (GQA/MQA → full H)
    if kv_heads != n_heads:
        assert n_heads % kv_heads == 0
        repeat = n_heads // kv_heads
        k = k.repeat_interleave(repeat, dim=1, output_size=n_heads)
        v = v.repeat_interleave(repeat, dim=1, output_size=n_heads)

    # 4) RoPE
    if getattr(cfg, "rope", True):
        q, k = block.rotary_emb(q, k)

    # 5) Attention scores + softmax
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)  # [B, H, T, T]
    attn_probs = torch.softmax(scores, dim=-1)

    return attn_probs


def build_full_input_from_ids_and_answer(tokenizer, prompt_ids, answer, device):
    '''
    Construct full [prompt | answer] token ids.

    - tokenizer: LLaDA tokenizer
    - prompt_ids: 1D LongTensor [L_prompt]
    - answer: string or numeric ground-truth answer
    Returns:
      input_ids_full: [1, L_total]
      prompt_len: int
      gen_length: int (len(answer tokens))
    '''
    if not isinstance(answer, str):
        answer = str(answer)

    ans_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]

    prompt_ids_list = prompt_ids.tolist()
    full_ids_list = prompt_ids_list + ans_ids

    input_ids_full = torch.tensor(
        full_ids_list,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)  # [1, L_total]

    prompt_len = len(prompt_ids_list)
    gen_length = len(ans_ids)

    return input_ids_full, prompt_len, gen_length
