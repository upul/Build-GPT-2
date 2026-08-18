import torch
import torch.nn.functional as F


def generate_top_k(
    model,
    prompt_tokens: list[int],
    *,
    num_return_sequences: int,
    max_length: int,
    top_k: int,
    device: str,
    device_type: str,
    seed: int,
) -> torch.Tensor:
    """Autoregressive top-k sampling for a decoder-only model."""
    tokens = torch.tensor(prompt_tokens, dtype=torch.long)
    x = tokens.unsqueeze(0).repeat(num_return_sequences, 1).to(device)

    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    while x.size(1) < max_length:
        with torch.no_grad(), torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits = model(x)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)
            sample_idx = torch.multinomial(topk_probs, 1, generator=rng)
            next_token = torch.gather(topk_indices, -1, sample_idx)
            x = torch.cat((x, next_token), dim=1)
    return x
