import math


def cosine_warmup_lr(
    step: int,
    *,
    max_lr: float,
    min_lr: float,
    warmup_steps: int,
    max_steps: int,
) -> float:
    """Linear warmup followed by cosine decay to min_lr on the final step."""
    if max_steps <= 1:
        raise ValueError("max_steps must be > 1")
    if not 0 <= warmup_steps < max_steps:
        raise ValueError("warmup_steps must satisfy 0 <= warmup_steps < max_steps")

    if warmup_steps > 0 and step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps - 1:
        return min_lr

    decay_ratio = (step - warmup_steps) / ((max_steps - 1) - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)
