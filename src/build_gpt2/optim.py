import inspect

import torch
from torch import nn


def configure_adamw(
    model: nn.Module,
    *,
    weight_decay: float,
    learning_rate: float,
    device: str,
) -> torch.optim.Optimizer:
    params = {name: p for name, p in model.named_parameters() if p.requires_grad}
    decay = [p for p in params.values() if p.dim() >= 2]
    no_decay = [p for p in params.values() if p.dim() < 2]

    print(
        f"num decayed parameter tensors: {len(decay)}, "
        f"with {sum(p.numel() for p in decay):,} parameters"
    )
    print(
        f"num non-decayed parameter tensors: {len(no_decay)}, "
        f"with {sum(p.numel() for p in no_decay):,} parameters"
    )

    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device.startswith("cuda")
    print(f"using fused AdamW: {use_fused}")

    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=use_fused,
    )
