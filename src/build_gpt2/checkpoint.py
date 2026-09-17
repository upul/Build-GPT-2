import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch

from .data import ShardedTokenLoader


def save(
    model,
    optimizer,
    step,
    train_cfg: dict,
    gpt_config: dict,
    curr_shard: int,
    curr_position: int,
    checkpoint_dir: str,
):
    checkpoint = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "train_cfg": train_cfg,
        "gpt_config": gpt_config,
        "curr_shard": curr_shard,
        "curr_position": curr_position,
    }

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    cp_dir = Path(checkpoint_dir)
    cp_dir.mkdir(parents=True, exist_ok=True)
    curr_checkpoint = cp_dir / f"checkpoint_{step}_{timestamp}.pt"
    curr_checkpoint_tmp = curr_checkpoint.with_name(curr_checkpoint.name + ".tmp")

    torch.save(checkpoint, curr_checkpoint_tmp)
    os.replace(curr_checkpoint_tmp, curr_checkpoint)


def load(
    model,
    checkpoint_dir: str,
    device: str,
    optimizer=None,
    token_loader: ShardedTokenLoader | None = None,
):
    """Restore a checkpoint in place and return the step to resume from.

    ``optimizer`` and ``token_loader`` are optional so an evaluation-only caller
    can load weights without constructing them.
    """
    checkpoints = list(Path(checkpoint_dir).glob("*.pt"))
    if len(checkpoints) == 0:
        raise FileNotFoundError(
            f"No checkpoints are available at {checkpoint_dir}. Hence, can't load the model."
        )

    latest_checkpoint = max(checkpoints, key=lambda path: int(path.name.split("_")[1]))

    # loading the checkpoint
    cp = torch.load(latest_checkpoint, weights_only=False, map_location=device)
    # validate the model
    if cp["gpt_config"] != asdict(model.config):
        raise ValueError(
            "The loaded GPT configuration doesn't match with model's configuration"
        )
    model.load_state_dict(cp["model"])
    if optimizer is not None:
        optimizer.load_state_dict(cp["optimizer"])
    if token_loader is not None:
        token_loader.reset(
            current_shard=int(cp["curr_shard"]),
            current_position=int(cp["curr_position"]),
        )

    return cp["step"] + 1
