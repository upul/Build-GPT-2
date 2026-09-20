from pathlib import Path

import numpy as np
import torch


def load_tokens(path: str | Path) -> torch.Tensor:
    """Load one uint16 NumPy shard and expose token ids as torch.long."""
    tokens = np.load(path)
    return torch.tensor(tokens, dtype=torch.long)


class ShardedTokenLoader:
    """Rank-aware loader for pre-tokenized contiguous .npy shards.

    Each rank reads a distinct B*T slice from the same global batch. The loader
    deliberately rolls to the next shard instead of spanning a physical shard
    boundary, trading a tiny tail of unused tokens for simpler training code.
    """

    def __init__(
        self,
        batch_size: int,
        context_length: int,
        process_rank: int,
        num_processes: int,
        split: str,
        data_root: str | Path,
        master_process: bool = False,
    ):
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")

        self.batch_size = batch_size
        self.context_length = context_length
        self.process_rank = process_rank
        self.num_processes = num_processes

        data_root = Path(data_root)
        self.shards = sorted(data_root.glob(f"*{split}*.npy"))
        if not self.shards:
            raise FileNotFoundError(f"no {split} shards found in {data_root}")
        if master_process:
            print(f"found {len(self.shards)} shards for split {split}")
        self.reset()

    @property
    def tokens_per_rank_batch(self) -> int:
        return self.batch_size * self.context_length

    @property
    def global_stride(self) -> int:
        return self.tokens_per_rank_batch * self.num_processes

    def reset(self, current_shard: int = 0, current_position: int | None = None) -> None:
        if current_shard >= len(self.shards):
            raise ValueError(
                f"Invalid shard number: {current_shard}. Max allowable: {len(self.shards)}"
            )

        self.current_shard = current_shard
        self.tokens = load_tokens(self.shards[self.current_shard])
        if current_position is None:
            self.current_position = self.process_rank * self.tokens_per_rank_batch
        else:
            self.current_position = current_position

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        b, t = self.batch_size, self.context_length
        end = self.current_position + b * t + 1
        buf = self.tokens[self.current_position : end]
        if buf.numel() != b * t + 1:
            raise RuntimeError("shard boundary invariant violated")

        x = buf[:-1].view(b, t)
        y = buf[1:].view(b, t)
        self.current_position += self.global_stride

        # Preserve the behavior of the original training script: all ranks
        # leave a small safety tail instead of forming a batch across shards.
        if self.current_position + self.global_stride + 1 > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = self.process_rank * b * t

        return x, y
