from dataclasses import dataclass
import os

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    ddp: bool
    rank: int
    local_rank: int
    world_size: int
    master: bool
    device: str

    @property
    def device_type(self) -> str:
        return "cuda" if self.device.startswith("cuda") else self.device


def setup_distributed() -> DistributedContext:
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for NCCL DDP")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl")
        return DistributedContext(ddp, rank, local_rank, world_size, rank == 0, device)

    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return DistributedContext(False, 0, 0, 1, True, device)


def cleanup_distributed(ctx: DistributedContext) -> None:
    if ctx.ddp and dist.is_initialized():
        dist.destroy_process_group()
