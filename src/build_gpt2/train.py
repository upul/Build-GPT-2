import time
from dataclasses import asdict, dataclass
from pathlib import Path

import tiktoken
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .checkpoint import load, save
from .data import ShardedTokenLoader
from .distributed import DistributedContext
from .generation import generate_top_k
from .hellaswag import HellaSwagEval
from .model import GPT, GPTConfig
from .optim import configure_adamw
from .scheduler import cosine_warmup_lr


@dataclass
class TrainConfig:
    data_root: Path
    total_batch_size: int = 524_288
    micro_batch_size: int = 16
    context_length: int = 1024
    max_lr: float = 6e-4
    min_lr: float = 6e-5
    warmup_steps: int = 100
    max_steps: int = 1907
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    val_interval: int = 100
    val_steps: int = 20
    generate_interval: int = 250
    resume: bool = False
    checkpoint_interval: int = 250
    hellaswag_interval: int = 0
    hellaswag_limit: int | None = None
    checkpoint_dir: str = "./checkpoints"
    seed: int = 1337


def evaluate(
    model,
    loader: ShardedTokenLoader,
    *,
    val_steps: int,
    ctx: DistributedContext,
) -> torch.Tensor:
    model.eval()
    loader.reset()
    loss_accum = torch.zeros((), device=ctx.device)
    with torch.no_grad():
        for _ in range(val_steps):
            x, y = loader.next_batch()
            x, y = x.to(ctx.device), y.to(ctx.device)
            with torch.autocast(device_type=ctx.device_type, dtype=torch.bfloat16):
                _, loss = model(x, y)
            loss_accum += loss.detach() / val_steps

    if ctx.ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    return loss_accum


def train(config: TrainConfig, ctx: DistributedContext) -> None:
    torch.manual_seed(config.seed)
    if ctx.device.startswith("cuda"):
        torch.cuda.manual_seed(config.seed)
    elif ctx.device == "mps":
        torch.mps.manual_seed(config.seed)

    if config.total_batch_size % (
        config.micro_batch_size * config.context_length * ctx.world_size
    ):
        raise ValueError("total_batch_size must be divisible by B*T*world_size")

    grad_accum_steps = config.total_batch_size // (
        config.micro_batch_size * config.context_length * ctx.world_size
    )
    if ctx.master:
        print(f"using device: {ctx.device}")
        print(f"total desired batch size: {config.total_batch_size}")
        print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    train_loader = ShardedTokenLoader(
        config.micro_batch_size,
        config.context_length,
        ctx.rank,
        ctx.world_size,
        "train",
        config.data_root,
        ctx.master,
    )
    val_loader = ShardedTokenLoader(
        config.micro_batch_size,
        config.context_length,
        ctx.rank,
        ctx.world_size,
        "val",
        config.data_root,
        ctx.master,
    )

    # setting up Hellaswag evaluation
    hellaswag_eval = None
    if ctx.master and config.hellaswag_interval > 0:
        hellaswag_eval = HellaSwagEval(num_evaluations=config.hellaswag_limit)

    torch.set_float32_matmul_precision("high")
    gpt_config = GPTConfig(block_size=config.context_length, vocab_size=50304)
    model = GPT(gpt_config).to(ctx.device)
    if ctx.master:
        num_parameters = sum([p.numel() for p in model.parameters()])
        print(f"number of parameters: {num_parameters / 1e6} M")

    if ctx.ddp:
        model = DDP(model, device_ids=[ctx.local_rank])
    raw_model = model.module if ctx.ddp else model

    optimizer = configure_adamw(
        raw_model,
        weight_decay=config.weight_decay,
        learning_rate=config.max_lr,
        device=ctx.device,
    )
    tokenizer = tiktoken.get_encoding("gpt2")
    start = 0
    if config.resume:
        start = load(
            model=raw_model,
            checkpoint_dir=config.checkpoint_dir,
            device=ctx.device,
            optimizer=optimizer,
            token_loader=train_loader,
        )
        if ctx.master:
            print(f"Resuming the training at step: {start}")

    for step in range(start, config.max_steps):
        last_step = step == config.max_steps - 1

        if step % config.val_interval == 0:
            val_loss = evaluate(model, val_loader, val_steps=config.val_steps, ctx=ctx)
            if ctx.master:
                print(f"validation loss: {val_loss.item():.4f}")

        # Use the raw model for master-only generation so no DDP collectives are required.
        if ctx.master and (
            (step > 0 and step % config.generate_interval == 0) or last_step
        ):
            raw_model.eval()
            generated = generate_top_k(
                raw_model,
                tokenizer.encode("Hello, I'm a language model,"),
                num_return_sequences=3,
                max_length=32,
                top_k=50,
                device=ctx.device,
                device_type=ctx.device_type,
                seed=42,
            )
            for sample_idx in range(generated.size(0)):
                decoded = tokenizer.decode(generated[sample_idx].tolist())
                print(f"sample {sample_idx}: {decoded}")

        # This is the Hellaswag Evaluation loop.
        if (
            ctx.master
            and (hellaswag_eval is not None)
            and ((step > 0 and step % config.hellaswag_interval == 0) or last_step)
        ):
            eval_result = hellaswag_eval.evaluate(
                raw_model, tokenizer, ctx.device, verbose=False
            )
            # TODO: (We need to correctly log the performance)
            print(f"hellaswag eval result: {eval_result}")

        # Now we start training
        model.train()
        optimizer.zero_grad()
        loss_accum = torch.zeros((), device=ctx.device)
        train_start = time.perf_counter()

        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(ctx.device), y.to(ctx.device)
            if ctx.ddp:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1

            with torch.autocast(device_type=ctx.device_type, dtype=torch.bfloat16):
                _, loss = model(x, y)
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()

        if ctx.ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        lr = cosine_warmup_lr(
            step,
            max_lr=config.max_lr,
            min_lr=config.min_lr,
            warmup_steps=config.warmup_steps,
            max_steps=config.max_steps,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()

        if ctx.device.startswith("cuda"):
            torch.cuda.synchronize()
        dt = time.perf_counter() - train_start
        tokens_per_sec = config.total_batch_size / dt

        if ctx.master:
            print(
                f"step {step:>5d} | loss: {loss_accum.item():>9.5f} | "
                f"lr: {lr:10.4e} | norm: {norm:8.4f} | "
                f"dt: {dt * 1000:>8.2f}ms | tok/sec: {tokens_per_sec:>10.4f}"
            )

        if (
            (step > 0 and step % config.checkpoint_interval == 0) or last_step
        ) and ctx.master:
            print(f"saving checkpoint at {step} step")
            save(
                step=step,
                model=raw_model,
                optimizer=optimizer,
                train_cfg=asdict(config),
                gpt_config=asdict(gpt_config),
                curr_shard=train_loader.current_shard,
                curr_position=train_loader.current_position,
                checkpoint_dir=config.checkpoint_dir,
            )
