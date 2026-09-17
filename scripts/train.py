import argparse
from ast import arg
from pathlib import Path

import wandb
from build_gpt2.distributed import cleanup_distributed, setup_distributed
from build_gpt2.train import TrainConfig, train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GPT-2 124M from scratch")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=1907)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--micro-batch-size", type=int, default=16)
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--total-batch-size", type=int, default=524_288)
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--checkpoint-interval", type=int, default=250)
    parser.add_argument("--hellaswag-interval", type=int, default=0)
    parser.add_argument("--hellaswag-limit", type=int, default=None)

    parser.add_argument("--wandb-project", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ctx = setup_distributed()
    run = None
    try:
        if ctx.master and args.wandb_project is not None:
            run = wandb.init(project=args.wandb_project)

        train(
            TrainConfig(
                data_root=args.data_root,
                max_steps=args.max_steps,
                warmup_steps=args.warmup_steps,
                micro_batch_size=args.micro_batch_size,
                context_length=args.context_length,
                total_batch_size=args.total_batch_size,
                checkpoint_interval=args.checkpoint_interval,
                hellaswag_interval=args.hellaswag_interval,
                hellaswag_limit=args.hellaswag_limit,
                resume=args.resume,
            ),
            ctx,
            run,
        )
    finally:
        if ctx.master and run is not None:
            wandb.finish()
        cleanup_distributed(ctx)


if __name__ == "__main__":
    main()
