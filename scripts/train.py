#!/usr/bin/env python3
import argparse
from pathlib import Path

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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ctx = setup_distributed()
    try:
        train(
            TrainConfig(
                data_root=args.data_root,
                max_steps=args.max_steps,
                warmup_steps=args.warmup_steps,
                micro_batch_size=args.micro_batch_size,
                context_length=args.context_length,
                total_batch_size=args.total_batch_size,
                checkpoint_interval=args.checkpoint_interval,
                resume=args.resume,
            ),
            ctx,
        )
    finally:
        cleanup_distributed(ctx)


if __name__ == "__main__":
    main()
