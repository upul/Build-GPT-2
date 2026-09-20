import argparse
from dataclasses import asdict
from pathlib import Path

import wandb
from build_gpt2.checkpoint import peek
from build_gpt2.distributed import cleanup_distributed, setup_distributed
from build_gpt2.train import TrainConfig, train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GPT-2 124M from scratch")
    parser.add_argument("--data-root", type=Path, required=True)
    # Run-identity flags default to None so a resume can tell "not passed"
    # apart from "passed the default value", and restore from the checkpoint.
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--micro-batch-size", type=int, default=None)
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--total-batch-size", type=int, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--checkpoint-interval", type=int, default=250)
    parser.add_argument("--hellaswag-interval", type=int, default=0)
    parser.add_argument("--hellaswag-limit", type=int, default=None)

    parser.add_argument("--wandb-project", type=str, default=None)
    return parser.parse_args()


# Hyperparameters that define the run. Changing any of these mid-run corrupts it
# (they feed the LR schedule and the meaning of the saved data-loader position),
# so a resume takes them from the checkpoint unless explicitly overridden.
RUN_IDENTITY = (
    "total_batch_size",
    "micro_batch_size",
    "context_length",
    "max_lr",
    "min_lr",
    "warmup_steps",
    "max_steps",
    "weight_decay",
    "grad_clip",
    "seed",
)


def build_config(args: argparse.Namespace, stored: dict) -> TrainConfig:
    """Build a TrainConfig, restoring run-identity fields from ``stored``.

    Paths stay with the CLI: a checkpoint written on one machine may be resumed
    somewhere the original --data-root does not exist.
    """
    kwargs = {
        "data_root": args.data_root,
        "resume": args.resume,
        "checkpoint_interval": args.checkpoint_interval,
        "hellaswag_interval": args.hellaswag_interval,
        "hellaswag_limit": args.hellaswag_limit,
        "checkpoint_dir": args.checkpoint_dir,
    }

    for name in RUN_IDENTITY:
        cli_value = getattr(args, name, None)
        if cli_value is not None:
            kwargs[name] = cli_value
            if name in stored and stored[name] != cli_value:
                print(f"overriding {name} from checkpoint: {stored[name]} -> {cli_value}")
        elif name in stored:
            kwargs[name] = stored[name]

    return TrainConfig(**kwargs)


def main() -> None:
    args = parse_args()
    ctx = setup_distributed()
    wandb_run = None
    try:
        wandb_run_id, stored_cfg = peek(args.checkpoint_dir) if args.resume else (None, {})
        config = build_config(args, stored_cfg)

        if ctx.master and args.wandb_project is not None:
            if args.resume:
                if wandb_run_id is None:
                    raise ValueError(
                        "checkpoint was written without W&B, so there is no run to "
                        "continue: drop --resume, or omit --wandb-project."
                    )

                wandb_run = wandb.init(
                    project=args.wandb_project,
                    config=asdict(config),
                    id=wandb_run_id,
                    resume="must",
                )
            else:
                wandb_run = wandb.init(project=args.wandb_project, config=asdict(config))

        train(
            config,
            ctx,
            wandb_run,
        )
    finally:
        if ctx.master and wandb_run is not None:
            wandb.finish()
        cleanup_distributed(ctx)


if __name__ == "__main__":
    main()
