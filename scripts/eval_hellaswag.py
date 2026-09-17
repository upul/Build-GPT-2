"""Score a model on HellaSwag.

Without --checkpoint-dir this evaluates OpenAI's GPT-2 124M, which is the
reference point for validating the harness (~29.55% acc_norm on the full split).
"""

import argparse

import tiktoken
import torch

from build_gpt2.checkpoint import load
from build_gpt2.hellaswag import HellaSwagEval
from build_gpt2.model import GPT, GPTConfig


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a model on HellaSwag")
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="score the latest checkpoint here; omit to score OpenAI GPT-2 124M",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="evaluate a strided sample of this many examples (default: all)",
    )
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or pick_device()
    tokenizer = tiktoken.get_encoding("gpt2")

    if args.checkpoint_dir is None:
        model = GPT.from_pretrained("gpt2").to(device)
    else:
        config = GPTConfig(block_size=args.context_length, vocab_size=50304)
        model = GPT(config).to(device)
        step = load(model=model, checkpoint_dir=args.checkpoint_dir, device=device)
        print(f"loaded checkpoint trained through step {step - 1}")

    harness = HellaSwagEval(num_evaluations=args.limit)
    result = harness.evaluate(model=model, tokenizer=tokenizer, device=device)

    print(f"device: {result['device']} | n = {result['n']}")
    print(f"acc      : {result['acc']:.2f}%")
    print(f"acc_norm : {result['acc_norm']:.2f}%")


if __name__ == "__main__":
    main()
