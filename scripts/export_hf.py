#!/usr/bin/env python3
"""Convert a training checkpoint into a Hugging Face GPT-2 model directory.

The checkpoints this repo writes are keyed to its own ``GPT`` class, which is
useless to anyone who has not cloned it. This rewrites the weights into
``GPT2LMHeadModel`` layout so the model loads with::

    AutoModelForCausalLM.from_pretrained("<user>/<model>")

Two differences have to be reconciled:

* This model uses ``nn.Linear`` (``out, in``) where HF GPT-2 uses ``Conv1D``
  (``in, out``), so four weight groups per block are transposed.
* Training pads the vocabulary to 50304 for kernel efficiency while the GPT-2
  tokenizer emits 50257 ids, so the unused rows are dropped.

The conversion is verified by comparing logits against the source model before
anything is written.

    uv run python scripts/export_hf.py --checkpoint-dir ./checkpoints --out ./hf-export
"""

import argparse
from pathlib import Path

import torch

from build_gpt2.checkpoint import _read_latest_checkpoint
from build_gpt2.model import GPT, GPTConfig

# HF GPT-2 stores these as Conv1D (in, out); this repo uses nn.Linear (out, in).
TRANSPOSED = (
    "attn.c_attn.weight",
    "attn.c_proj.weight",
    "mlp.c_fc.weight",
    "mlp.c_proj.weight",
)
GPT2_VOCAB_SIZE = 50257


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a checkpoint to Hugging Face format")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--out", type=Path, default=Path("hf-export"))
    parser.add_argument(
        "--push-to",
        default=None,
        help="repo id to upload to, e.g. upul/build-gpt2-124m (requires `hf auth login`)",
    )
    parser.add_argument("--private", action="store_true", help="create the Hub repo as private")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-4,
        help="max absolute logit difference accepted when verifying the conversion",
    )
    return parser.parse_args()


def load_source_model(checkpoint_dir: str) -> tuple[GPT, dict]:
    path = _read_latest_checkpoint(checkpoint_dir)
    cp = torch.load(path, weights_only=False, map_location="cpu")
    model = GPT(GPTConfig(**cp["gpt_config"]))
    model.load_state_dict(cp["model"])
    model.eval()
    print(f"loaded {path.name} (step {cp['step']})")
    return model, cp


def to_hf_state_dict(state: dict, vocab_size: int) -> dict:
    """Rewrite this repo's state dict into GPT2LMHeadModel layout."""
    converted = {}
    for key, tensor in state.items():
        if key.endswith(TRANSPOSED):
            tensor = tensor.t().contiguous()
        elif key in ("transformer.wte.weight", "lm_head.weight"):
            # Drop the padding rows the tokenizer can never produce.
            tensor = tensor[:vocab_size].contiguous()
        converted[key] = tensor
    return converted


def build_hf_model(source: GPT, vocab_size: int):
    from transformers import GPT2Config, GPT2LMHeadModel

    cfg = source.config
    hf_config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=cfg.block_size,
        n_ctx=cfg.block_size,
        n_embd=cfg.n_embd,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        bos_token_id=50256,
        eos_token_id=50256,
    )
    hf_model = GPT2LMHeadModel(hf_config)
    converted = to_hf_state_dict(source.state_dict(), vocab_size)
    missing, unexpected = hf_model.load_state_dict(converted, strict=False)
    # HF registers causal-mask buffers this model does not carry; nothing else may be missing.
    missing = [k for k in missing if not k.endswith((".attn.bias", ".attn.masked_bias"))]
    if missing or unexpected:
        raise ValueError(f"state dict mismatch\n  missing: {missing}\n  unexpected: {unexpected}")
    hf_model.eval()
    return hf_model


def verify(source: GPT, hf_model, vocab_size: int, tolerance: float) -> float:
    """Compare logits on random ids; the converted model must reproduce the source."""
    generator = torch.Generator().manual_seed(0)
    seq_len = min(64, source.config.block_size)
    ids = torch.randint(0, vocab_size, (2, seq_len), generator=generator)

    with torch.no_grad():
        source_logits = source(ids)[..., :vocab_size]
        hf_logits = hf_model(ids).logits

    delta = (source_logits - hf_logits).abs().max().item()
    if delta > tolerance:
        raise ValueError(f"conversion changed the model: max |Δlogit| = {delta:.3e} > {tolerance}")
    return delta


def main() -> None:
    args = parse_args()
    source, cp = load_source_model(args.checkpoint_dir)
    vocab_size = min(GPT2_VOCAB_SIZE, source.config.vocab_size)

    hf_model = build_hf_model(source, vocab_size)
    delta = verify(source, hf_model, vocab_size, args.tolerance)
    print(f"verified: max |Δlogit| = {delta:.3e}")

    from transformers import GPT2TokenizerFast

    args.out.mkdir(parents=True, exist_ok=True)
    hf_model.save_pretrained(args.out)
    GPT2TokenizerFast.from_pretrained("gpt2").save_pretrained(args.out)
    print(f"wrote {args.out}/ (step {cp['step']}, vocab {vocab_size})")

    if args.push_to:
        hf_model.push_to_hub(args.push_to, private=args.private)
        GPT2TokenizerFast.from_pretrained("gpt2").push_to_hub(args.push_to, private=args.private)
        print(f"pushed to https://huggingface.co/{args.push_to}")
    else:
        print("add --push-to <user>/<model> to upload, or upload the directory yourself")


if __name__ == "__main__":
    main()
