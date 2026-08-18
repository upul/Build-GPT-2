# Build GPT-2 from Scratch

A compact GPT-2 124M training stack written in PyTorch, built to understand the full path from token shards to multi-GPU training rather than relying on a high-level trainer.

The implementation is inspired by Andrej Karpathy's `build-nanogpt` walkthrough and reorganized here as a small engineering codebase with explicit model, data, distributed, optimizer, scheduler, generation, and test modules.

<p align="center">
  <img src="assets/logo.png" alt="Build-GPT-2 logo" width="260">
</p>

<h1 align="center">Build-GPT-2</h1>

<p align="center">
  GPT-2 124M from scratch in PyTorch, trained on FineWeb-Edu with multi-GPU DDP.
</p>

## What this project demonstrates

- GPT-2-style decoder-only Transformer implemented from scratch
- GPT-2 weight tying and initialization
- PyTorch scaled dot-product attention (`scaled_dot_product_attention`)
- FineWeb-Edu data stored as pre-tokenized NumPy shards
- Rank-aware multi-GPU data loading
- DistributedDataParallel + NCCL
- BF16 autocast
- Gradient accumulation to a 524,288-token global batch
- Fused AdamW when available
- Linear warmup + cosine LR decay
- Validation and autoregressive top-k generation
- Unit tests for model shapes, LR scheduling, and data partitioning

## Measured training setup

| Item | Value |
|---|---:|
| Model | GPT-2 124M |
| Context length | 1024 |
| Global batch | 524,288 tokens |
| Precision | BF16 |
| GPUs | 4 |
| Measured throughput | ~235k tokens/s |
| Target training tokens | ~1B |

> The throughput above is a measured result from the author's training run and will vary by GPU and system configuration.

## Repository layout

```text
.
├── scripts/
│   ├── train.py            # CLI entry point
│   └── check_shards.py     # dataset integrity checks
├── src/build_gpt2/
│   ├── model.py            # GPTConfig, attention, MLP, blocks, GPT
│   ├── data.py             # rank-aware shard loader
│   ├── distributed.py      # DDP/NCCL setup and cleanup
│   ├── optim.py            # AdamW parameter grouping
│   ├── scheduler.py        # warmup + cosine decay
│   ├── generation.py       # top-k autoregressive generation
│   └── train.py            # validation and training orchestration
└── tests/
    ├── test_model.py
    ├── test_data.py
    └── test_scheduler.py
```

## Setup with uv

```bash
uv sync
```

On RunPod, mounted `/workspace` storage may be slow for Python environments. A practical setup is:

```bash
mkdir -p /root/venvs
export UV_PROJECT_ENVIRONMENT=/root/venvs/build-gpt2
export UV_LINK_MODE=copy
uv sync
```

Verify CUDA before launching a paid run:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

## Dataset layout

The trainer expects filenames containing `train` or `val`, for example:

```text
/workspace/data/
├── fineweb_val_000000.npy
├── fineweb_train_000001.npy
├── fineweb_train_000002.npy
└── ...
```

Validate the shards first:

```bash
uv run python scripts/check_shards.py /workspace/data
```

## Smoke test

Run 255 steps so validation and generation paths both execute:

```bash
uv run torchrun --standalone --nproc_per_node=4 scripts/train.py \
  --data-root /workspace/data \
  --max-steps 255 \
  --warmup-steps 100
```

## Full ~1B-token run

With a 524,288-token global batch, 1,907 optimizer steps expose about 999.8M training tokens:

```bash
uv run torchrun --standalone --nproc_per_node=4 scripts/train.py \
  --data-root /workspace/data \
  --max-steps 1907 \
  --warmup-steps 100
```

## Tests

```bash
uv run pytest -q
```

## Notes

This repository intentionally stays small. It is meant to make the training mechanics easy to inspect rather than hide them behind a framework. Natural next steps are checkpoint/resume support, structured experiment logging, standardized evaluation with `lm-evaluation-harness`, and a modern RoPE/RMSNorm/SwiGLU/GQA model.
