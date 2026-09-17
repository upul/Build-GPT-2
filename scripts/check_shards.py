import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate token shard metadata")
    parser.add_argument("data_root", type=Path)
    args = parser.parse_args()

    paths = sorted(args.data_root.glob("*.npy"))
    if not paths:
        raise SystemExit(f"no .npy files found in {args.data_root}")

    total = 0
    for path in paths:
        arr = np.load(path, mmap_mode="r")
        if arr.ndim != 1:
            raise ValueError(f"{path}: expected 1-D array, got shape {arr.shape}")
        if arr.dtype != np.uint16:
            raise ValueError(f"{path}: expected uint16, got {arr.dtype}")
        if len(arr) == 0:
            raise ValueError(f"{path}: empty shard")
        total += len(arr)
        print(f"{path.name:40s} tokens={len(arr):,} dtype={arr.dtype}")

    print(f"shards={len(paths)} total_tokens={total:,}")


if __name__ == "__main__":
    main()
