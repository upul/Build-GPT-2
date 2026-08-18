import numpy as np
import torch

from build_gpt2.data import ShardedTokenLoader


def test_shifted_targets_and_rank_partition(tmp_path):
    np.save(tmp_path / "fineweb_train_000001.npy", np.arange(256, dtype=np.uint16))

    rank0 = ShardedTokenLoader(2, 4, 0, 2, "train", tmp_path)
    rank1 = ShardedTokenLoader(2, 4, 1, 2, "train", tmp_path)

    x0, y0 = rank0.next_batch()
    x1, y1 = rank1.next_batch()

    assert torch.equal(y0[:, :-1], x0[:, 1:])
    assert torch.equal(y1[:, :-1], x1[:, 1:])
    assert x0.flatten().tolist() == list(range(0, 8))
    assert x1.flatten().tolist() == list(range(8, 16))
