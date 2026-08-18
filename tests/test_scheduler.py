import pytest

from build_gpt2.scheduler import cosine_warmup_lr


def test_scheduler_hits_peak_and_floor():
    kwargs = dict(max_lr=6e-4, min_lr=6e-5, warmup_steps=100, max_steps=1907)
    assert cosine_warmup_lr(0, **kwargs) == pytest.approx(6e-6)
    assert cosine_warmup_lr(99, **kwargs) == pytest.approx(6e-4)
    assert cosine_warmup_lr(1906, **kwargs) == pytest.approx(6e-5)


def test_cosine_decay_is_between_bounds():
    kwargs = dict(max_lr=6e-4, min_lr=6e-5, warmup_steps=100, max_steps=1907)
    lr = cosine_warmup_lr(1000, **kwargs)
    assert kwargs["min_lr"] < lr < kwargs["max_lr"]
