import torch

from build_gpt2.model import GPT, GPTConfig


def tiny_model() -> GPT:
    return GPT(GPTConfig(block_size=16, vocab_size=64, n_layer=2, n_head=2, n_embd=32))


def test_inference_preserves_sequence_dimension():
    model = tiny_model()
    x = torch.randint(0, 64, (3, 8))
    logits = model(x)
    assert logits.shape == (3, 8, 64)


def test_training_returns_scalar_loss():
    model = tiny_model()
    x = torch.randint(0, 64, (2, 8))
    y = torch.randint(0, 64, (2, 8))
    logits, loss = model(x, y)
    assert logits.shape == (2, 8, 64)
    assert loss.ndim == 0
