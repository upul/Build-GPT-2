import math
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class GPTConfig:
    block_size: int = 1024  # AKA sequence length
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        # Registering my causal mask
        self.register_buffer(
            "bias",
            torch.triu(
                torch.ones(config.block_size, config.block_size, dtype=torch.bool),
                diagonal=1,
            ),
        )
        self.config = config

    def forward(self, x):
        B, T, D = x.shape
        QKV = self.c_attn(x)
        Q, K, V = QKV.split(self.config.n_embd, dim=-1)

        # [B, T, D] => [B, T, n_heads, D // n_heads] => [B, n_heads, T, D // n_heads]
        assert D % self.config.n_head == 0, f"D: {D}, n_head: {self.config.n_head}"
        Q = Q.view(B, T, self.config.n_head, D // self.config.n_head).transpose(1, 2)
        K = K.view(B, T, self.config.n_head, D // self.config.n_head).transpose(1, 2)
        V = V.view(B, T, self.config.n_head, D // self.config.n_head).transpose(1, 2)

        scores = Q @ K.transpose(-1, -2)
        causal_scores = scores.masked_fill(self.bias[:T, :T], float("-inf"))
        scaled_causal_scores = causal_scores / math.sqrt(K.size(-1))
        attn_scores = F.softmax(scaled_causal_scores, dim=-1)  # [B, n_head, T, T]

        # [B, n_head, T, T] @ [B, n_heads, T, D // n_heads] => [B, n_heads, T, D // n_heads]
        head_output = attn_scores @ V

        head_output = head_output.transpose(
            1, 2
        ).contiguous()  # [B, T, n_heads, D // n_heads]
        head_output = head_output.view(B, T, D)

        y = self.c_proj(head_output)
        return y


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
        self.attn = CausalSelfAttention(config)

    def forward(self, x):
        # This is the pre-normalization version
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "wpe": nn.Embedding(config.block_size, config.n_embd),
                "h": nn.ModuleList(Block(config) for _ in range(config.n_layer)),
                "ln_f": nn.LayerNorm(config.n_embd),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, idx, target=None):
        B, T = idx.shape
        assert T <= self.config.block_size, f"Can not forward sequences  of length: {T}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)  # [T, n_embd]
        tok_emb = self.transformer.wte(idx)  # [B, T, n_embd]

        x = pos_emb + tok_emb
        for module in self.transformer.h:
            x = module(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # [B, T, vocab_size]
        logits = logits.view(B * T, -1)
        if target is None:
            return logits
        target = target.view(-1)
        return logits, self.loss(logits, target=target)

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel

        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        config_args["vocab_size"] = 50257  # always 50257 for GPT model checkpoints
        config_args["block_size"] = 1024  # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [
            k for k in sd_keys if not k.endswith(".attn.bias")
        ]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")
        ]  # ignore these, just a buffer
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.bias")
        ]  # same, just the mask (buffer)
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), (
            f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        )
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model


# OK, I am going to detect the device available to me.
start_time = time.perf_counter()
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.mps.is_available():
    device = "mps"
print(f"using device: {device}")

num_return_sequences = 5
max_length = 30
# print("Loading original GPT-2 weights")
# model = GPT.from_pretrained("gpt2")
# model = GPT(GPTConfig())
# print("Wow, loading worked!")


# -----
import tiktoken

enc = tiktoken.get_encoding("gpt2")
with open("./input.txt", "r") as file:
    text = file.read()
tokens = enc.encode(text)
B, T = 4, 32

buf = torch.tensor(tokens[: B * T + 1])
x = buf[:-1].view(B, T).to(device=device)
y = buf[1:].view(B, T).to(device=device)

# get the logits
model = GPT(GPTConfig())
model.to(device=device)
# logits, loss = model(x, y)

# This is very important
# We can assume that weights will be ~ randomly initialized
# Also, our vocab size is 50257
# So our initial loss ~ -ln(1/50257)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(50):
    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    print(f"step {i:>4d} | loss: {loss.item():>.4f}")

end_time = time.perf_counter()
print(f"Elapsed time: {(end_time - start_time):<.4f} seconds")
import sys

sys.exit()


# OK, let's generate
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.mps.manual_seed(42)

model.eval()
model.to(device=device)
# x -> [B, T]
while x.size(1) < max_length:
    with torch.no_grad():
        logits = model(x)
        logits = logits[:, -1, :]  # [B, vocab]
        probs = F.softmax(logits, dim=-1)

        # top-k sampling here
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        ix = torch.multinomial(topk_probs, 1)  # [B, 1]
        xcol = torch.gather(topk_indices, -1, ix)
        x = torch.cat((x, xcol), dim=1)

# printing the generated data
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(f"> {decoded}")
