import inspect
import math
import os
import time
from dataclasses import dataclass

import numpy as np
import tiktoken
import torch
import torch.nn.functional as F
from torch import nn

from prepare_dataset import split


def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


class DataLoader:
    def __init__(
        self,
        batch_size,
        context_length,
        process_rank,
        num_processes,
        master_process,
        split,
        data_root,
    ):
        self.batch_size = batch_size
        self.context_length = context_length
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {"train", "val"}

        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = (
            self.process_rank * self.batch_size * self.context_length
        )

    def next_batch(self):
        B, T = self.batch_size, self.context_length
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += B * T * self.num_processes

        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = self.process_rank * B * T
        return x, y


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
        self.c_proj.NANOGPT_SCALE_INIT = 1

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
        self.c_proj.NANOGPT_SCALE_INIT = 1

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

        # The following lines will be replaced by flash attention
        # scores = Q @ K.transpose(-1, -2)
        # causal_scores = scores.masked_fill(self.bias[:T, :T], float("-inf"))
        # scaled_causal_scores = causal_scores / math.sqrt(K.size(-1))
        # attn_scores = F.softmax(scaled_causal_scores, dim=-1)  # [B, n_head, T, T]
        # # [B, n_head, T, T] @ [B, n_heads, T, D // n_heads] => [B, n_heads, T, D // n_heads]
        # head_output = attn_scores @ V

        head_output = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
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

        # GPT-2 weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # initialize parameters
        self.apply(self._init_weight)

    def _init_weight(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

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
        # logits = logits.view(B * T, -1)
        logits = logits.view(-1, self.config.vocab_size)
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

    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and "cuda" in device
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            fused=use_fused,
        )
        return optimizer


max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715  # warmup over 375M tokens -> 375*10**6 / 2**19 -> 715
max_steps = 1907  # 1B tokens / 0.5M batch size -> 10^9 / 2**19


def get_lr(it):
    # linear warmup
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    elif it > max_steps:
        return min_lr
    decay_ration = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ration <= 1
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ration))
    return min_lr + coeff * (max_lr - min_lr)


# run the training loop using distributed data parallel
import os

import torch.distributed as dist
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
    assert torch.cuda.is_available(), "we need CUDA to run DDP"

    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])

    device = f"cuda:{ddp_local_rank}"
    # select the GPU before initializing NCCL
    torch.cuda.set_device(device=device)

    init_process_group(backend="nccl")

    master_process = ddp_rank == 0
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True

    # OK, I am going to detect the device available to me.
    start_time = time.perf_counter()
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.mps.is_available():
        device = "mps"

    # TODO:(Upul) manually setting the device to cpu
    print(f"using device: {device}")

torch.manual_seed(1337)
if device.startswith("cuda"):
    torch.cuda.manual_seed(1337)
elif device == "mps":
    torch.mps.manual_seed(1337)

# gradient accumulation
total_batch_size = 524288  # 2^19 ~ 0.5M batch size
B = 16  # This is my micro-batch size
T = 1024  # This is my context or sequence length
assert total_batch_size % (B * T * ddp_world_size) == 0, (
    "make sure the total batch_size is divisible by B * T ddp_world_size"
)
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulated steps: {grad_accum_steps}")

train_loader = DataLoader(
    batch_size=B,
    context_length=T,
    process_rank=ddp_rank,
    num_processes=ddp_world_size,
    master_process=master_process,
    split="train",
    data_root=".",
)
val_loader = DataLoader(
    batch_size=B,
    context_length=T,
    process_rank=ddp_rank,
    num_processes=ddp_world_size,
    master_process=master_process,
    split="val",
    data_root=".",
)
torch.set_float32_matmul_precision("high")
num_return_sequences = 5
max_length = 30

# -----

# create the model
print(
    f"[rank {ddp_rank}] local_rank={ddp_local_rank} device={device}",
    flush=True,
)
model = GPT(GPTConfig(vocab_size=50304))
model.to(device=device)
print(f"[rank {ddp_rank}] model on GPU", flush=True)

# model = torch.compile(model=model)

if ddp:
    print(f"[rank {ddp_rank}] entering DDP constructor", flush=True)
    model = DDP(model, device_ids=[ddp_local_rank])
    print(f"[rank {ddp_rank}] DDP ready", flush=True)

raw_model = model.module if ddp else model  #

# This is very important
# We can assume that weights will be ~ randomly initialized
# Also, our vocab size is 50257
# So our initial loss ~ -ln(1/50257)

# Let's optimize it
# optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), eps=1e-8)
print(f"[rank {ddp_rank}] creating optimizer", flush=True)
optimizer = raw_model.configure_optimizers(
    weight_decay=0.1, learning_rate=6e-4, device=device
)
print(f"[rank {ddp_rank}] optimizer ready", flush=True)

for step in range(max_steps):
    t_0 = time.perf_counter()
    # once in a while evaluate the validation loss
    if step % 100 == 0:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(x, y)

                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")

    # This is my training loop
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    device_type = "cuda" if device.startswith("cuda") else device
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x = x.to(device=device)
        y = y.to(device=device)
        if ddp:
            model.require_backward_grad_sync = micro_step == grad_accum_steps - 1

        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    optimizer.step()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t_1 = time.perf_counter()
    dt = t_1 - t_0
    tokens_processed = (
        train_loader.batch_size
        * train_loader.context_length
        * grad_accum_steps
        * ddp_world_size
    )
    tokens_per_sec = tokens_processed / dt
    if master_process:
        print(
            f"step {step:>5d} | loss: {loss_accum.item():>9.5f} | lr: {lr:10.4e} | norm: {norm:8.4f} | dt: {dt * 1000:>8.2f}ms | tok/sec: {tokens_per_sec:>10.4f}"
        )

if ddp:
    destroy_process_group()

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
