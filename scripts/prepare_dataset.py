import numpy as np
import tiktoken
from datasets import load_dataset

SHARD_SIZE = 100_000_000
MAX_TOKENS = 2_600_000_000  # 100M validation + 2.5B training
DATA_FOLDER = "./data"

dataset = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
enc = tiktoken.get_encoding("gpt2")

shard = np.empty(SHARD_SIZE, dtype=np.uint16)
token_count = 0
shard_index = 0

num_docs = 0
total_tokens = 0

for data in dataset:
    text = data["text"]
    tokens = [enc.eot_token]
    tokens.extend(enc.encode_ordinary(text))

    tokens_np = np.array(tokens, dtype=np.uint16)

    start = 0

    while start < len(tokens_np):
        remaining_space = SHARD_SIZE - token_count
        remaining_tokens = len(tokens_np) - start
        remaining_budget = MAX_TOKENS - total_tokens

        n = min(remaining_space, remaining_tokens, remaining_budget)

        shard[token_count : token_count + n] = tokens_np[start : start + n]
        token_count += n
        start += n
        total_tokens += n

        if token_count == SHARD_SIZE:
            split = "val" if shard_index == 0 else "train"
            filename = f"fineweb_{split}_{shard_index:06d}.npy"
            np.save(f"{DATA_FOLDER}/{filename}", shard)

            print(f"saved {filename} | tokens={SHARD_SIZE:,} | bytes={shard.nbytes:,}")
            shard_index += 1
            token_count = 0

        if total_tokens >= MAX_TOKENS:
            break

    if start == len(tokens_np):
        num_docs += 1

    if num_docs % 10_000 == 0:
        print(
            f"docs={num_docs:,} | "
            f"tokens={total_tokens:,} | "
            f"shards={shard_index} | "
            f"current_fill={token_count:,}/{SHARD_SIZE:,}"
        )

    if total_tokens >= MAX_TOKENS:
        break

if token_count > 0:
    split = "val" if shard_index == 0 else "train"
    filename = f"{DATA_FOLDER}/fineweb_{split}_{shard_index:06d}.npy"

    np.save(filename, shard[:token_count])

    print(
        f"saved {filename} | "
        f"tokens={token_count:,} | "
        f"bytes={shard[:token_count].nbytes:,}"
    )
