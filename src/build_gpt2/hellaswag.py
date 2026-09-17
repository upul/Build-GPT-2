import torch
import torch.nn.functional as F
from datasets import load_dataset


class HellaSwagEval:
    def __init__(
        self,
        num_evaluations: int | None = None,
        dataset="Rowan/hellaswag",
        split="validation",
    ):
        self.dataset = load_dataset(dataset, split=split)
        total = len(self.dataset)

        # The validation split is grouped by source activity, not shuffled, so a
        # prefix is biased: the first 500 examples score ~33.7% for GPT-2 124M
        # against a true 29.55% over the full split. Sample with a stride so a
        # limited evaluation stays comparable to the full-set number.
        if num_evaluations is None or num_evaluations >= total:
            self.indices = range(total)
        else:
            self.indices = range(0, total, total // num_evaluations)[:num_evaluations]

        self.num_evaluations = len(self.indices)
        self.cache: list[tuple[torch.Tensor, torch.Tensor, int]] = []

    def evaluate(self, model, tokenizer, device, verbose: bool = True):
        if not self.cache:
            for index in self.indices:
                self.cache.append(self._encode_single(tokenizer, self.dataset[index]))
            if verbose:
                print(f"tokenized {len(self.cache)} examples")

        n_total = len(self.cache)
        n_correct = 0
        n_correct_norm = 0
        model.eval()
        for i, (tokens, mask, label) in enumerate(self.cache):
            pred, pred_norm = self._validate_single(model, tokens, mask, device)
            n_correct += int(pred == label)
            n_correct_norm += int(pred_norm == label)

            if verbose and i > 0 and i % 500 == 0:
                seen = i + 1
                print(
                    f"progress {seen}/{n_total} | "
                    f"acc {n_correct / seen * 100:.2f}% | "
                    f"acc_norm {n_correct_norm / seen * 100:.2f}%"
                )

        return {
            "device": device,
            "n": n_total,
            "acc": (n_correct / n_total) * 100.0,
            "acc_norm": (n_correct_norm / n_total) * 100.0,
        }

    def _encode_single(
        self, tokenizer, data, eot_token=50256
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        ctx = data["ctx"]
        ctx_enc = tokenizer.encode(ctx)
        label = int(data["label"])

        ending_tokens = []
        max_seq = 0
        for ending in data["endings"]:
            encoded = tokenizer.encode(" " + ending)
            ending_tokens.append(encoded)
            max_seq = max(max_seq, len(encoded) + len(ctx_enc))

        encoded_data = torch.full((4, max_seq), eot_token, dtype=torch.long)
        mask = torch.full((4, max_seq), 0, dtype=torch.long)

        for i, ending in enumerate(ending_tokens):
            curr_ctx_enc = ctx_enc.copy()
            curr_ctx_enc.extend(ending)
            size = len(curr_ctx_enc)
            encoded_data[i, :size] = torch.tensor(curr_ctx_enc)
            mask[i, len(ctx_enc) : len(ctx_enc) + len(ending)] = 1
        return (encoded_data, mask, label)

    def _validate_single(self, model, tokens, mask, device):
        tokens, mask = tokens.to(device), mask.to(device)
        with torch.no_grad():
            logits = model(tokens)
            B, _, vocab_size = logits.shape
            grid = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, vocab_size),
                tokens[:, 1:].reshape(-1),
                reduction="none",
            )
            grid = grid.view(B, -1)
            loss = (grid * mask[:, 1:]).sum(dim=1)
            pred = torch.argmin(loss).item()
            lengths = mask[:, 1:].sum(dim=1)
            acc_norm_scores = loss / lengths
            pred_norm = torch.argmin(acc_norm_scores).item()

        return pred, pred_norm
