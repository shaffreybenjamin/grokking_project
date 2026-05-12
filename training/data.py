"""Dataset construction, model invocation, and evaluation helpers."""
import random

import torch
import torch.nn.functional as F

from .config import GrokConfig


def _default_device(device=None):
    if device is not None:
        return device
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_dataset(config: GrokConfig, device=None):
    """Build the (BOS, a, b) → (a+b mod p) dataset and split it deterministically.

    Convention (Poncini): tokens are (BOS, a, b) with BOS = p; loss is taken at the
    final position (index 2, the b position), which predicts c. The shuffle uses
    `random.Random(config.seed)` so the train/test split is reproducible across runs.
    """
    device = _default_device(device)
    pairs = [(config.p, i, j) for i in range(config.p) for j in range(config.p)]
    random.Random(config.seed).shuffle(pairs)
    labels = [config.fn(i, j) for _, i, j in pairs]
    all_data = torch.tensor(pairs, dtype=torch.long)
    all_labels = torch.tensor(labels, dtype=torch.long)
    n_train = int(config.frac_train * len(all_data))
    train_data = all_data[:n_train].to(device)
    train_labels = all_labels[:n_train].to(device)
    test_data = all_data[n_train:].to(device)
    test_labels = all_labels[n_train:].to(device)
    return train_data, train_labels, test_data, test_labels


def pretty_split(train_data, test_data) -> None:
    print(f'Train size: {len(train_data)}, Test size: {len(test_data)}')
    print(f'Train fraction: {len(train_data)/(len(train_data)+len(test_data)):.3f}')


def cross_entropy_high_precision(logits, labels):
    """Float64 log-softmax avoids the float32 underflow that creates spurious
    near-zero loss spikes during the post-grokking generalisation phase."""
    logprobs = F.log_softmax(logits.to(torch.float64), dim=-1)
    return -logprobs.gather(-1, labels[:, None].long()).mean()


def model_forward_logits(model, batch, is_transformer: bool):
    """Apply the model and slice out the logits used for loss / accuracy.

    Transformer: take the last-position logits, drop the BOS column.
    MLP        : take columns 1 and 2 of the (BOS,a,b) batch as (a, b).
    """
    if is_transformer:
        return model(batch)[:, -1, :-1]
    return model(batch[:, 1], batch[:, 2])


@torch.no_grad()
def evaluate(model, data, labels, is_transformer: bool):
    model.eval()
    logits = model_forward_logits(model, data, is_transformer)
    loss = cross_entropy_high_precision(logits, labels).item()
    accuracy = (logits.argmax(dim=-1) == labels).float().mean().item()
    return loss, accuracy
