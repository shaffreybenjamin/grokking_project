"""Run configuration for the matched grokking setup."""
import random
from dataclasses import dataclass

import numpy as np
import torch


SEED = 0


def set_global_seeds(seed: int = SEED) -> None:
    """Seed Python / NumPy / Torch (CPU + CUDA) and disable CuDNN nondeterminism."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass(frozen=True)
class GrokConfig:
    p: int = 113
    frac_train: float = 0.30
    lr: float = 1e-3
    weight_decay: float = 1.0
    betas: tuple = (0.9, 0.98)
    d_model: int = 128
    d_mlp: int = 512
    mlp_d_hidden: int = 256
    n_ctx: int = 3
    transformer_epochs: int = 25_000
    mlp_epochs: int = 100_000
    num_checkpoints: int = 200
    seed: int = SEED
    fn_name: str = 'add'

    @property
    def d_vocab(self):
        return self.p + 1

    @property
    def fn(self):
        return lambda a, b: (a + b) % self.p

    @property
    def train_size(self):
        return int(self.frac_train * self.p * self.p)
