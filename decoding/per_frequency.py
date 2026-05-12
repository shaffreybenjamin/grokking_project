"""Per-frequency decoding: fit activations to single-circle predictive vectors."""
import numpy as np

from .decode import supervised_decode, semisupervised_decode
from .predictive_vectors import circle_predictive_vectors


def single_circle_target(indices: np.ndarray, omega: int, p: int) -> np.ndarray:
    """Stack of ⟨v_ω^(i)| for i in indices. Shape (n, 2)."""
    return circle_predictive_vectors(omega, p)[indices]


def sum_of_circles_target(idx_a: np.ndarray, idx_b: np.ndarray,
                          omega: int, p: int) -> np.ndarray:
    """⟨v_ω^(a)| + ⟨v_ω^(b)|. Shape (n, 2)."""
    return single_circle_target(idx_a, omega, p) + single_circle_target(idx_b, omega, p)


def per_frequency_decode(activations: np.ndarray, targets_per_omega: dict,
                         kind: str = 'supervised', alpha: float = 1e-3,
                         n_pcs: int = 10) -> dict:
    """Run a separate ridge regression per frequency. Returns {ω: result_dict}."""
    out = {}
    for w, target in targets_per_omega.items():
        if kind == 'supervised':
            out[w] = supervised_decode(activations, target, alpha=alpha)
        elif kind == 'semisupervised':
            out[w] = semisupervised_decode(activations, target, n_pcs=n_pcs, alpha=alpha)
        else:
            raise ValueError(f'Unknown kind: {kind}')
    return out
