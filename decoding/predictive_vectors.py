"""Predictive vectors for the modular-addition processes (Poncini 2025)."""
import numpy as np


def simplex_predictive_vectors(p: int) -> np.ndarray:
    """RRMod_p HMM control: ⟨e_p^(i)| = i-th elementary basis row vector. Shape (p, p)."""
    return np.eye(p)


def circle_predictive_vectors(omega: int, p: int) -> np.ndarray:
    """Vertices of the ω polygon: (cos(2π ω i/p), sin(2π ω i/p)). Shape (p, 2)."""
    angles = 2.0 * np.pi * omega * np.arange(p) / p
    return np.stack([np.cos(angles), np.sin(angles)], axis=1)


def ehmm_predictive_vectors(omegas, alphas, p: int) -> np.ndarray:
    """sRRMod_p EHMM: ⟨v^(i)| = ⊕_k α_k (cos, sin) at frequency ω_k. Shape (p, 2·n_freq)."""
    blocks = [a * circle_predictive_vectors(w, p) for w, a in zip(omegas, alphas)]
    return np.concatenate(blocks, axis=1)


def ehmm_predictive_vectors_unweighted(omegas, p: int) -> np.ndarray:
    """sRRMod_p EHMM with α_k = 1 — used as the semi-supervised target where the
    per-frequency amplitudes are deliberately not borrowed from the logit fit."""
    return ehmm_predictive_vectors(omegas, [1.0] * len(omegas), p)
