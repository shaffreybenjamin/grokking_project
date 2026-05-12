"""Fit (α_k, ω_k) to model logits via Poncini's Fourier procedure."""
import numpy as np
import torch


def transformer_logit_tensor(model, p: int) -> np.ndarray:
    """z[a,b,c] for the (BOS,a,b)→c transformer; BOS = p. Shape (p,p,p), float64."""
    device = next(model.parameters()).device
    a, b = torch.meshgrid(torch.arange(p), torch.arange(p), indexing='ij')
    a, b = a.flatten().to(device), b.flatten().to(device)
    bos = torch.full_like(a, p)
    x = torch.stack([bos, a, b], dim=1)
    with torch.no_grad():
        logits = model(x)[:, -1, :p]
    return logits.reshape(p, p, p).double().cpu().numpy()


def mlp_logit_tensor(model, p: int) -> np.ndarray:
    """z[a,b,c] for the (a,b)→c MLP. Shape (p,p,p), float64."""
    device = next(model.parameters()).device
    a, b = torch.meshgrid(torch.arange(p), torch.arange(p), indexing='ij')
    a, b = a.flatten().to(device), b.flatten().to(device)
    with torch.no_grad():
        logits = model(a, b)
    return logits.reshape(p, p, p).double().cpu().numpy()


def fit_frequencies(z: np.ndarray, p: int, target_r2: float = 0.99,
                    max_freqs: int = 10) -> dict:
    """Greedy Fourier fit  z(a,b,c) ≈ Σ_k (α_k²/p²) cos(2π ω_k (a+b−c)/p).

    Logits are first centered per (a,b) to fix the softmax gauge. The cos basis is
    orthogonal over (a,b,c) ∈ {0,..,p-1}³, so OLS coefficients reduce to a 1D
    DCT-style sum on z̄(u), where u = (a+b−c) mod p. Frequencies are then added in
    descending |coef| order until R² ≥ target_r2 (or max_freqs is hit).
    """
    z_c = z - z.mean(axis=2, keepdims=True)

    a, b, c = np.meshgrid(np.arange(p), np.arange(p), np.arange(p), indexing='ij')
    u = ((a + b - c) % p).reshape(-1)
    z_flat = z_c.reshape(-1)
    sums = np.bincount(u, weights=z_flat, minlength=p)
    counts = np.bincount(u, minlength=p).astype(np.float64)
    z_bar = sums / counts

    omegas_all = np.arange(1, p // 2 + 1)
    u_axis = np.arange(p)
    cos_basis = np.cos(2.0 * np.pi * np.outer(omegas_all, u_axis) / p)
    coefs = (2.0 / p) * (cos_basis @ z_bar)

    ss_tot = float(np.sum(z_c ** 2))
    u_grid = (a + b - c) % p

    order = np.argsort(np.abs(coefs))[::-1]
    selected, sel_coefs, r2_curve = [], [], []
    pred_u = np.zeros(p)
    for idx in order:
        w = int(omegas_all[idx])
        c_k = float(coefs[idx])
        selected.append(w)
        sel_coefs.append(c_k)
        pred_u = pred_u + c_k * cos_basis[idx]
        ss_res = float(np.sum((z_c - pred_u[u_grid]) ** 2))
        r2_curve.append(1.0 - ss_res / ss_tot)
        if r2_curve[-1] >= target_r2 or len(selected) >= max_freqs:
            break

    sel_coefs_arr = np.array(sel_coefs)
    alpha_squared = sel_coefs_arr * (p ** 2)
    alphas = np.sqrt(np.maximum(alpha_squared, 0.0))
    return {
        'omegas': selected,
        'alphas': alphas,
        'alpha_squared': alpha_squared,
        'coefs': sel_coefs_arr,
        'r2': r2_curve[-1],
        'r2_curve': r2_curve,
        'all_omegas': omegas_all.tolist(),
        'all_coefs': coefs,
        'z_bar': z_bar,
    }


def reconstructed_logits(fit: dict, p: int) -> np.ndarray:
    """Reconstruct z̃(a,b,c) from a fit. Shape (p,p,p)."""
    a, b, c = np.meshgrid(np.arange(p), np.arange(p), np.arange(p), indexing='ij')
    u = (a + b - c) % p
    pred_u = np.zeros(p)
    for w, c_k in zip(fit['omegas'], fit['coefs']):
        pred_u += c_k * np.cos(2.0 * np.pi * w * np.arange(p) / p)
    return pred_u[u]
