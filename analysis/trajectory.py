"""Per-checkpoint sweep: belief-geometry decoding R² across training (Poncini Fig 2).

Decodes a position-appropriate EHMM target and the HMM simplex control
⟨e_p^(c)| at every saved checkpoint:

    POS1, POS2 → ⟨v^(a)| or ⟨v^(b)|        (single-circle, single index)
    POS3, M1   → ⟨v^(a)| + ⟨v^(b)|         (additive, pre-modular-sum)
    POS4, M2   → ⟨v^(a+b mod p)|           (single-circle, post-modular-sum)

The EHMM's (ω_k, α_k) are fitted once on the final checkpoint and held fixed
across training — this measures how much of the *eventually-learned* algorithm
is present at each developmental stage.
"""
import glob
import os
import re

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from training.models import GrokTransformer, ChughtaiMLP
from training.checkpoints import load_final_checkpoint
from training.loop import DEFAULT_RUN_ROOT

from decoding.fit_logits import (
    transformer_logit_tensor, mlp_logit_tensor, fit_frequencies,
)
from decoding.predictive_vectors import (
    ehmm_predictive_vectors, simplex_predictive_vectors,
)
from decoding.decode import (
    collect_transformer_activations, collect_mlp_activations,
    supervised_decode,
)


def _list_epoch_checkpoints(run_name: str, run_root: str = DEFAULT_RUN_ROOT):
    pattern = os.path.join(run_root, run_name, 'checkpoints', 'epoch_[0-9]*.pt')
    paths = sorted(glob.glob(pattern))
    out = []
    for p in paths:
        m = re.search(r'epoch_(\d+)\.pt$', p)
        if m:
            out.append((int(m.group(1)), p))
    out.sort()
    return out


# Position → which EHMM target is appropriate. POS3 and M1 are pre-modular-sum
# layers carrying additive structure ⟨v(a)| + ⟨v(b)|; everywhere else the model
# has either picked out a single index already or has computed (a+b) mod p.
_EHMM_TARGET_KIND = {
    'POS1': 'a',
    'POS2': 'b',
    'POS3': 'sum',
    'POS4': 'c',
    'M1':   'sum',
    'M2':   'c',
}


def _ehmm_target(v: np.ndarray, position: str, a, b, c) -> np.ndarray:
    kind = _EHMM_TARGET_KIND.get(position)
    if kind == 'a':
        return v[a]
    if kind == 'b':
        return v[b]
    if kind == 'c':
        return v[c]
    if kind == 'sum':
        return v[a] + v[b]
    raise ValueError(f'Unknown position for EHMM target: {position}')


def _fit_at_final(model_factory, logit_fn, cfg, final_state, device,
                  target_r2: float, max_freqs: int):
    """Build EHMM (ω_k, α_k) by fitting Poncini's Fourier procedure on the final
    checkpoint's logits. Returns (omegas, alphas, fit_dict)."""
    model = model_factory(cfg).to(device)
    model.load_state_dict(final_state)
    model.eval()
    z = logit_fn(model, cfg.p)
    fit = fit_frequencies(z, cfg.p, target_r2=target_r2, max_freqs=max_freqs)
    return fit['omegas'], fit['alphas'], fit


def sweep_transformer(run_name: str = 'transformer_run', position: str = 'POS4',
                      omegas=None, alphas=None,
                      target_r2: float = 0.99, max_freqs: int = 10,
                      ridge_alpha: float = 1e-3, simplex_alpha: float = 1.0,
                      run_root: str = DEFAULT_RUN_ROOT, device=None,
                      verbose: bool = True) -> dict:
    """Sweep all checkpoints of a transformer run, decoding EHMM and HMM-simplex
    R² at `position` (default POS4 — post-MLP at the b position).

    If `omegas` / `alphas` are not provided, they are fitted once on the final
    checkpoint (Poncini's procedure) and held fixed across the sweep.

    The simplex target has p=113 columns and gets ill-conditioned at early /
    untrained checkpoints, producing negative R² with the EHMM ridge alpha.
    `simplex_alpha=1.0` regularises that fit independently; at converged
    checkpoints both alphas give effectively identical R² values, so this
    preserves comparability with Phase 1's POS4 simplex control.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg, final_ckpt, _ = load_final_checkpoint(run_name, run_root=run_root, device=device)
    p = cfg.p
    final_state = final_ckpt['model_state']

    if omegas is None or alphas is None:
        omegas, alphas, fit = _fit_at_final(
            GrokTransformer, transformer_logit_tensor, cfg, final_state, device,
            target_r2=target_r2, max_freqs=max_freqs,
        )
        if verbose:
            print(f'Fitted at final checkpoint:  ω = {omegas}')
            print(f'                            α = {[f"{a:.3f}" for a in alphas]}')
            print(f'                          R² = {fit["r2"]:.4f}')
    else:
        fit = None

    v = ehmm_predictive_vectors(omegas, alphas, p)
    e_p = simplex_predictive_vectors(p)

    ckpts = _list_epoch_checkpoints(run_name, run_root)
    if verbose:
        print(f'Sweeping {len(ckpts)} checkpoints...')

    epochs, r2_ehmm, r2_simplex = [], [], []
    model = GrokTransformer(cfg).to(device)
    iterator = tqdm(ckpts, desc=run_name) if verbose else ckpts
    for epoch, path in iterator:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        model.eval()
        acts = collect_transformer_activations(model, p)
        c = acts['c']
        ehmm_res = supervised_decode(acts[position],
                                     _ehmm_target(v, position, acts['a'], acts['b'], c),
                                     alpha=ridge_alpha)
        simp_res = supervised_decode(acts[position], e_p[c], alpha=simplex_alpha)
        epochs.append(epoch)
        r2_ehmm.append(ehmm_res['r2'])
        r2_simplex.append(simp_res['r2'])

    return {
        'epochs': np.array(epochs),
        'r2_ehmm': np.array(r2_ehmm),
        'r2_simplex': np.array(r2_simplex),
        'omegas': list(omegas),
        'alphas': np.array(alphas),
        'final_fit': fit,
        'position': position,
        'p': p,
        'run_name': run_name,
    }


def sweep_mlp(run_name: str = 'mlp_run', position: str = 'M2',
              omegas=None, alphas=None,
              target_r2: float = 0.99, max_freqs: int = 10,
              ridge_alpha: float = 1e-3, simplex_alpha: float = 1.0,
              run_root: str = DEFAULT_RUN_ROOT, device=None,
              verbose: bool = True) -> dict:
    """Sweep all checkpoints of an MLP run, decoding EHMM and HMM-simplex R² at
    `position` (default M2 — post-ReLU). See `sweep_transformer` for the
    `ridge_alpha` / `simplex_alpha` rationale."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg, final_ckpt, _ = load_final_checkpoint(run_name, run_root=run_root, device=device)
    p = cfg.p
    final_state = final_ckpt['model_state']

    if omegas is None or alphas is None:
        def factory(c):
            return ChughtaiMLP(c.p, d_hidden=c.mlp_d_hidden)
        omegas, alphas, fit = _fit_at_final(
            factory, mlp_logit_tensor, cfg, final_state, device,
            target_r2=target_r2, max_freqs=max_freqs,
        )
        if verbose:
            print(f'Fitted at final checkpoint:  ω = {omegas}')
            print(f'                            α = {[f"{a:.3f}" for a in alphas]}')
            print(f'                          R² = {fit["r2"]:.4f}')
    else:
        fit = None

    v = ehmm_predictive_vectors(omegas, alphas, p)
    e_p = simplex_predictive_vectors(p)

    ckpts = _list_epoch_checkpoints(run_name, run_root)
    if verbose:
        print(f'Sweeping {len(ckpts)} checkpoints...')

    epochs, r2_ehmm, r2_simplex = [], [], []
    model = ChughtaiMLP(p, d_hidden=cfg.mlp_d_hidden).to(device)
    iterator = tqdm(ckpts, desc=run_name) if verbose else ckpts
    for epoch, path in iterator:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        model.eval()
        acts = collect_mlp_activations(model, p)
        c = acts['c']
        ehmm_res = supervised_decode(acts[position],
                                     _ehmm_target(v, position, acts['a'], acts['b'], c),
                                     alpha=ridge_alpha)
        simp_res = supervised_decode(acts[position], e_p[c], alpha=simplex_alpha)
        epochs.append(epoch)
        r2_ehmm.append(ehmm_res['r2'])
        r2_simplex.append(simp_res['r2'])

    return {
        'epochs': np.array(epochs),
        'r2_ehmm': np.array(r2_ehmm),
        'r2_simplex': np.array(r2_simplex),
        'omegas': list(omegas),
        'alphas': np.array(alphas),
        'final_fit': fit,
        'position': position,
        'p': p,
        'run_name': run_name,
    }


def plot_trajectory(sweep_result: dict, ax=None, title: str = None,
                    polygon_color: str = 'crimson',
                    simplex_color: str = 'steelblue',
                    xscale: str = 'linear',
                    clip_negative: bool = True):
    """Replicate Poncini Fig 2: R² vs training epoch for polygon (EHMM) and
    simplex (HMM) targets at a single residual-stream position. With
    `clip_negative=True`, negative R² values (which can occur from ill-conditioned
    ridge fits at very-untrained checkpoints) are floored at 0 for display."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    epochs = sweep_result['epochs']
    plot_epochs = np.where(epochs == 0, 1, epochs) if xscale == 'log' else epochs
    r2_e = sweep_result['r2_ehmm']
    r2_s = sweep_result['r2_simplex']
    if clip_negative:
        r2_e = np.maximum(r2_e, 0.0)
        r2_s = np.maximum(r2_s, 0.0)
    ax.plot(plot_epochs, r2_e, 'o-', color=polygon_color, label='Polygon (EHMM)', markersize=4)
    ax.plot(plot_epochs, r2_s, 'o-', color=simplex_color, label='Simplex (HMM)', markersize=4)
    ax.set_xlabel('Training Epoch')
    ax.set_ylabel('R²-Value')
    ax.set_xscale(xscale)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='center right')
    if title is None:
        title = f'{sweep_result["run_name"]} — {sweep_result["position"]}'
    ax.set_title(title)
    return ax


def plot_trajectory_comparison(t_sweep: dict, m_sweep: dict, figsize=(13, 5),
                               xscale: str = 'linear'):
    """Side-by-side trajectory plots for the transformer and MLP runs."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    plot_trajectory(t_sweep, ax=axes[0],
                    title=f'Transformer  ({t_sweep["position"]})', xscale=xscale)
    plot_trajectory(m_sweep, ax=axes[1],
                    title=f'MLP  ({m_sweep["position"]})', xscale=xscale)
    plt.tight_layout()
    return fig
