"""Developmental signature: per-checkpoint complexity / structure metrics.

For each saved checkpoint, fit (ω_k, α_k) freshly (NOT reused from the final
checkpoint) and record several scalar diagnostics:

    n_eff            number of frequencies needed for the logit fit to reach
                     target_r2 (capped at max_freqs if not reached)
    logit_r2         cumulative R² achieved by the fit at this checkpoint
    alpha_entropy    Shannon entropy of the normalised α² distribution
                     (low = concentrated on few ω; high = spread)
    per_freq_r2      supervised R² for each of the top-K fitted frequencies,
                     decoded at `position`
    r2_ehmm          supervised R² of the full EHMM (fitted at THIS checkpoint)
    r2_simplex       supervised R² of the simplex one-hot control

These are the signals you'd plot over training to see *when* the algorithm
formed, *which* frequencies emerged, and how concentrated the spectrum is.
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
    circle_predictive_vectors,
)
from decoding.decode import (
    collect_transformer_activations, collect_mlp_activations,
    supervised_decode,
)
from decoding.per_frequency import sum_of_circles_target, single_circle_target


# Matches the convention in analysis/trajectory.py.
_EHMM_TARGET_KIND = {
    'POS1': 'a', 'POS2': 'b', 'POS3': 'sum',
    'POS4': 'c', 'M1':   'sum', 'M2':   'c',
}


def _ehmm_target(v: np.ndarray, position: str, a, b, c) -> np.ndarray:
    kind = _EHMM_TARGET_KIND[position]
    return {'a': v[a], 'b': v[b], 'c': v[c], 'sum': v[a] + v[b]}[kind]


def _per_freq_target(omega: int, p: int, position: str, a, b, c) -> np.ndarray:
    kind = _EHMM_TARGET_KIND[position]
    if kind == 'sum':
        return sum_of_circles_target(a, b, omega, p)
    idx = {'a': a, 'b': b, 'c': c}[kind]
    return single_circle_target(idx, omega, p)


def _entropy(arr: np.ndarray) -> float:
    arr = np.maximum(np.asarray(arr, dtype=np.float64), 0.0)
    s = arr.sum()
    if s <= 0:
        return 0.0
    pk = arr / s
    pk = pk[pk > 0]
    return float(-np.sum(pk * np.log(pk)))


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


def _step_metrics(model, p, position, target_r2, max_freqs, top_k,
                  ridge_alpha, simplex_alpha, logit_fn, activation_fn) -> dict:
    """Compute all developmental metrics at one checkpoint."""
    z = logit_fn(model, p)
    fit = fit_frequencies(z, p, target_r2=target_r2, max_freqs=max_freqs)
    acts = activation_fn(model, p)
    a_arr, b_arr, c = acts['a'], acts['b'], acts['c']

    r2_curve = list(fit['r2_curve'])
    n_eff = next((i + 1 for i, r in enumerate(r2_curve) if r >= target_r2), len(r2_curve))

    H = _entropy(np.array(fit['alpha_squared']))

    omegas = list(fit['omegas'])[:top_k]
    per_freq_r2 = []
    for w in omegas:
        target = _per_freq_target(w, p, position, a_arr, b_arr, c)
        per_freq_r2.append(supervised_decode(acts[position], target, alpha=ridge_alpha)['r2'])
    while len(per_freq_r2) < top_k:
        per_freq_r2.append(np.nan)

    v = ehmm_predictive_vectors(fit['omegas'], fit['alphas'], p)
    e_p = simplex_predictive_vectors(p)
    r2_e = supervised_decode(acts[position],
                             _ehmm_target(v, position, a_arr, b_arr, c),
                             alpha=ridge_alpha)['r2']
    r2_s = supervised_decode(acts[position], e_p[c], alpha=simplex_alpha)['r2']

    return {
        'n_eff': n_eff,
        'logit_r2': r2_curve[-1],
        'alpha_entropy': H,
        'omegas': fit['omegas'],
        'alpha_squared': list(fit['alpha_squared']),
        'per_freq_r2': per_freq_r2,
        'r2_ehmm': r2_e,
        'r2_simplex': r2_s,
    }


def _developmental_sweep(run_name, model_factory, logit_fn, activation_fn,
                         position, target_r2, max_freqs, top_k,
                         ridge_alpha, simplex_alpha,
                         run_root, device, verbose):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg, _, _ = load_final_checkpoint(run_name, run_root=run_root, device=device)
    p = cfg.p

    ckpts = _list_epoch_checkpoints(run_name, run_root)
    if verbose:
        print(f'Sweeping {len(ckpts)} checkpoints with fresh per-checkpoint fits...')

    out = {k: [] for k in ['epochs', 'n_eff', 'logit_r2', 'alpha_entropy',
                            'per_freq_r2', 'r2_ehmm', 'r2_simplex',
                            'omegas', 'alpha_squared']}

    model = model_factory(cfg).to(device)
    iterator = tqdm(ckpts, desc=run_name) if verbose else ckpts
    for epoch, path in iterator:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        model.eval()
        m = _step_metrics(model, p, position, target_r2, max_freqs, top_k,
                          ridge_alpha, simplex_alpha, logit_fn, activation_fn)
        out['epochs'].append(epoch)
        for k in ['n_eff', 'logit_r2', 'alpha_entropy', 'r2_ehmm', 'r2_simplex',
                  'per_freq_r2', 'omegas', 'alpha_squared']:
            out[k].append(m[k])

    out['epochs'] = np.array(out['epochs'])
    for k in ['n_eff', 'logit_r2', 'alpha_entropy', 'r2_ehmm', 'r2_simplex']:
        out[k] = np.array(out[k])
    out['per_freq_r2'] = np.array(out['per_freq_r2'])  # (n_epochs, top_k)
    out['top_k'] = top_k
    out['position'] = position
    out['p'] = p
    out['run_name'] = run_name
    out['target_r2'] = target_r2
    out['max_freqs'] = max_freqs
    return out


def sweep_developmental_transformer(run_name: str = 'transformer_run',
                                    position: str = 'POS4',
                                    target_r2: float = 0.99,
                                    max_freqs: int = 20,
                                    top_k: int = 4,
                                    ridge_alpha: float = 1e-3,
                                    simplex_alpha: float = 1.0,
                                    run_root: str = DEFAULT_RUN_ROOT,
                                    device=None, verbose: bool = True) -> dict:
    """Per-checkpoint developmental sweep on a transformer run. Returns the
    result dict described in the module docstring."""
    return _developmental_sweep(
        run_name, GrokTransformer, transformer_logit_tensor,
        collect_transformer_activations,
        position, target_r2, max_freqs, top_k,
        ridge_alpha, simplex_alpha, run_root, device, verbose,
    )


def sweep_developmental_mlp(run_name: str = 'mlp_run',
                            position: str = 'M2',
                            target_r2: float = 0.99,
                            max_freqs: int = 20,
                            top_k: int = 4,
                            ridge_alpha: float = 1e-3,
                            simplex_alpha: float = 1.0,
                            run_root: str = DEFAULT_RUN_ROOT,
                            device=None, verbose: bool = True) -> dict:
    """Per-checkpoint developmental sweep on an MLP run."""
    def factory(cfg):
        return ChughtaiMLP(cfg.p, d_hidden=cfg.mlp_d_hidden)
    return _developmental_sweep(
        run_name, factory, mlp_logit_tensor,
        collect_mlp_activations,
        position, target_r2, max_freqs, top_k,
        ridge_alpha, simplex_alpha, run_root, device, verbose,
    )


def detect_structure_formation(sweep: dict,
                               ehmm_threshold: float = 0.95,
                               simplex_threshold: float = 0.20,
                               n_eff_threshold: int = 10) -> dict:
    """Mark each checkpoint as "structure formed" iff all three triggers fire:

        r2_ehmm    > ehmm_threshold
        r2_simplex < simplex_threshold
        n_eff      <= n_eff_threshold

    The first qualifying epoch is reported as `first_formation_epoch`. Designed
    for automated developmental-stage labelling so you don't have to inspect
    the per-frequency plots at every step.
    """
    e = sweep['epochs']
    e_mask = sweep['r2_ehmm'] > ehmm_threshold
    s_mask = sweep['r2_simplex'] < simplex_threshold
    n_mask = sweep['n_eff'] <= n_eff_threshold
    formed = e_mask & s_mask & n_mask
    formed_epochs = e[formed]
    return {
        'structure_formed': formed,
        'formed_epochs': formed_epochs,
        'first_formation_epoch': int(formed_epochs[0]) if len(formed_epochs) else None,
        'thresholds': {
            'ehmm': ehmm_threshold,
            'simplex': simplex_threshold,
            'n_eff': n_eff_threshold,
        },
    }


def plot_developmental_signature(sweep: dict, figsize=(10, 12), title_prefix: str = ''):
    """Four-panel developmental signature: logit-fit R², N_eff, α² entropy,
    top-K per-frequency R² (mean / min). Read top-to-bottom for the algorithm's
    formation story."""
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
    e = sweep['epochs']

    axes[0].plot(e, sweep['logit_r2'], 'o-', markersize=3, color='tab:green')
    axes[0].axhline(sweep['target_r2'], ls='--', color='gray', alpha=0.5,
                    label=f"target R²={sweep['target_r2']}")
    axes[0].set_ylabel('Logit-fit R²')
    axes[0].set_title(f"{title_prefix}Logit-fit quality")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].legend(loc='lower right')

    axes[1].plot(e, sweep['n_eff'], 'o-', markersize=3, color='tab:blue')
    axes[1].axhline(sweep['max_freqs'], ls=':', color='gray', alpha=0.4,
                    label=f"max_freqs={sweep['max_freqs']}")
    axes[1].set_ylabel('N_eff')
    axes[1].set_title(f"{title_prefix}Effective frequency count (freqs to reach R²={sweep['target_r2']})")
    axes[1].legend(loc='upper right')

    axes[2].plot(e, sweep['alpha_entropy'], 'o-', markersize=3, color='tab:purple')
    axes[2].set_ylabel('H(α²)  [nats]')
    axes[2].set_title(f"{title_prefix}Spectrum entropy (low = concentrated, high = spread)")

    pk = sweep['per_freq_r2']
    valid = ~np.isnan(pk)
    mean_r2 = np.where(valid.any(axis=1), np.nanmean(pk, axis=1), np.nan)
    min_r2 = np.where(valid.any(axis=1), np.nanmin(pk, axis=1), np.nan)
    axes[3].plot(e, mean_r2, 'o-', markersize=3, label=f'mean (top {sweep["top_k"]})', color='darkred')
    axes[3].plot(e, min_r2, '^-', markersize=3, label='min', color='red', alpha=0.6)
    axes[3].set_ylabel('Per-frequency R²')
    axes[3].set_xlabel('Training Epoch')
    axes[3].set_title(f"{title_prefix}Polygon cleanness at {sweep['position']}")
    axes[3].set_ylim(-0.05, 1.05)
    axes[3].legend(loc='lower right')

    plt.tight_layout()
    return fig


def plot_structure_formation(sweep: dict, detection: dict = None,
                             ax=None, title: str = None,
                             ehmm_threshold: float = 0.95,
                             simplex_threshold: float = 0.20,
                             n_eff_threshold: int = 10):
    """Single-panel plot showing R²(EHMM), R²(simplex), and N_eff together,
    with the first-formation epoch annotated by a vertical line. Computes
    `detect_structure_formation` if a detection dict isn't supplied."""
    if detection is None:
        detection = detect_structure_formation(
            sweep, ehmm_threshold=ehmm_threshold,
            simplex_threshold=simplex_threshold,
            n_eff_threshold=n_eff_threshold,
        )
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
    e = sweep['epochs']
    ax.plot(e, sweep['r2_ehmm'], 'o-', label='R²(EHMM)', color='crimson', markersize=4)
    ax.plot(e, sweep['r2_simplex'], 'o-', label='R²(simplex)', color='steelblue', markersize=4)
    ax.set_xlabel('Training Epoch')
    ax.set_ylabel('R²')
    ax.set_ylim(-0.05, 1.05)

    ax2 = ax.twinx()
    ax2.plot(e, sweep['n_eff'], 's--', color='purple', alpha=0.45,
             markersize=3, label='N_eff')
    ax2.set_ylabel('N_eff', color='purple')
    ax2.tick_params(axis='y', colors='purple')
    ax2.set_ylim(0, sweep['max_freqs'] + 0.5)

    epoch0 = detection['first_formation_epoch']
    if epoch0 is not None:
        ax.axvline(epoch0, ls='--', color='black', alpha=0.6,
                   label=f'first formation: epoch {epoch0}')

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc='center right')

    if title is None:
        title = f'{sweep["run_name"]} — structure formation at {sweep["position"]}'
    ax.set_title(title)
    return ax.figure
