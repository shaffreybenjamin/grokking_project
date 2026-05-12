"""Phase 1: transformer belief geometry analysis (Poncini 2025, Section 5)."""
import numpy as np
import matplotlib.pyplot as plt

from decoding.fit_logits import transformer_logit_tensor, fit_frequencies
from decoding.predictive_vectors import (
    ehmm_predictive_vectors,
    ehmm_predictive_vectors_unweighted,
    simplex_predictive_vectors,
)
from decoding.decode import (
    collect_transformer_activations,
    supervised_decode,
    semisupervised_decode,
)
from decoding.per_frequency import (
    single_circle_target,
    sum_of_circles_target,
    per_frequency_decode,
)


POSITIONS = ['POS1', 'POS2', 'POS3', 'POS4']
COLOR_INDEX = {'POS1': 'a', 'POS2': 'b', 'POS3': 'c', 'POS4': 'c'}
TARGET_INDEX = {'POS1': 'a', 'POS2': 'b', 'POS4': 'c'}


def run_transformer_phase1(model, p: int, target_r2: float = 0.99,
                           max_freqs: int = 10, n_pcs: int = 10,
                           ridge_alpha: float = 1e-3) -> dict:
    """Full Phase 1 pipeline: fit logits, decode predictive vectors at all four
    positions (supervised + semi-supervised + simplex control), per-frequency grid.
    """
    z = transformer_logit_tensor(model, p)
    fit = fit_frequencies(z, p, target_r2=target_r2, max_freqs=max_freqs)
    omegas, alphas = fit['omegas'], fit['alphas']

    v = ehmm_predictive_vectors(omegas, alphas, p)
    v_unit = ehmm_predictive_vectors_unweighted(omegas, p)
    e_p = simplex_predictive_vectors(p)

    acts = collect_transformer_activations(model, p)
    a, b, c = acts['a'], acts['b'], acts['c']

    targets_full = {'POS1': v[a], 'POS2': v[b], 'POS3': v[a] + v[b], 'POS4': v[c]}
    targets_unit = {'POS1': v_unit[a], 'POS2': v_unit[b],
                    'POS3': v_unit[a] + v_unit[b], 'POS4': v_unit[c]}

    decoding = {}
    for pos in POSITIONS:
        decoding[pos] = {
            'supervised': supervised_decode(acts[pos], targets_full[pos], alpha=ridge_alpha),
            'semisupervised': semisupervised_decode(acts[pos], targets_unit[pos],
                                                    n_pcs=n_pcs, alpha=ridge_alpha),
        }
    decoding['POS4_simplex_control'] = supervised_decode(acts['POS4'], e_p[c], alpha=ridge_alpha)

    target_builder = {
        'POS1': lambda w: single_circle_target(a, w, p),
        'POS2': lambda w: single_circle_target(b, w, p),
        'POS3': lambda w: sum_of_circles_target(a, b, w, p),
        'POS4': lambda w: single_circle_target(c, w, p),
    }
    per_freq = {pos: per_frequency_decode(
                    acts[pos],
                    {w: target_builder[pos](w) for w in omegas},
                    kind='supervised', alpha=ridge_alpha,
                ) for pos in POSITIONS}

    return {'p': p, 'fit': fit, 'predictive_vectors': v,
            'activations': acts, 'decoding': decoding, 'per_frequency': per_freq}


def summarize_phase1(result: dict) -> None:
    fit = result['fit']
    print(f"Logit fit:  R² = {fit['r2']:.4f}  ({len(fit['omegas'])} frequencies)")
    print(f"  ω_k         = {fit['omegas']}")
    print(f"  α_k         = {[f'{x:.3f}' for x in fit['alphas']]}")
    print(f"  R² curve    = {[f'{x:.3f}' for x in fit['r2_curve']]}")
    print()
    print("Decoding R² by position:")
    for pos in POSITIONS:
        sup = result['decoding'][pos]['supervised']['r2']
        semi = result['decoding'][pos]['semisupervised']['r2']
        print(f"  {pos}:  supervised = {sup:.4f}   semi-supervised = {semi:.4f}")
    ctrl = result['decoding']['POS4_simplex_control']['r2']
    print(f"  POS4 simplex control (RRMod_p):  {ctrl:.4f}")
    print()
    print("Per-frequency supervised R²:")
    for pos in POSITIONS:
        row = ',  '.join(f'ω={w}: {res["r2"]:.3f}'
                         for w, res in result['per_frequency'][pos].items())
        print(f"  {pos}:  {row}")


def plot_position_pca(result: dict, position: str, color_by: str = None, ax=None):
    """Top-2-PC scatter at one position, colored by the relevant index."""
    color_idx_name = color_by or COLOR_INDEX[position]
    Z = result['decoding'][position]['semisupervised']['pca_coords'][:, :2]
    color_idx = result['activations'][color_idx_name]
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=color_idx, cmap='hsv', s=8, alpha=0.7)
    r2 = result['decoding'][position]['semisupervised']['r2']
    ev = result['decoding'][position]['semisupervised']['explained_variance_ratio'][:2].sum()
    ax.set_title(f"{position}: R²={r2:.3f}, EV(top 2 PCs)={ev:.2%}")
    plt.colorbar(sc, ax=ax, label=color_idx_name)
    ax.set_aspect('equal')
    return ax


def plot_per_frequency_pairs(result: dict, position: str, color_by: str = None,
                             n_pcs: int = 10, ridge_alpha: float = 1e-3,
                             cmap: str = 'viridis', figsize=None):
    """Replicate Poncini Fig 3/4/5: rows = fitted frequencies, columns = (theoretical
    predictive vector, PCA→ridge fitted prediction). Each panel is a 2D scatter
    colored by the relevant index. Uses semi-supervised decode (PCA→ridge) per
    frequency, so R² isn't capped by n_pcs as long as n_pcs ≥ 2."""
    p = result['p']
    omegas = result['fit']['omegas']
    acts = result['activations']

    color_idx_name = color_by or COLOR_INDEX[position]
    color_idx = acts[color_idx_name]
    activations = acts[position]

    if position == 'POS3':
        a_arr, b_arr = acts['a'], acts['b']
        target_fn = lambda w: sum_of_circles_target(a_arr, b_arr, w, p)
    else:
        idx = acts[TARGET_INDEX[position]]
        target_fn = lambda w: single_circle_target(idx, w, p)

    n = len(omegas)
    fig, axes = plt.subplots(n, 2, figsize=figsize or (7, 3 * n), squeeze=False)
    last_sc = None
    for i, w in enumerate(omegas):
        target = target_fn(w)
        res = semisupervised_decode(activations, target, n_pcs=n_pcs, alpha=ridge_alpha)
        pred = res['predictions']
        for col, (data, label) in enumerate(
            [(target, f'Theoretical: ω={w}'),
             (pred, f'PCA-Fitted: ω={w}, R²={res["r2"]:.3f}')]
        ):
            sc = axes[i, col].scatter(data[:, 0], data[:, 1],
                                      c=color_idx, cmap=cmap, s=8, alpha=0.7)
            axes[i, col].set_aspect('equal')
            axes[i, col].set_title(label)
            if i == n - 1:
                axes[i, col].set_xlabel('cos component')
            if col == 0:
                axes[i, col].set_ylabel('sin component')
            last_sc = sc
    fig.colorbar(last_sc, ax=axes, label=f'{color_idx_name} mod {p}', shrink=0.6)
    fig.suptitle(f'{position} — per-frequency semi-supervised analysis')
    return fig


def plot_phase1_summary(result: dict, figsize=(12, 11)):
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    for ax, pos in zip(axes.flatten(), POSITIONS):
        plot_position_pca(result, pos, ax=ax)
    plt.tight_layout()
    return fig


def plot_per_frequency_grid(result: dict, figsize=None):
    """Heatmap: rows = positions, columns = fitted frequencies, cells = supervised R²."""
    omegas = result['fit']['omegas']
    grid = np.array([[result['per_frequency'][pos][w]['r2'] for w in omegas]
                     for pos in POSITIONS])
    fig, ax = plt.subplots(figsize=figsize or (1.2 + 0.7 * len(omegas), 4))
    im = ax.imshow(grid, vmin=0, vmax=1, aspect='auto', cmap='viridis')
    ax.set_xticks(range(len(omegas)))
    ax.set_xticklabels([f'ω={w}' for w in omegas], rotation=45)
    ax.set_yticks(range(len(POSITIONS)))
    ax.set_yticklabels(POSITIONS)
    for i in range(len(POSITIONS)):
        for j in range(len(omegas)):
            ax.text(j, i, f'{grid[i, j]:.2f}', ha='center', va='center',
                    color='white' if grid[i, j] < 0.5 else 'black', fontsize=9)
    plt.colorbar(im, ax=ax, label='R²')
    ax.set_title('Per-frequency supervised R² (transformer)')
    plt.tight_layout()
    return fig
