"""Phase 2: MLP belief geometry analysis."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from decoding.fit_logits import mlp_logit_tensor, fit_frequencies
from decoding.predictive_vectors import (
    ehmm_predictive_vectors,
    ehmm_predictive_vectors_unweighted,
    simplex_predictive_vectors,
)
from decoding.decode import (
    collect_mlp_activations,
    supervised_decode,
    semisupervised_decode,
)
from decoding.per_frequency import (
    single_circle_target,
    sum_of_circles_target,
    per_frequency_decode,
)


POSITIONS = ['M1', 'M2']
COLOR_INDEX = {'M1': 'c', 'M2': 'c'}
CLASS_INDEX = {'M1': 'a', 'M2': 'c'}


def run_mlp_phase2(model, p: int, target_r2: float = 0.99,
                   max_freqs: int = 10, n_pcs: int = 10,
                   ridge_alpha: float = 1e-3) -> dict:
    """Full Phase 2 pipeline: fit logits, decode predictive vectors at M1 (pre-ReLU)
    and M2 (post-ReLU), supervised + semi-supervised + simplex control + per-frequency.
    """
    z = mlp_logit_tensor(model, p)
    fit = fit_frequencies(z, p, target_r2=target_r2, max_freqs=max_freqs)
    omegas, alphas = fit['omegas'], fit['alphas']

    v = ehmm_predictive_vectors(omegas, alphas, p)
    v_unit = ehmm_predictive_vectors_unweighted(omegas, p)
    e_p = simplex_predictive_vectors(p)

    acts = collect_mlp_activations(model, p)
    a, b, c = acts['a'], acts['b'], acts['c']

    targets_full = {'M1': v[a] + v[b], 'M2': v[c]}
    targets_unit = {'M1': v_unit[a] + v_unit[b], 'M2': v_unit[c]}

    decoding = {}
    for pos in POSITIONS:
        decoding[pos] = {
            'supervised': supervised_decode(acts[pos], targets_full[pos], alpha=ridge_alpha),
            'semisupervised': semisupervised_decode(acts[pos], targets_unit[pos],
                                                    n_pcs=n_pcs, alpha=ridge_alpha),
        }
    decoding['M2_simplex_control'] = supervised_decode(acts['M2'], e_p[c], alpha=ridge_alpha)

    per_freq = {
        'M1': per_frequency_decode(
            acts['M1'], {w: sum_of_circles_target(a, b, w, p) for w in omegas},
            kind='supervised', alpha=ridge_alpha,
        ),
        'M2': per_frequency_decode(
            acts['M2'], {w: single_circle_target(c, w, p) for w in omegas},
            kind='supervised', alpha=ridge_alpha,
        ),
    }

    return {'p': p, 'fit': fit, 'predictive_vectors': v,
            'activations': acts, 'decoding': decoding, 'per_frequency': per_freq}


def summarize_phase2(result: dict) -> None:
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
    ctrl = result['decoding']['M2_simplex_control']['r2']
    print(f"  M2 simplex control (RRMod_p):  {ctrl:.4f}")
    print()
    print("Per-frequency supervised R²:")
    for pos in POSITIONS:
        row = ',  '.join(f'ω={w}: {res["r2"]:.3f}'
                         for w, res in result['per_frequency'][pos].items())
        print(f"  {pos}:  {row}")


def plot_position_pca(result: dict, position: str, color_by: str = None, ax=None):
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
                             kind: str = 'supervised', class_mean: str = None,
                             n_pcs: int = 10, ridge_alpha: float = 1e-3,
                             cmap: str = 'viridis', figsize=None,
                             scatter_kwargs: dict = None):
    """Replicate Poncini Fig 3/4/5 for the MLP. M1 (pre-ReLU) decodes the additive
    target ⟨v_ω^(a)| + ⟨v_ω^(b)|; M2 (post-ReLU) decodes ⟨v_ω^(a+b mod p)|.

    kind='supervised' (default) fits the 2-D target using full-dim ridge, which is
    the right choice for spread-frequency MLPs where no individual frequency plane
    dominates the top PCs. kind='semisupervised' restricts the regression to the
    top-n_pcs principal components.

    class_mean: optional name of an index in acts ('a', 'b', 'c'). When set, the
    activations are averaged within each class before regression — collapsing the
    p² noisy samples to p class-mean points and removing per-(a,b) noise that
    would otherwise show up as ring-thickness in the predictions. Recommended
    for M2 (class_mean='c'), where post-ReLU frequency-mixing inflates the
    per-sample residual and the rings look like fuzzy discs without it. For M1,
    class_mean='a' (or 'b') collapses the rose pattern to a single-frequency
    ring in a (or b)."""
    p = result['p']
    omegas = result['fit']['omegas']
    acts = result['activations']
    scatter_kwargs = scatter_kwargs or {}

    activations = acts[position]

    if class_mean is not None:
        class_idx_arr = acts[class_mean]
        activations = class_mean_activations(activations, class_idx_arr, p)
        unique_idx = np.arange(p)
        color_idx = unique_idx
        color_idx_name = class_mean
        target_fn = lambda w: single_circle_target(unique_idx, w, p)
        default_scatter = dict(s=40, alpha=0.9)
    else:
        color_idx_name = color_by or COLOR_INDEX[position]
        color_idx = acts[color_idx_name]
        if position == 'M1':
            a_arr, b_arr = acts['a'], acts['b']
            target_fn = lambda w: sum_of_circles_target(a_arr, b_arr, w, p)
        elif position == 'M2':
            c_arr = acts['c']
            target_fn = lambda w: single_circle_target(c_arr, w, p)
        else:
            raise ValueError(f'Unknown MLP position: {position}')
        default_scatter = dict(s=8, alpha=0.7)
    default_scatter.update(scatter_kwargs)

    fit_label = 'Supervised-Fitted' if kind == 'supervised' else 'PCA-Fitted'
    suffix = f' (class-mean by {class_mean})' if class_mean else ''
    n = len(omegas)
    fig, axes = plt.subplots(n, 2, figsize=figsize or (7, 3 * n), squeeze=False)
    last_sc = None
    for i, w in enumerate(omegas):
        target = target_fn(w)
        if kind == 'supervised':
            res = supervised_decode(activations, target, alpha=ridge_alpha)
        elif kind == 'semisupervised':
            res = semisupervised_decode(activations, target, n_pcs=n_pcs, alpha=ridge_alpha)
        else:
            raise ValueError(f"Unknown kind: {kind!r}")
        pred = res['predictions']
        for col, (data, label) in enumerate(
            [(target, f'Theoretical: ω={w}'),
             (pred, f'{fit_label}: ω={w}, R²={res["r2"]:.3f}')]
        ):
            sc = axes[i, col].scatter(data[:, 0], data[:, 1],
                                      c=color_idx, cmap=cmap, **default_scatter)
            axes[i, col].set_aspect('equal')
            axes[i, col].set_title(label)
            if i == n - 1:
                axes[i, col].set_xlabel('cos component')
            if col == 0:
                axes[i, col].set_ylabel('sin component')
            last_sc = sc
    fig.colorbar(last_sc, ax=axes, label=f'{color_idx_name} mod {p}', shrink=0.6)
    fig.suptitle(f'{position} — per-frequency {kind} analysis{suffix}')
    return fig


def plot_phase2_summary(result: dict, figsize=(12, 5.5)):
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, pos in zip(axes, POSITIONS):
        plot_position_pca(result, pos, ax=ax)
    plt.tight_layout()
    return fig


def class_mean_activations(activations: np.ndarray, class_idx: np.ndarray,
                            n_classes: int) -> np.ndarray:
    """Average activations within each class. Returns (n_classes, d)."""
    out = np.zeros((n_classes, activations.shape[1]), dtype=activations.dtype)
    for k in range(n_classes):
        mask = class_idx == k
        if mask.any():
            out[k] = activations[mask].mean(axis=0)
    return out


def plot_position_classmean(result: dict, position: str, class_by: str = None,
                             ax=None, cmap: str = 'hsv'):
    """Top-2-PC scatter of class-mean activations.

    For M2 we average over all (a,b) with a+b≡c (class=c), which kills the
    per-sample noise that drowns the cross-c Fourier structure in the raw
    activation scatter. For M1 we average over b at fixed a (class=a) — the
    activation is linear in (a, b) so this isolates the per-a embedding.
    """
    p = result['p']
    class_idx_name = class_by or CLASS_INDEX[position]
    activations = result['activations'][position]
    class_idx = result['activations'][class_idx_name]
    means = class_mean_activations(activations, class_idx, p)
    means_centered = means - means.mean(axis=0, keepdims=True)
    pca = PCA(n_components=2)
    Z = pca.fit_transform(means_centered)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=np.arange(p), cmap=cmap, s=40)
    ev = pca.explained_variance_ratio_[:2].sum()
    ax.set_title(f"{position}: class-mean by {class_idx_name}, "
                 f"EV(top 2 PCs)={ev:.2%}")
    plt.colorbar(sc, ax=ax, label=class_idx_name)
    ax.set_aspect('equal')
    return ax


def plot_phase2_summary_classmean(result: dict, figsize=(12, 5.5)):
    """Class-averaged version of plot_phase2_summary. Far cleaner than the raw
    p²-point scatter for spread-frequency MLPs: per-sample noise integrates out
    and the dominant Fourier mode emerges as a ring in the top-2-PC plane."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, pos in zip(axes, POSITIONS):
        plot_position_classmean(result, pos, ax=ax)
    plt.tight_layout()
    return fig


def plot_per_frequency_grid(result: dict, figsize=None):
    omegas = result['fit']['omegas']
    grid = np.array([[result['per_frequency'][pos][w]['r2'] for w in omegas]
                     for pos in POSITIONS])
    fig, ax = plt.subplots(figsize=figsize or (1.2 + 0.7 * len(omegas), 3))
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
    ax.set_title('Per-frequency supervised R² (MLP)')
    plt.tight_layout()
    return fig
