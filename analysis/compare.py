"""Phase 3: cross-architecture comparison of belief geometry."""
import numpy as np
import matplotlib.pyplot as plt


def compare_fits(transformer_result: dict, mlp_result: dict) -> dict:
    t, m = transformer_result['fit'], mlp_result['fit']
    return {
        'transformer': {'omegas': t['omegas'], 'alphas': list(t['alphas']), 'r2': t['r2']},
        'mlp':         {'omegas': m['omegas'], 'alphas': list(m['alphas']), 'r2': m['r2']},
        'shared_omegas': sorted(set(t['omegas']) & set(m['omegas'])),
    }


def compare_decoding(transformer_result: dict, mlp_result: dict) -> dict:
    t = transformer_result['decoding']
    m = mlp_result['decoding']
    return {
        'transformer': {pos: t[pos]['supervised']['r2'] for pos in ['POS1', 'POS2', 'POS3', 'POS4']},
        'mlp':         {pos: m[pos]['supervised']['r2'] for pos in ['M1', 'M2']},
        'transformer_simplex_control': t['POS4_simplex_control']['r2'],
        'mlp_simplex_control':         m['M2_simplex_control']['r2'],
    }


def summarize_phase3(transformer_result: dict, mlp_result: dict) -> None:
    fits = compare_fits(transformer_result, mlp_result)
    dec = compare_decoding(transformer_result, mlp_result)
    print("Fitted Fourier modes:")
    print(f"  Transformer:  ω = {fits['transformer']['omegas']}")
    print(f"                α = {[f'{x:.3f}' for x in fits['transformer']['alphas']]}")
    print(f"  MLP        :  ω = {fits['mlp']['omegas']}")
    print(f"                α = {[f'{x:.3f}' for x in fits['mlp']['alphas']]}")
    print(f"  Shared ω    : {fits['shared_omegas']}")
    print()
    print("EHMM decoding R² (supervised):")
    for pos, r2 in dec['transformer'].items():
        print(f"  Transformer {pos}:  {r2:.4f}")
    print(f"  Transformer POS4 simplex control:  {dec['transformer_simplex_control']:.4f}")
    for pos, r2 in dec['mlp'].items():
        print(f"  MLP         {pos}:  {r2:.4f}")
    print(f"  MLP         M2 simplex control:    {dec['mlp_simplex_control']:.4f}")


def plot_alpha_comparison(transformer_result: dict, mlp_result: dict,
                          p: int, figsize=(11, 4)):
    """Bar chart of α_k² across all candidate frequencies, transformer vs MLP."""
    omegas_all = list(range(1, p // 2 + 1))

    def to_full(fit):
        full = np.zeros(len(omegas_all))
        for w, asq in zip(fit['omegas'], fit['alpha_squared']):
            full[omegas_all.index(w)] = asq
        return full

    t_arr = to_full(transformer_result['fit'])
    m_arr = to_full(mlp_result['fit'])

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(omegas_all))
    w = 0.4
    ax.bar(x - w / 2, t_arr, w, label='Transformer')
    ax.bar(x + w / 2, m_arr, w, label='MLP')
    step = max(1, len(omegas_all) // 14)
    ax.set_xticks(x[::step])
    ax.set_xticklabels([str(o) for o in omegas_all[::step]], rotation=45)
    ax.set_xlabel('ω')
    ax.set_ylabel('α²')
    ax.set_title('Fitted Fourier amplitudes: transformer vs MLP')
    ax.legend()
    plt.tight_layout()
    return fig


def plot_decoding_comparison(transformer_result: dict, mlp_result: dict, figsize=(8, 4)):
    """Side-by-side bar chart of supervised EHMM-decoding R² at every available
    position, plus the simplex-control bars at the rightmost positions."""
    t_dec = transformer_result['decoding']
    m_dec = mlp_result['decoding']
    rows = [
        ('T:POS1', t_dec['POS1']['supervised']['r2'], 'tab:blue'),
        ('T:POS2', t_dec['POS2']['supervised']['r2'], 'tab:blue'),
        ('T:POS3', t_dec['POS3']['supervised']['r2'], 'tab:blue'),
        ('T:POS4', t_dec['POS4']['supervised']['r2'], 'tab:blue'),
        ('T:POS4 (simplex)', t_dec['POS4_simplex_control']['r2'], 'tab:gray'),
        ('M:M1', m_dec['M1']['supervised']['r2'], 'tab:orange'),
        ('M:M2', m_dec['M2']['supervised']['r2'], 'tab:orange'),
        ('M:M2 (simplex)', m_dec['M2_simplex_control']['r2'], 'tab:gray'),
    ]
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar([r[0] for r in rows], [r[1] for r in rows],
           color=[r[2] for r in rows])
    ax.axhline(0.99, ls='--', color='k', alpha=0.4)
    ax.set_ylabel('R²')
    ax.set_ylim(0, 1.05)
    ax.set_title('EHMM (and HMM-simplex control) decoding R²')
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    plt.tight_layout()
    return fig
