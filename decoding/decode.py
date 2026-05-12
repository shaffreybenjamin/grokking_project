"""Activation extraction and supervised / semi-supervised decoding."""
import math
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge


def transformer_intermediates(model, x: torch.Tensor) -> dict:
    """Forward GrokTransformer, returning the post-embed, post-attn, and post-mlp
    residual streams (each of shape (B, n_ctx, d_model)). Mirrors the model's
    forward exactly, including the sqrt(d_model // 4) attention scaling.
    """
    with torch.no_grad():
        embed = model.embed(x) + model.pos_embed[None, :, :]
        q, k, v = model.W_Q(embed), model.W_K(embed), model.W_V(embed)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(model.d_model // 4)
        mask = model.causal_mask[None, :, :].to(scores.dtype)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        attended = torch.matmul(attn, v)
        post_attn = embed + model.W_O(attended)
        mlp_hidden = F.relu(model.mlp_in(post_attn))
        post_mlp = post_attn + model.mlp_out(mlp_hidden)
    return {'post_embed': embed, 'post_attn': post_attn, 'post_mlp': post_mlp}


def collect_transformer_activations(model, p: int) -> dict:
    """Forward all p² (a,b) pairs and slice out the four positions of interest.

    Keys:
      'POS1': post-embed at index 1 (a)         — decode ⟨v^(a)|
      'POS2': post-embed at index 2 (b)         — decode ⟨v^(b)|
      'POS3': post-attn  at index 2 (b)         — decode ⟨v^(a)| + ⟨v^(b)|
      'POS4': post-mlp   at index 2 (b)         — decode ⟨v^(a+b mod p)|
    plus 'a', 'b', 'c' index arrays of length p².
    """
    device = next(model.parameters()).device
    ag, bg = torch.meshgrid(torch.arange(p), torch.arange(p), indexing='ij')
    a, b = ag.flatten().to(device), bg.flatten().to(device)
    bos = torch.full_like(a, p)
    x = torch.stack([bos, a, b], dim=1)
    inter = transformer_intermediates(model, x)
    return {
        'POS1': inter['post_embed'][:, 1, :].cpu().numpy(),
        'POS2': inter['post_embed'][:, 2, :].cpu().numpy(),
        'POS3': inter['post_attn'][:, 2, :].cpu().numpy(),
        'POS4': inter['post_mlp'][:, 2, :].cpu().numpy(),
        'a': a.cpu().numpy(),
        'b': b.cpu().numpy(),
        'c': ((a + b) % p).cpu().numpy(),
    }


def mlp_intermediates(model, a: torch.Tensor, b: torch.Tensor) -> dict:
    """Forward ChughtaiMLP, returning pre- and post-ReLU hidden states."""
    with torch.no_grad():
        a_emb = F.embedding(a, model.W_a.T)
        b_emb = F.embedding(b, model.W_b.T)
        pre_relu = a_emb + b_emb
        post_relu = F.relu(pre_relu)
    return {'pre_relu': pre_relu, 'post_relu': post_relu}


def collect_mlp_activations(model, p: int) -> dict:
    """Forward all p² (a,b) pairs through ChughtaiMLP.

    Keys:
      'M1': pre-ReLU  (W_a a + W_b b)           — analogue of POS3
      'M2': post-ReLU                            — analogue of POS4
    plus 'a', 'b', 'c' index arrays of length p².
    """
    device = next(model.parameters()).device
    ag, bg = torch.meshgrid(torch.arange(p), torch.arange(p), indexing='ij')
    a, b = ag.flatten().to(device), bg.flatten().to(device)
    inter = mlp_intermediates(model, a, b)
    return {
        'M1': inter['pre_relu'].cpu().numpy(),
        'M2': inter['post_relu'].cpu().numpy(),
        'a': a.cpu().numpy(),
        'b': b.cpu().numpy(),
        'c': ((a + b) % p).cpu().numpy(),
    }


def ridge_decode(X: np.ndarray, Y: np.ndarray, alpha: float = 1e-3) -> dict:
    """Ridge regression X → Y with intercept. Returns total R², per-output-dim R²,
    coefficients, intercept and predictions."""
    reg = Ridge(alpha=alpha, fit_intercept=True)
    reg.fit(X, Y)
    Y_pred = reg.predict(X)
    Y_atleast2 = Y if Y.ndim == 2 else Y[:, None]
    Y_pred_2 = Y_pred if Y_pred.ndim == 2 else Y_pred[:, None]
    ss_res = np.sum((Y_atleast2 - Y_pred_2) ** 2, axis=0)
    ss_tot = np.sum((Y_atleast2 - Y_atleast2.mean(axis=0, keepdims=True)) ** 2, axis=0)
    r2_per_dim = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)
    return {
        'r2': float(1.0 - ss_res.sum() / max(ss_tot.sum(), 1e-12)),
        'r2_per_dim': r2_per_dim,
        'coefs': reg.coef_,
        'intercept': reg.intercept_,
        'predictions': Y_pred,
    }


def supervised_decode(activations: np.ndarray, targets: np.ndarray,
                      alpha: float = 1e-3) -> dict:
    """Ridge: full activations → α-weighted predictive vectors."""
    return ridge_decode(activations, targets, alpha=alpha)


def semisupervised_decode(activations: np.ndarray, targets: np.ndarray,
                          n_pcs: int = 10, alpha: float = 1e-3) -> dict:
    """PCA → top-n_pcs → ridge. Targets are typically the unweighted EHMM vectors
    (α_k = 1) so that the PCs aren't asked to recover the per-frequency amplitudes
    that the logit fit already extracted.
    """
    n_components = min(n_pcs, activations.shape[0], activations.shape[1])
    pca = PCA(n_components=n_components)
    Z = pca.fit_transform(activations)
    res = ridge_decode(Z, targets, alpha=alpha)
    res['pca'] = pca
    res['pca_coords'] = Z
    res['explained_variance_ratio'] = pca.explained_variance_ratio_
    return res
