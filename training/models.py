"""Model architectures for the matched grokking setup."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GrokConfig


class GrokTransformer(nn.Module):
    """Single-layer, single-head transformer (Nanda/Poncini setup).

    No LayerNorm, no Q/K/V/O biases. MLP biases zero-initialised. All linear weights
    init ~ N(0, 1/sqrt(d_model)) except unembed ~ N(0, 1/sqrt(d_vocab)). Attention
    scores divided by sqrt(d_model // 4) — this matches Nanda's 4-head reference
    after the single-head simplification (sqrt(32) for d_model=128). Reverting to
    sqrt(d_model) gives a soft test-acc ramp instead of the flat-then-jump shape.
    """
    def __init__(self, config: GrokConfig):
        super().__init__()
        self.p = config.p
        self.vocab_size = config.d_vocab
        self.d_model = config.d_model
        self.d_mlp = config.d_mlp
        self.n_ctx = config.n_ctx
        self.embed = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_embed = nn.Parameter(torch.randn(self.n_ctx, self.d_model) / math.sqrt(self.d_model))
        self.W_Q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_K = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_V = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_O = nn.Linear(self.d_model, self.d_model, bias=False)
        self.mlp_in = nn.Linear(self.d_model, self.d_mlp, bias=True)
        self.mlp_out = nn.Linear(self.d_mlp, self.d_model, bias=True)
        self.unembed = nn.Linear(self.d_model, self.vocab_size, bias=False)
        self.register_buffer('causal_mask', torch.tril(torch.ones(self.n_ctx, self.n_ctx)))

        nn.init.normal_(self.embed.weight, std=1.0 / math.sqrt(self.d_model))
        for layer in (self.W_Q, self.W_K, self.W_V, self.W_O, self.mlp_in, self.mlp_out):
            nn.init.normal_(layer.weight, std=1.0 / math.sqrt(self.d_model))
        nn.init.normal_(self.unembed.weight, std=1.0 / math.sqrt(self.vocab_size))
        nn.init.zeros_(self.mlp_in.bias)
        nn.init.zeros_(self.mlp_out.bias)

    def forward(self, x):
        x = self.embed(x) + self.pos_embed[None, :, :]
        q = self.W_Q(x)
        k = self.W_K(x)
        v = self.W_V(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_model // 4)
        mask = self.causal_mask[None, :, :].to(scores.dtype)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        attended = torch.matmul(attn, v)
        x = x + self.W_O(attended)
        mlp = F.relu(self.mlp_in(x))
        x = x + self.mlp_out(mlp)
        return self.unembed(x)


class ChughtaiMLP(nn.Module):
    """One-hidden-layer ReLU MLP with separate left/right embeddings.

    Forward:  logits = W_U @ ReLU(W_a @ a + W_b @ b)
    The summed embedding *is* the hidden layer, so embedding dim = hidden dim.
    """
    def __init__(self, p: int, d_hidden: int = 128):
        super().__init__()
        self.p = p
        self.d_hidden = d_hidden
        self.W_a = nn.Parameter(torch.randn(d_hidden, p) / math.sqrt(p))
        self.W_b = nn.Parameter(torch.randn(d_hidden, p) / math.sqrt(p))
        self.W_U = nn.Parameter(torch.randn(p, d_hidden) / math.sqrt(d_hidden))

    def forward(self, a, b):
        a_emb = F.embedding(a, self.W_a.T)
        b_emb = F.embedding(b, self.W_b.T)
        x = F.relu(a_emb + b_emb)
        return x @ self.W_U.T
