"""
Neural models: MLP baseline and TabTransformer
Member 2 (Model Engineer).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


# MLP baseline
class MLP(nn.Module):
    """Simple feed-forward model using the prepared feature matrix."""

    def __init__(self, n_features: int, n_classes: int = 3,
                 hidden_dims=(128, 64), dropout: float = 0.3):
        super().__init__()
        layers, prev = [], n_features
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h),
                       nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x_cont, x_cat=None):
        return self.net(x_cont)


# TabTransformer
class TransformerBlock(nn.Module):
    """Transformer block used by TabTransformer."""

    def __init__(self, dim: int, n_heads: int, ff_mult: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout,
                                          batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim * ff_mult, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop(attn_out)
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


class TabTransformer(nn.Module):
    """Transformer model for categorical and continuous features.

    Parameters
    ----------
    cat_cardinalities : number of categories for each categorical feature
    n_continuous      : number of continuous features
    embed_dim         : embedding size
    n_heads           : attention heads
    n_blocks          : transformer layers
    """

    def __init__(self, cat_cardinalities: list[int], n_continuous: int,
                 n_classes: int = 3, embed_dim: int = 32, n_heads: int = 8,
                 n_blocks: int = 6, attn_dropout: float = 0.1,
                 ff_dropout: float = 0.1, mlp_hidden=(128, 64),
                 mlp_dropout: float = 0.3):
        super().__init__()
        self.n_cat = len(cat_cardinalities)
        self.n_continuous = n_continuous
        self.embed_dim = embed_dim

        # embedding for each categorical column
        self.embeddings = nn.ModuleList(
            [nn.Embedding(c, embed_dim) for c in cat_cardinalities])

        # helps the model identify different feature columns
        self.column_embed = nn.Parameter(
            torch.randn(self.n_cat, embed_dim) * 0.02)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, dropout=attn_dropout)
            for _ in range(n_blocks)])

        self.cont_norm = nn.LayerNorm(n_continuous) if n_continuous else None

        head_in = self.n_cat * embed_dim + n_continuous
        layers, prev = [], head_in
        for h in mlp_hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h),
                       nn.ReLU(), nn.Dropout(mlp_dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.head = nn.Sequential(*layers)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, x_cont, x_cat):
        # create feature tokens
        tokens = torch.stack(
            [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)], dim=1)
        tokens = tokens + self.column_embed.unsqueeze(0)

        for block in self.blocks:
            tokens = block(tokens)

        flat = tokens.flatten(start_dim=1)

        if self.n_continuous:
            flat = torch.cat([flat, self.cont_norm(x_cont)], dim=1)

        return self.head(flat)

    def attention_maps(self, x_cont, x_cat):
        """Return attention values from transformer blocks."""
        tokens = torch.stack(
            [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)], dim=1)
        tokens = tokens + self.column_embed.unsqueeze(0)

        maps = []
        for block in self.blocks:
            h = block.norm1(tokens)
            out, w = block.attn(h, h, h, need_weights=True)
            maps.append(w.detach().cpu().numpy())
            tokens = tokens + block.drop(out)
            tokens = tokens + block.drop(block.ff(block.norm2(tokens)))

        return maps


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_all_seeds(seed: int) -> None:
    """Set random seeds for reproducible runs."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False