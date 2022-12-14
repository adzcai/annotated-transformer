from typing import Optional
import math
from copy import deepcopy

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def clones(module: nn.Module, n: int):
    return nn.ModuleList([deepcopy(module) for _ in range(n)])


def subsequent_mask(n_ctx: int, device=default_device) -> torch.BoolTensor:
    """Returns 'True' in the entries where the model is allowed to pay attention."""
    attn_shape = (1, n_ctx, n_ctx)
    return torch.tril(torch.ones(attn_shape, dtype=torch.bool, device=device))


class LayerNorm(nn.Module):
    """Estimates the mean and standard deviation of a layer's outputs."""

    def __init__(self, d_layer: int, eps=1e-6):
        super().__init__()

        self.mean_hat = nn.Parameter(torch.zeros(d_layer))
        self.std_hat = nn.Parameter(torch.ones(d_layer))
        self.eps = eps

    def forward(self, x: Tensor):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        normalized = (x - mean) / (std + self.eps)  # zero mean, unit variance
        return self.std_hat * normalized + self.mean_hat


class SublayerConnection(nn.Module):
    def __init__(self, d_layer: int, p_dropout: float):
        super().__init__()

        self.layer_norm = LayerNorm(d_layer)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x: Tensor, sublayer: nn.Module):
        """
        Apply a residual connection to any sublayer,
        as long as the input and output of the sublayer have the same shape.
        What order should layer norm, sublayer, and dropout go in?
        Probably layer norm first, since we want to normalize the residuals
        from the previous layer.
        """

        return x + self.dropout(sublayer(self.layer_norm(x)))


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, p_dropout=0.1):
        super().__init__()

        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x: Tensor):
        hidden = F.relu(self.w_1(x))
        hidden = self.dropout(hidden)
        return self.w_2(hidden)


class Embeddings(nn.Module):
    def __init__(self, d_model: int, n_vocab: int):
        super().__init__()

        self.lookup_table = nn.Embedding(n_vocab, d_model)
        self.d_model = d_model

    def forward(self, x: Tensor):
        """
        TODO Why do we scale the embeddings by the model size? Probably some "training stability" argument
        """
        return self.lookup_table(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """Add a fixed positional encoding to give the model information about token ordering."""

    def __init__(self, d_model: int, p_dropout: Optional[float] = 0, n_ctx=4096):
        super().__init__()

        self.dropout = nn.Dropout(p_dropout)
        pe = get_fixed_positional_embeddings(
            d_model, n_ctx
        ).detach()  # we don't want these to update
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor):
        """
        :param x: a Tensor of shape (batch, n_ctx, d_model)
        :return: the Tensor with positional embeddings and dropout added
        """
        x = x + self.pe[:, : x.size(1)]
        x = self.dropout(x)
        return x


def get_fixed_positional_embeddings(d_model: int, n_ctx: int):
    """
    Returns a (n_ctx, d_model) matrix where the odd-indexed embedding dimensions are distributed as a sine wave
    and the even ones are distributed as a cosine.
    """
    pe = torch.empty(n_ctx, d_model)  # matrix to store positional embeddings
    position = torch.arange(0, n_ctx).unsqueeze(1)  # (n_ctx, 1)

    evens = torch.arange(0, d_model, 2)
    # calculating exponents is faster with exp than pow
    div_term = torch.exp(evens * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)  # (1, n_ctx, d_model)
    return pe
