import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


def get_fixed_positional_embeddings(d_model: int, n_ctx: int):
    """
    Get fixed positional embeddings for the input.
    """

    assert d_model % 2 == 0

    position = torch.arange(n_ctx)[None, :]
    div_term = torch.exp(
        torch.arange(0, d_model, 2) * (-math.log(10_000) / d_model)
    )
    pe = torch.empty(n_ctx, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe




class Absolute