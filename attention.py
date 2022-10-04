import math
from typing import Optional, Callable

import torch
import torch.nn as nn
from torch import Tensor, BoolTensor

from utils_model import clones

AttentionFunction = Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]


def attention(query: Tensor,
              key: Tensor,
              value: Tensor,
              mask: Optional[torch.BoolTensor] = None,
              dropout_module: Optional[nn.Module] = None):
    """
    Takes the generated query, key, and value matrices,
    and returns the resulting output (weighted sum of the values)
    along with the attention pattern itself.
    """
    d_key = query.size(-1)
    attn_logits = query @ key.transpose(-2, -1) / math.sqrt(d_key)

    if mask is not None:
        attn_logits.masked_fill_(mask == False, -1e9)  # negative infinity, or close enough

    attn_pattern = torch.softmax(attn_logits, dim=-1)

    if dropout_module is not None:
        attn_pattern = dropout_module(attn_pattern)

    return attn_pattern @ value, attn_pattern


class MultiHeadedAttention(nn.Module):
    def __init__(self, n_heads: int, d_model: int, p_dropout=0.1):
        super(MultiHeadedAttention, self).__init__()

        assert d_model % n_heads == 0

        self.n_heads = n_heads
        self.d_key = d_model // n_heads
        self.w_query, self.w_key, self.w_value, self.w_output = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p_dropout)

    def forward(self,
                query: Tensor,
                key: Tensor,
                value: Tensor,
                mask: Optional[BoolTensor] = None):
        if mask is not None:
            mask = mask.unsqueeze(1)  # insert additional dim for each head

        n_batches = query.size(0)

        query, key, value = [
            linear(x)
            .view(n_batches, -1, self.n_heads, self.d_key)
            .transpose(1, 2)  # end up with (batch, head, sequence, key)
            for linear, x in (
                (self.w_query, query),
                (self.w_key, key),
                (self.w_value, value),
            )
        ]

        x, self.attn = attention(query, key, value, mask=mask, dropout_module=self.dropout)

        # end up with (batch, sequence, model)
        x = (x.transpose(1, 2)
             .reshape(n_batches, -1, self.h * self.d_key))

        del query
        del key
        del value

        return self.w_output(x)
