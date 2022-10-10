import torch
import torch.nn as nn

from .attention import AttentionFunction
from .utils import clones, LayerNorm, SublayerConnection


class Encoder(nn.Module):
    """Just a stack of encoder layers followed by a layer norm."""

    def __init__(self,
                 layer_module: nn.Module,
                 n_layers: int):
        super(Encoder, self).__init__()

        self.layers = clones(layer_module, n_layers)
        self.norm = LayerNorm(layer_module.d_layer)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """Pass the input through each layer, preserving the mask."""
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    """A pair of sub-layers: a self-attention layer and a feed-forward layer."""
    
    def __init__(self,
                 d_layer: int,
                 self_attn: AttentionFunction,
                 feed_forward: nn.Module,
                 p_dropout: float):
        super(EncoderLayer, self).__init__()

        self.d_layer = d_layer
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.attn_connection, self.ff_connection = [
            SublayerConnection(d_layer, p_dropout) for _ in range(2)
        ]

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        def self_attn_with_mask(y):
            """Wrap the mask into a closure."""
            return self.self_attn(y, y, y, mask)

        x = self.attn_connection(x, self_attn_with_mask)
        return self.ff_connection(x, self.feed_forward)
