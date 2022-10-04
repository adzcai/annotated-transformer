from torch import Tensor
import torch.nn as nn

from attention import AttentionFunction
from utils_model import clones, LayerNorm, SublayerConnection


class Decoder(nn.Module):

    def __init__(self,
                 layer_module: nn.Module,
                 n_layers: int):
        super(Decoder, self).__init__()

        self.layers = clones(layer_module, n_layers)
        self.norm = LayerNorm(layer_module.d_layer)

    def forward(self, encoded: Tensor, x: Tensor, src_mask: Tensor, tgt_mask: Tensor):
        for layer in self.layers:
            x = layer(x, encoded, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self,
                 d_layer: int,
                 self_attn: AttentionFunction,
                 src_attn: AttentionFunction,
                 feed_forward: nn.Module,
                 p_dropout: float):
        super(DecoderLayer, self).__init__()

        self.d_layer = d_layer
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.self_attn_connection, self.src_attn_connection, self.ff_connection = [
            SublayerConnection(d_layer, p_dropout) for _ in range(3)
        ]

    def forward(self, encoded: Tensor, x: Tensor, src_mask: Tensor, tgt_mask: Tensor):
        def src_attn_with_mask(y):
            """
            Wrap the encoded embeddings and the source mask into a closure.
            Pay attention to the source embeddings.
            """
            return self.src_attn(y, encoded, encoded, src_mask)

        def self_attn_with_mask(y):
            """Wrap the mask into a closure. Pay attention to the target embeddings"""
            return self.self_attn(y, y, y, tgt_mask)

        x = self.self_attn_connection(x, self_attn_with_mask)
        x = self.src_attn_connection(x, src_attn_with_mask)
        return self.ff_connection(x, self.feed_forward)
