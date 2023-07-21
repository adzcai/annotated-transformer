from torch import Tensor
import torch.nn as nn

from .attention import AttentionFunction
from .utils import clones, LayerNorm, SublayerConnection


class Decoder(nn.Module):
    """
    The half of the transformer that also pays attention to the encoder's outputs.
    """

    def __init__(self, decoder_layer_module: nn.Module, n_layers: int):
        super(Decoder, self).__init__()

        self.layers = clones(decoder_layer_module, n_layers)
        self.norm = LayerNorm(decoder_layer_module.d_layer)

    def forward(self, encoded: Tensor, x: Tensor, src_mask: Tensor, tgt_mask: Tensor):
        """
        :param encoded: (n_batches, n_ctx, d_model)
        :param x: (n_batches, n_ctx, d_model) (batch first)
        :param src_mask: (1, n_ctx, n_ctx)
        :param tgt_mask: (1, n_ctx, n_ctx)
        """
        for layer in self.layers:
            x = layer(encoded, x, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """
    A self-attention (over the target tokens),
    source attention (over the encoder's outputs),
    followed by a feedforward layer.
    """

    def __init__(
        self,
        d_layer: int,
        src_attn: AttentionFunction,
        self_attn: AttentionFunction,
        feed_forward: nn.Module,
        p_dropout: float,
    ):
        super(DecoderLayer, self).__init__()

        self.d_layer = d_layer
        self.src_attn = src_attn
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.self_attn_connection, self.src_attn_connection, self.ff_connection = [
            SublayerConnection(d_layer, p_dropout) for _ in range(3)
        ]

    def forward(self, encoded: Tensor, x: Tensor, src_mask: Tensor, tgt_mask: Tensor):
        """
        :param encoded: the outputs from the encoder. (n_batches, n_ctx, d_model)
        :param x: the target embeddings / residual stream. (n_batches, n_ctx, d_model)
        :param src_mask: a mask over the encoder outputs. (1, n_ctx, n_ctx)
        :param tgt_mask: a mask over the target embeddings. (1, n_ctx, n_ctx)
        """

        def self_attn_with_mask(y):
            """Wrap the mask into a closure. Pay attention to the target embeddings"""
            return self.self_attn(y, y, y, tgt_mask)

        def src_attn_with_mask(y):
            """
            Wrap the encoded embeddings and the source mask into a closure.
            Pay attention to the source embeddings.
            """
            return self.src_attn(y, encoded, encoded, src_mask)

        # pay attention to own (target) tokens, then to the encoded source tokens
        x = self.self_attn_connection(x, self_attn_with_mask)
        x = self.src_attn_connection(x, src_attn_with_mask)
        return self.ff_connection(x, self.feed_forward)
