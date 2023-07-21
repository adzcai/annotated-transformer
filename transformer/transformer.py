import torch
from torch import Tensor
import torch.nn as nn
from copy import deepcopy as c
from dataclasses import dataclass

from .attention import MultiHeadedAttention
from .decoder import DecoderLayer, Decoder
from .encoder import Encoder, EncoderLayer
from .utils import (
    subsequent_mask,
    FeedForward,
    PositionalEncoding,
    Embeddings,
)


class EncoderDecoder(nn.Module):
    """
    The encoder and decoder stacks of the Transformer,
    forming the full Transformer architecture.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def encode(self, src: Tensor, src_mask: Tensor):
        """
        Encode the input and mask tokens according to src_mask.
        :param src: (n_batches, n_ctx, n_vocab)
        :param src_mask: (1, n_ctx, n_ctx)
        """
        src_embeds = self.src_embed(src)
        return self.encoder(src_embeds, src_mask)

    def decode(self, encoded: Tensor, tgt: Tensor, src_mask: Tensor, tgt_mask: Tensor):
        """
        Decode the input using the masked target sequence and the masked outputs of the encoder.
        :param encoded, tgt: (n_batches, n_ctx, d_model)
        :param src_mask, tgt_mask: (1, n_ctx, n_ctx)
        """
        tgt_embeds = self.tgt_embed(tgt)
        return self.decoder(encoded, tgt_embeds, src_mask, tgt_mask)

    def forward(self, src: Tensor, tgt: Tensor, src_mask: Tensor, tgt_mask: Tensor):
        """
        :param src, tgt: (n_batches, n_ctx, d_model)
        """
        encoded = self.encode(src, src_mask)
        return self.decode(encoded, tgt, src_mask, tgt_mask)

    def share_embeddings(self):
        """Set the source, target, and generator embedding weights to be the same."""
        weights = self.tgt_embed[0].lut.weight
        self.src_embed[0].lut.weight = weights
        self.generator.lut.weight = weights

    def greedy_decode(self, src, src_mask, n_ctx, start_token):
        """
        Autoregressively generate tokens from the source
        """
        encoded = self.encode(src, src_mask)
        # (1, outputs) the zeroth dimension is one since only one "batch", we append along first dimension
        outputs = torch.tensor([[start_token]], dtype=src.dtype)

        for _ in range(n_ctx - 1):
            tgt_mask = subsequent_mask(outputs.size(1))
            out = self.decode(encoded, outputs, src_mask, tgt_mask)

            # generate distribution over next token
            probs = self.generator(out[:, -1])
            _, next_word = torch.max(probs, dim=1)  # greedily select next word
            next_word = next_word.item()

            # append the chosen token
            outputs = torch.cat(
                [outputs, torch.tensor([[next_word]], dtype=src.dtype)], dim=1
            )

        return outputs


class Generator(nn.Module):
    """
    Turn model outputs (in embedding space) into probabilities over vocab tokens.
    """

    def __init__(self, d_model: int, n_vocab: int):
        super(Generator, self).__init__()

        self.unembedding = nn.Linear(d_model, n_vocab)

    def forward(self, outputs: torch.Tensor):
        """
        :param outputs: (..., d_model)
        :return: (..., n_vocab) distribution over token
        """
        logits = self.unembedding(outputs)
        return torch.log_softmax(logits, dim=-1)


@dataclass
class ModelConfig(object):

    n_src_vocab: int
    n_tgt_vocab: int
    n_layers: int = 6
    d_model: int = 512
    d_ff: int = 2048
    n_heads: int = 8
    p_dropout: float = 0.1
    pad_token: int = 2


def make_model(cfg: ModelConfig):
    attn = MultiHeadedAttention(cfg.n_heads, cfg.d_model, cfg.p_dropout)
    ff = FeedForward(cfg.d_model, cfg.d_ff, cfg.p_dropout)
    pe = PositionalEncoding(cfg.d_model, cfg.p_dropout)

    encoder_layer = EncoderLayer(cfg.d_model, c(attn), c(ff), cfg.p_dropout)
    encoder = Encoder(encoder_layer, cfg.n_layers)

    decoder_layer = DecoderLayer(cfg.d_model, c(attn), c(attn), c(ff), cfg.p_dropout)
    decoder = Decoder(decoder_layer, cfg.n_layers)

    src_embed, tgt_embed = [
        nn.Sequential(Embeddings(cfg.d_model, n_vocab), c(pe))
        for n_vocab in (cfg.n_src_vocab, cfg.n_tgt_vocab)
    ]

    generator = Generator(cfg.d_model, cfg.n_tgt_vocab)

    model = EncoderDecoder(encoder, decoder, src_embed, tgt_embed, generator)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model
