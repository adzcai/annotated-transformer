import torch
import torch.nn as nn
from copy import deepcopy as c
from dataclasses import dataclass

from .attention import MultiHeadedAttention
from .decoder import DecoderLayer, Decoder
from .encoder import Encoder, EncoderLayer
from .utils import (
    clones,
    subsequent_mask,
    LayerNorm,
    SublayerConnection,
    FeedForward,
    PositionalEncoding,
    Embeddings,
)


class EncoderDecoder(nn.Module):
    """
    The encoder and decoder stacks of the Transformer.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def encode(self, src, src_mask):
        """Encode the masked input."""
        src_embeds = self.src_embed(src)
        return self.encoder(src_embeds, src_mask)

    def decode(self, encoded, tgt, src_mask, tgt_mask):
        """Decode the input using the masked target sequence and the masked outputs of the encoder."""
        tgt_embeds = self.tgt_embed(tgt)
        return self.decoder(encoded, tgt_embeds, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask, tgt_mask):
        encoded = self.encode(src, src_mask)
        return self.decode(encoded, tgt, src_mask, tgt_mask)

    def share_embeddings(self):
        """Set the source, target, and generator embedding weights to be the same."""
        weights = self.tgt_embed[0].lut.weight
        self.src_embed[0].lut.weight = weights
        self.generator.lut.weight = weights

    def greedy_decode(self, src, src_mask, n_ctx, start_token):
        encoded = self.encode(src, src_mask)
        outputs = torch.ones(1, 1, dtype=src.dtype) * start_token

        for i in range(n_ctx - 1):
            tgt_mask = subsequent_mask(outputs.size(1))
            out = self.decode(encoded, outputs, src_mask, tgt_mask)

            probs = self.generator(out[:, -1])
            _, next_word = torch.max(probs, dim=1)  # greedily select next word
            next_word = next_word.item()

            outputs = torch.cat(
                [outputs, torch.ones(1, 1, dtype=src.dtype) * next_word], dim=1
            )

        return outputs


class Generator(nn.Module):
    def __init__(self, d_model: int, n_vocab: int):
        super(Generator, self).__init__()

        self.unembedding = nn.Linear(d_model, n_vocab)

    def forward(self, outputs: torch.Tensor):
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
