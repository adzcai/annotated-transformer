import torch
import torch.nn as nn
import torch.nn.functional as F

from utils_model import clones, LayerNorm, SublayerConnection


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        encoded = self.encode(src, src_mask)
        return self.decode(encoded, src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        src_embeds = self.src_embed(src)
        return self.encoder(src_embeds, src_mask)

    def decode(self, encoded, src_mask, tgt, tgt_mask):
        tgt_embeds = self.tgt_embed(tgt)
        return self.decoder(tgt_embeds, encoded, src_mask, tgt_mask)


class Generator(nn.Module):
    def __init__(self, d_model: int, n_vocab: int):
        super(Generator, self).__init__()

        self.unembedding = nn.Linear(d_model, n_vocab)

    def forward(self, outputs: torch.Tensor):
        logits = self.unembedding(outputs)
        return F.log_softmax(logits, dim=-1)
