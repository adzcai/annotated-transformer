from copy import deepcopy as c
import torch.nn as nn
from attention import MultiHeadedAttention
from decoder import DecoderLayer, Decoder
from encoder import Encoder, EncoderLayer
from transformer import EncoderDecoder, Generator
from utils_model import FeedForward, PositionalEncoding, Embeddings


def make_model(n_src_vocab: int,
               n_tgt_vocab: int,
               n_layers=6,
               d_model=512,
               d_ff=2048,
               n_heads=8,
               p_dropout=0.1):
    attn = MultiHeadedAttention(n_heads, d_model, p_dropout)
    ff = FeedForward(d_model, d_ff, p_dropout)
    pe = PositionalEncoding(d_model, p_dropout)

    encoder_layer = EncoderLayer(d_model, c(attn), c(ff), p_dropout)
    encoder = Encoder(encoder_layer, n_layers)

    decoder_layer = DecoderLayer(d_model, c(attn), c(attn), c(ff), p_dropout)
    decoder = Decoder(decoder_layer, n_layers)

    src_embed, tgt_embed = [
        nn.Sequential(Embeddings(d_model, n_vocab), c(pe))
        for n_vocab in (n_src_vocab, n_tgt_vocab)
    ]

    generator = Generator(d_model, n_tgt_vocab)

    model = EncoderDecoder(encoder, decoder, src_embed, tgt_embed, generator)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model
