import torch
import torch.nn as nn
from copy import deepcopy as c

from attention import MultiHeadedAttention
from decoder import DecoderLayer, Decoder
from encoder import Encoder, EncoderLayer
from utils_model import clones, subsequent_mask, LayerNorm, SublayerConnection, FeedForward, PositionalEncoding, Embeddings


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

    def forward(self, src, tgt, src_mask, tgt_mask):
        encoded = self.encode(src, src_mask)
        return self.decode(encoded, tgt, src_mask, tgt_mask)

    def encode(self, src, src_mask):
        src_embeds = self.src_embed(src)
        return self.encoder(src_embeds, src_mask)

    def decode(self, encoded, tgt, src_mask, tgt_mask):
        tgt_embeds = self.tgt_embed(tgt)
        return self.decoder(encoded, tgt_embeds, src_mask, tgt_mask)
    
    def greedy_decode(self, src, src_mask, n_ctx, start_token):
        encoded = self.encode(src, src_mask)
        outputs = torch.ones(1, 1, dtype=src.dtype) * start_token
        
        for i in range(n_ctx - 1):
            tgt_mask = subsequent_mask(outputs.size(1))
            out = self.decode(
                encoded,
                outputs,
                src_mask,
                tgt_mask
            )
            
            probs = self.generator(out[:, -1])
            _, next_word = torch.max(probs, dim=1)  # greedily select next word
            next_word = next_word.item()
            
            outputs = torch.cat([
                outputs,
                torch.ones(1, 1, dtype=src.dtype) * next_word
            ], dim=1)
        
        return outputs
        


class Generator(nn.Module):
    def __init__(self, d_model: int, n_vocab: int):
        super(Generator, self).__init__()

        self.unembedding = nn.Linear(d_model, n_vocab)

    def forward(self, outputs: torch.Tensor):
        logits = self.unembedding(outputs)
        return torch.log_softmax(logits, dim=-1)


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
