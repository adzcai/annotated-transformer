from typing import Optional, List, Tuple

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.functional import pad

from transformer import EncoderDecoder
from utils_model import subsequent_mask

global max_src_in_batch, max_tgt_in_batch


class Batch(object):
    def __init__(self, src, tgt=None, pad_token=0):
        self.src = src
        self.src_mask = (src != pad_token).unsqueeze(-2)

        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_labels = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad_token)
            self.n_tokens = torch.sum(self.tgt_labels != pad_token)

    @staticmethod
    def make_std_mask(tgt: Tensor, pad_token):
        """

        :param tgt: a Tensor of shape (batch, vocab, sequence)
        :param pad_token:
        :return:
        """

        tgt_mask = (tgt != pad_token).unsqueeze(-2)  # add the batch dimension
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1))
        return tgt_mask


def batch_size_fn(new_batch: Batch, count: int, so_far: int):
    """
    Batch the sentence pairs such that each batch contains the maximum number of tokens.

    :param new_batch:
    :param count:
    :param so_far: Included for torchtext compatibility.
    :return:
    """
    global max_src_in_batch, max_tgt_in_batch

    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0

    max_src_in_batch = max(max_src_in_batch, len(new_batch.src))
    max_tgt_in_batch = max(max_tgt_in_batch, len(new_batch.tgt) + 2)  # since we add a meta token at the beginning

    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch

    return max(src_elements, tgt_elements)


def transform_text(text, tokenize, vocab, device, n_ctx=2048, bos_token=0, eos_token=1, pad_token=2):
    """transform input sentences to include metadata tokens"""
    bos_token, eos_token = [torch.tensor([token], device=device) for token in (bos_token, eos_token)]
    src = torch.tensor(vocab(tokenize(text)), dtype=torch.long, device=device)
    wrapped = torch.cat([bos_token, src, eos_token], dim=0)
    return pad(wrapped, (0, n_ctx - len(wrapped)), value=pad_token)
    

def collate_batch(batch: List[Tuple[str, str]],
                  src_tokenize,
                  tgt_tokenize,
                  src_vocab,
                  tgt_vocab,
                  device,
                  n_ctx=2048,
                  eos_token=0,
                  bos_token=1,
                  pad_token=2):
    src_list, tgt_list = zip([
        (
            transform_text(src, src_tokenize, src_vocab, device, n_ctx, bos_token, eos_token, pad_token),
            transform_text(tgt, tgt_tokenize, tgt_vocab, device, n_ctx, bos_token, eos_token, pad_token)
        )
        for src, tgt in batch
    ])
    
    return torch.stack(src_list, tgt_list)


def get_loader(iter_data, batch_size, collate_fn, is_distributed=True):
    """Change the iterable-style Dataset to a map-style one since DistributedSampler requires a length"""
    data_map = to_map_style_dataset(iter_data)
    sampler = DistributedSampler(data_map) if is_distributed else None
    
    DataLoader(
        data_map,
        batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=collate_fn
    )
    return data_map
