# Handles data preprocessing and batching.

from typing import Optional, List, Tuple, Callable

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.functional import pad
from torch.utils.data import IterableDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.functional import to_map_style_dataset
import torchtext.datasets as datasets

from architecture.transformer import EncoderDecoder
from architecture.utils import subsequent_mask

import os


global max_src_in_batch, max_tgt_in_batch

BOS_TOKEN = '<s>'
EOS_TOKEN = '</s>'
BLANK_WORD = '<blank>'
UNK_WORD = '<unk>'

Tokenizer = Callable[[str], List[str]]  # splits up a string into string tokens
Vocabulary = Callable[[List[str]], List[int]]


def get_tokenizer(spacy) -> Tokenizer:
    def tokenize(text: str):
        return [tok.text for tok in spacy.tokenizer(text)]
    return tokenize


def build_vocab(spacy_src, spacy_tgt, vocab_path='vocab.pt', force_build=False):
    """
    Build a torchtext vocabulary using the Multi30k dataset.
    spacy_src and spacy_tgt should be spacy 
    """
    
    if os.path.exists(vocab_path) and not force_build:
        vocab_src, vocab_tgt = torch.load(vocab_path)
        print(f"Loaded from {vocab_path}")
        return vocab_src, vocab_tgt
    
    train_iter, val_iter, test_iter = datasets.Multi30k(language_pair=('de', 'en'))

    def vocab_generator(index: int, tokenize: Tokenizer):
        for pair in train_iter + val_iter + test_iter:
            yield tokenize(pair[index])

    vocab_src, vocab_tgt = [
        build_vocab_from_iterator(
            tqdm(vocab_generator(i, tokenize)),
            min_freq=2,
            specials=[BOS_TOKEN, EOS_TOKEN, BLANK_WORD, UNK_WORD],
        )
        for i, tokenize in enumerate((get_tokenizer(spacy_src), get_tokenizer(spacy_tgt)))
    ]

    vocab_src.set_default_index(vocab_src[UNK_WORD])
    vocab_tgt.set_default_index(vocab_tgt[UNK_WORD])

    torch.save((vocab_src, vocab_tgt), vocab_path)
    print(f'saved vocab to {vocab_path}')
    
    return vocab_src, vocab_tgt


def get_loader(iter_data: IterableDataset,
               batch_size: int,
               collate_fn,
               is_distributed=True):
    """
    Change the iterable-style Dataset to a map-style one since DistributedSampler requires a length.
    :param batch_size: the number of sequences per batch.
    :param collate_fn: a function for collating sequences within the batch.
    :param is_distributed: whether to use a distributed sampler.
    """
    data_map = to_map_style_dataset(iter_data)
    sampler = DistributedSampler(data_map) if is_distributed else None
    
    return DataLoader(
        data_map,
        batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=collate_fn
    )


def create_loaders(device,
                   tokenize_src: Tokenizer,
                   tokenize_tgt: Tokenizer,
                   vocab_src: Vocabulary,
                   vocab_tgt: Vocabulary,
                   batch_size=12000,
                   n_ctx=128,
                   is_distributed=False):
    """
    Create the PyTorch DataLoaders for the source and target.
    """
    def collate_fn(batch):
        """A simple closure to wrap all the arguments except the batch."""
        return collate_batch(
            batch,
            tokenize_src,
            tokenize_tgt,
            vocab_src,
            vocab_tgt,
            device,
            n_ctx=n_ctx,
        )

    train_iter, valid_iter, test_iter = datasets.Multi30k(language_pair=('de', 'en'))
    return tuple(
        get_loader(data_iter,
                   batch_size,
                   collate_fn,
                   is_distributed)
        for data_iter in (train_iter, valid_iter)
    )


class Batch(object):
    """
    Hold a batch of sequences with a mask during training.
    """
    
    def __init__(self, src: Tensor, tgt: Optional[Tensor] = None, pad_token: int = 2):
        """
        :param src: a Tensor of shape (batch, position)
        :param tgt: a Tensor of shape (batch, position)
        """
        
        self.src = src
        self.src_mask = (src != pad_token).unsqueeze(-2)

        if tgt is not None:
            self.tgt = tgt[:, :-1]
            
            # ignore the first token, for the "start classification"
            self.tgt_labels = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad_token)
            self.n_tokens = torch.sum(self.tgt_labels != pad_token)

    @staticmethod
    def make_std_mask(tgt: Tensor, pad_token: int):
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


def transform_text(text: str,
                   tokenize: Tokenizer,
                   vocab: Vocabulary,
                   device,
                   n_ctx=2048,
                   bos_token=0,
                   eos_token=1,
                   pad_token=2):
    """Transform input sentences to include metadata tokens."""
    bos_token, eos_token = [
        torch.tensor([token], device=device)
        for token in (bos_token, eos_token)
    ]
    vocab_tokens = torch.tensor(vocab(tokenize(text)), dtype=torch.long, device=device)
    wrapped = torch.cat([bos_token, vocab_tokens, eos_token])  # wrap with beginning and end of sentence
    return pad(wrapped, (0, n_ctx - len(wrapped)), value=pad_token)  # pad up to n_ctx
    

def collate_batch(batch: List[Tuple[str, str]],
                  src_tokenize,
                  tgt_tokenize,
                  src_vocab,
                  tgt_vocab,
                  device,
                  n_ctx=2048,
                  eos_token=0,
                  bos_token=1,
                  pad_token=2) -> Tuple[Tensor, Tensor]:
    """
    :return: The (batch_size, n_ctx) array of tokens for both the source and target batches. 
    """
    src_list, tgt_list = [], []
    
    def closure(text, tokenize, vocab):
        return transform_text(text,
                              tokenize,
                              vocab,
                              device,
                              n_ctx,
                              bos_token,
                              eos_token,
                              pad_token)
    
    for src_text, tgt_text in batch:
        src_list.append(closure(src_text, src_tokenize, src_vocab))
        tgt_list.append(closure(tgt_text, tgt_tokenize, tgt_vocab))
    
    src, tgt = torch.stack(src_list), torch.stack(tgt_list)
    print(f"Batch device: {src.device}")
    return src, tgt
