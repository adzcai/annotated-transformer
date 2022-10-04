from typing import Optional

import torch
from torch import Tensor, nn

from transformer import EncoderDecoder
from utils_model import subsequent_mask

max_src_in_batch = 0
max_tgt_in_batch = 0


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
    max_tgt_in_batch = max(max_tgt_in_batch, len(new_batch.tgt))

    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch

    return max(src_elements, tgt_elements)


class NoamOpt(object):
    """Learning rate scheduler."""

    def __init__(self, d_model: int, factor, n_warmup_steps, optimizer):
        self.d_model = d_model
        self.factor = factor
        self.warmup = n_warmup_steps
        self.optimizer = optimizer

        self._step = 0
        self._rate = 0

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step

        return (self.factor * (
                self.d_model ** (-.5)
                * min(step ** (-.5), step * self.warmup ** (-1.5))
        ))


def get_standard_optimizer(model: EncoderDecoder):
    optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    return NoamOpt(
        model.src_embed[0].d_model,
        factor=2,
        n_warmup_steps=4000,
        optimizer=optimizer
    )


class LabelSmoothing(nn.Module):
    """
    Penalize the model for being overconfident.
    """

    def __init__(self, n_classes: int, padding_idx: int, smoothing=0.):
        super(LabelSmoothing, self).__init__()

        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1. - smoothing
        self.smoothing = smoothing
        self.n_classes = n_classes
        self.true_dist = None

    def forward(self, x: Tensor, target: Tensor):
        """
        Use some tricky PyTorch to distribute the confidence evenly.

        :param x: a Tensor of shape (batch, class)
        :param target:
        :return:
        """
        assert x.size(1) == self.n_classes

        true_dist = torch.ones_like(x).detach() * self.smoothing / (self.n_classes - 2)
        true_dist.scatter_(dim=1, index=target.unsqueeze(1), value=self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target == self.padding_idx)
        if mask.dim() > 0:  # nonempty
            true_dist.index_fill_(0, mask.squeeze(), 0.)
        self.true_dist = true_dist
        return self.criterion(x, true_dist)


class SimpleLoss(object):
    def __init__(self, generator, criterion, scheduler: Optional[NoamOpt] = None):
        self.generator = generator
        self.criterion = criterion
        self.scheduler = scheduler

    def __call__(self, x: Tensor, y: Tensor, norm: float):
        x = self.generator(x)
        loss = self.criterion(
            x.reshape(-1, x.size(-1)),
            y.reshape(-1)
        ) / norm
        loss.backward()

        if self.scheduler is not None:
            self.scheduler.step()
            self.scheduler.optimizer.zero_grad()

        return loss.item() * norm
