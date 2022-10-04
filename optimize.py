from typing import Optional
import torch
from torch import Tensor
import torch.nn as nn


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


class NoamOpt(object):
    """Learning rate scheduler."""

    def __init__(self, d_model: int, factor: float, n_warmup_steps: int, optimizer):
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

        if loss.requires_grad:
            loss.backward()

        if self.scheduler is not None:
            self.scheduler.step()
            self.scheduler.optimizer.zero_grad()

        return loss.item() * norm
