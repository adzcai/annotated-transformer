# Optimization criteria and training.

import torch
from torch import Tensor
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR


class LabelSmoothing(nn.Module):
    """
    Label smoothing penalizes the model for being overconfident.
    """

    def __init__(self, n_classes: int, padding_idx: int, smoothing=0.0):
        """
        :param smoothing: The model's confidence in the correct model will be the complement of this.
                          That is, the higher it is, the lower the model's confidence in the correct class.
        """
        super(LabelSmoothing, self).__init__()

        self.criterion = nn.KLDivLoss(reduction="sum")  # TODO why not cross-entropy?
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.n_classes = n_classes  # set to n_vocab in NLP context
        self.true_dist = None

    def forward(self, x: Tensor, target: Tensor):
        """
        Use some tricky PyTorch to distribute the confidence evenly.

        :param x: a Tensor of shape (n_batch, n_classes)
        :param target: n_batch
        :return: the loss
        """
        assert x.size(1) == self.n_classes

        # don't backprop through this. set probability of incorrect labels
        # there are n_classes - 2 incorrect classes, ignoring the true class and padding
        # in total these should have probability self.smoothing
        true_dist = torch.ones_like(x).detach() * self.smoothing / (self.n_classes - 2)
        # insert self.confidence into the true_dist matrix at the indices given by target
        true_dist.scatter_(dim=1, index=target.unsqueeze(1), value=self.confidence)
        true_dist[:, self.padding_idx] = 0  # and set padding token to zero probability

        # ignore padding tokens in the target (set to zero probability)
        mask = torch.nonzero(target == self.padding_idx)
        if mask.dim() > 0:  # nonempty
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist)


class DummyOptimizer(torch.optim.Optimizer):
    """Used in eval loops."""
    def __init__(self):
        super(torch.optim.Optimizer, self).__init__()

        self.param_groups = [{"lr": 0}]

    def step(self):
        pass

    def zero_grad(self, set_to_none=False):
        pass


class DummyScheduler(object):
    """Used in eval loops."""
    def step(self):
        pass


def get_lr(step: int, d_model: int, scale: float, n_warmup_steps: int):
    """
    Gets the learning rate at the given step.
    Set 0th step to be equal to the 1st step to avoid division by 0 in LambdaLR scheduler.
    """
    if step == 0:
        step = 1
    lr = d_model ** (-0.5) * min(step ** (-0.5), step * n_warmup_steps ** (-1.5))
    return scale * lr


def get_scheduler(
    model: nn.Module, d_model=None, scale=1.0, n_warmup_steps=400, lr=0.5
):
    """
    :return: The default Adam optimizer and the LambdaLR scheduler
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)

    if d_model is None:
        d_model = model.src_embed[0].d_model

    def lr_lambda(step: int):
        return get_lr(step, d_model, scale, n_warmup_steps)

    scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda)

    return optimizer, scheduler


class SimpleLoss(object):
    """A simple function for computing the loss."""

    def __init__(self, generator, criterion):
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x: Tensor, y: Tensor) -> Tensor:
        """
        :param x:    A Tensor of shape (batch, sequence, vocab).
        :param y:    The Tensor containing the true tokens. A tensor of shape (batch, sequence).
        :param norm: The number of "items" (ie tokens) that this loss should be distributed over.
        :return:     The total summed loss across all of the items.
        """
        x = self.generator(x)

        loss = self.criterion(x.reshape(-1, x.size(-1)), y.reshape(-1))

        return loss
