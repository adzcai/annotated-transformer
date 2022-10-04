import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from utils_training import Batch


def run_epoch(data_loader: DataLoader[Batch], model: nn.Module, loss_fn, log_interval=50):
    """
    :param data_loader:
    :param model:
    :param loss_fn: Calculates the loss from the model's output logits and the target labels.
                    Also handles model parameter updates.
                    Returns the total loss across all the tokens.
    :param log_interval:
    :return: the average loss per token.
    """
    start = time.time()
    total_tokens = 0
    total_loss = 0
    running_tokens = 0

    for i, batch in enumerate(data_loader):
        out = model(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
        loss = loss_fn(out, batch.tgt_labels, batch.n_tokens)
        total_loss += loss
        total_tokens += batch.n_tokens
        running_tokens += batch.n_tokens

        if i % log_interval == 0:
            elapsed = time.time() - start

            print(f"Epoch step: {i + 1} | "
                  + f"Loss (per token): {loss / batch.n_tokens} | "
                  + f"Tokens per sec: {running_tokens / elapsed}")

            start = time.time()
            running_tokens = 0

    return total_loss / total_tokens


def data_gen(n_vocab: int, batch_size: int, n_batches: int):
    for i in range(n_batches):
        data = torch.randint(1, n_vocab, size=(batch_size, 10)).detach()
        data[:, 0] = 1
        yield Batch(src=data, tgt=data, pad_token=0)

        
        