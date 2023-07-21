from dataclasses import dataclass
from typing import Callable, Tuple, Sequence, Optional
import GPUtil, time, os
from pathlib import Path

import torch
from torch import Tensor
from torch import nn
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from transformer.transformer import make_model, ModelConfig
from preprocess import Batch, create_loaders, get_tokenizer, Vocabulary
from optimize import (
    SimpleLoss,
    get_scheduler,
    LabelSmoothing,
    DummyOptimizer,
    DummyScheduler,
)


@dataclass
class TrainState(object):
    """Track statistics during training process"""

    step: int = 0
    accum_step: int = 0
    samples: int = 0
    tokens: int = 0


@dataclass
class TrainingConfig(object):
    """Training configuration"""

    batch_size: int
    distributed: bool
    n_epochs: int
    accum_interval: int
    lr_init: float
    n_ctx: int
    n_warmup_steps: int
    file_prefix: str


def run_epoch(
    data_loader: Sequence[Batch],
    model: nn.Module,
    loss_compute: Callable[[Tensor, Tensor, int], Tuple[Tensor, Tensor]],
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None,
    mode="train",
    accum_interval=1,
    log_interval=50,
    train_state=TrainState(),
    desc: Optional[str] = None,
):
    """
    :param data_loader:
    :param model:          The model to train. Assumes it is already in proper train / eval mode.
    :param loss_compute:   Calculates the loss from the model's output logits and the target labels.
                           Returns the total loss across all the tokens and the normed loss.
    :param optimizer:      The optimizer for updating the model parameters.
    :param scheduler:      The learning rate scheduler.
    :param mode:           "train" to train the model, "eval" to only run evaluation.
    :param accum_interval: The number of gradients to accumulate before running the optimizer to update the model weights.
    :param log_interval:   The number of timesteps to iterate between logging.
    :return:               The average loss per token and the train state containing the number of steps,
                           accumulation steps, samples, and tokens.
    """
    start_time = time.time()
    desc = desc + " " if desc is not None else ""
    total_tokens = 0
    total_loss = 0
    running_tokens = 0
    n_accum = 0

    if optimizer is None:
        optimizer = DummyOptimizer()
    if scheduler is None:
        scheduler = DummyScheduler()

    for i, batch in enumerate(data_loader):
        out = model(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
        loss = loss_compute(out, batch.tgt_labels)
        loss_per_token = loss / batch.n_tokens

        if mode == "train":
            loss_per_token.backward()

            train_state.step += 1
            train_state.samples += batch.src.size(0)
            train_state.tokens += batch.n_tokens

            if i % accum_interval == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_accum += 1
                train_state.accum_step += 1

            scheduler.step()

        total_loss += loss.item()
        total_tokens += batch.n_tokens
        running_tokens += batch.n_tokens

        if i % log_interval == log_interval - 1 and mode == "train":
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start_time

            print(
                f"{desc}Epoch step: {i + 1} | "
                + f"Accumulation step: {n_accum} | "
                + f"Loss (per token): {loss_per_token.item():.3f} | "
                + f"Tokens per sec: {(running_tokens / elapsed):.3f} | "
                + f"Learning rate: {lr:.3f}"
            )

            start_time = time.time()
            running_tokens = 0

        del loss
        del loss_per_token

    return total_loss / total_tokens, train_state


def train_worker(
    gpu: int,
    n_gpus_per_node: int,
    vocab_src: Vocabulary,
    vocab_tgt: Vocabulary,
    spacy_src,
    spacy_tgt,
    model_config: ModelConfig,
    train_config: TrainingConfig,
    is_distributed=False,
):
    """
    Training process for a single GPU.
    The source and target vocab lengths in the model config should match the ones provided here.
    """

    print(f"Training using GPU {gpu}")

    model = make_model(model_config)
    model.cuda(gpu)

    if is_distributed:
        dist.init_process_group("nccl", rank=gpu, world_size=n_gpus_per_node)
        model = DDP(model, device_ids=[gpu])
        module = model.module
        is_main_process = gpu == 0
    else:
        module = model
        is_main_process = True

    criterion = LabelSmoothing(
        n_classes=len(vocab_tgt), padding_idx=model_config.pad_token, smoothing=0.1
    )
    criterion.cuda(gpu)

    train_loader, valid_loader = create_loaders(
        gpu,
        get_tokenizer(spacy_src),
        get_tokenizer(spacy_tgt),
        vocab_src,
        vocab_tgt,
        batch_size=train_config.batch_size // n_gpus_per_node,
        n_ctx=train_config.n_ctx,
        is_distributed=is_distributed,
    )

    def batch_generator(loader: DataLoader):
        for src, tgt in loader:
            yield Batch(src, tgt, model_config.pad_token)

    def save_if_main(suffix: str):
        if is_main_process:
            file_path = Path(train_config.file_prefix + suffix + ".pt")
            file_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(module.state_dict(), file_path)
            print(f"Saved checkpoint to {file_path}")

    loss_fn = SimpleLoss(module.generator, criterion)
    optimizer, scheduler = get_scheduler(
        model,
        d_model=model_config.d_model,
        lr=train_config.lr_init,
        n_warmup_steps=train_config.n_warmup_steps,
    )

    train_state = TrainState()

    for epoch in range(train_config.n_epochs):
        if is_distributed:
            train_loader.sampler.set_epoch(epoch)
            valid_loader.sampler.set_epoch(epoch)

        # ==============================

        print(f"[GPU {gpu}] Epoch {epoch} training")

        model.train()

        _, train_state = run_epoch(
            batch_generator(train_loader),
            model,
            loss_fn,
            optimizer,
            scheduler,
            mode="train",
            accum_interval=train_config.accum_interval,
            train_state=train_state,
            desc=f"[GPU {gpu}]",
        )

        GPUtil.showUtilization()

        save_if_main(str(epoch))

        torch.cuda.empty_cache()

        # ==============================

        print(f"[GPU {gpu}] Epoch {epoch} validation")

        with torch.no_grad():
            model.eval()

            loss_per_token, _ = run_epoch(
                batch_generator(valid_loader),
                model,
                loss_fn,
                mode="eval",
                desc=f"[GPU {gpu}]",
            )

        print(f"Average loss per token: {loss_per_token}")

        torch.cuda.empty_cache()

    save_if_main("final")

    print("Done training!")


def train_distributed_model(
    vocab_src,
    vocab_tgt,
    spacy_src,
    spacy_tgt,
    model_config: ModelConfig,
    train_config: TrainingConfig,
):

    n_gpus = torch.cuda.device_count()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"

    print(f"Detected {n_gpus} GPUs")
    print("Spawning training processes...")

    mp.spawn(
        train_worker,
        nprocs=n_gpus,
        args=(  # must match the order of arguments to train_worker exactly
            n_gpus,
            vocab_src,
            vocab_tgt,
            spacy_src,
            spacy_tgt,
            model_config,
            train_config,
            True,
        ),
    )


def train_model(
    vocab_src,
    vocab_tgt,
    spacy_src,
    spacy_tgt,
    model_cfg: ModelConfig,
    train_cfg: TrainingConfig,
):
    if train_cfg.distributed:
        train_distributed_model(
            vocab_src, vocab_tgt, spacy_src, spacy_tgt, model_cfg, train_cfg
        )
    else:
        train_worker(
            0,  # gpu
            1,  # n_gpus_per_node
            vocab_src,
            vocab_tgt,
            spacy_src,
            spacy_tgt,
            model_cfg,
            train_cfg,
            False,
        )


def ensemble(model, models):
    """Merge past models into the given model"""
    params = [m.params() for m in [model] + models]
    for ps in zip(*params):
        ps[0].copy_(torch.sum(*ps[1:]) / len(ps[1:]))
