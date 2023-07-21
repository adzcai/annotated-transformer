import spacy, os
from tqdm import tqdm, trange

from transformer.transformer import ModelConfig
from train import train_model, TrainingConfig
from preprocess import build_vocab


def main():
    spacy_de = spacy.load("de_core_news_sm")
    spacy_en = spacy.load("en_core_web_sm")

    vocab_src, vocab_tgt = build_vocab(spacy_de, spacy_en)

    model_cfg = ModelConfig(n_src_vocab=len(vocab_src), n_tgt_vocab=len(vocab_tgt))

    train_cfg = TrainingConfig(
        batch_size=32,
        distributed=True,
        n_epochs=8,
        accum_interval=10,
        lr_init=1.0,
        n_ctx=72,
        n_warmup_steps=3000,
        file_prefix="models/multi30k_epoch_",
    )

    train_model(vocab_src, vocab_tgt, spacy_de, spacy_en, model_cfg, train_cfg)


if __name__ == "__main__":
    main()
