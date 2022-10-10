# Annotated Transformer Implementation


A reproduction of [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) article.

[architecture](./architecture) contains the Transformer architecture.
- [attention.py](./architecture/attention.py) defines the attention mechanism.
- [encoder.py](./encoder.py) and [decoder.py](./decoder.py) are very short and create the encoder and decoder as a stack of layers.
- [transformer.py](./transformer.py) puts everything together into the `EncoderDecoder` class, which is a Transformer.
- [utils.py](./architecture/utils.py) contains additional architectural components such as LayerNorm and embeddings.



