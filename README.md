# Annotated Transformer Implementation


A reproduction of [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) article.

Highly annotated for pedagogical purposes.

[architecture](./transformer) contains the Transformer architecture.
- [attention.py](./transformer/attention.py) defines the attention mechanism.
- [encoder.py](./transformer/encoder.py) and [decoder.py](./transformer/decoder.py) are very short and create the encoder and decoder as a stack of layers.
- [transformer.py](./transformer/transformer.py) puts everything together into the `EncoderDecoder` class, which is a Transformer.
- [utils.py](./transformer/utils.py) contains additional architectural components such as LayerNorm and embeddings.


