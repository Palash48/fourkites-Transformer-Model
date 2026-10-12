---
language: [fr, en]
license: mit
tags: [translation, transformer, from-scratch]
datasets: [Helsinki-NLP/opus-100]
---

# French → English Transformer, trained from scratch

A small encoder–decoder Transformer (9.9M parameters) trained **from scratch** (no pretrained weights) on
902,783 cleaned sentence pairs from `Helsinki-NLP/opus-100` (en-fr, train split).

## Architecture
- Encoder–decoder, 4 + 4 layers, d_model 256, 4 heads, FFN 1024, pre-LayerNorm
- Sinusoidal position encodings; one embedding matrix shared by encoder, decoder and output layer
- Tokenizer: SentencePiece BPE, 10,000 pieces shared between French and English, byte fallback
- Training: 30,000 steps × ~4,096 tokens, AdamW, inverse-sqrt LR (peak 0.0008), label smoothing 0.1, dropout 0.1; weights = last checkpoint
- Decoding: beam 5, alpha 1.0

## Usage
```python
from inference import load, translate      # inference.py is in this repo
model, sp, cfg = load(".")
print(translate(model, sp, ["Ne t'inquiète pas, tout ira bien."], **cfg["decoding"]))
```

## Results (challenge dev set, lower-cased BLEU / chrF from the official `score.py`)
| slice | n | BLEU | chrF |
|---|---|---|---|
| seen | 60 | 26.75 | 42.49 |
| long | 30 | 33.09 | 58.87 |
| unseen_domain | 60 | 18.06 | 40.77 |
| all | 150 | 27.47 | 45.08 |

OVERALL (0.4·BLEU + 0.4·chrF + 0.2·chrF_unseen) = **37.17**

## Limitations
- Small model trained on a laptop: translations are often fluent but lose meaning on complex sentences.
- Trained only on opus-100 (subtitles / web / official text). Quality drops on literature (`unseen_domain`): archaic vocabulary, names and dialogue punctuation.
- Long inputs are split into sentences when `split_long=True`; a single very long sentence is still difficult.
- Not suitable for any use where translation errors matter.
