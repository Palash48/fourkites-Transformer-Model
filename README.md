# French → English Transformer, trained from scratch

An encoder–decoder Transformer (9.9M parameters) trained **from scratch** on 902,783 cleaned
`Helsinki-NLP/opus-100` (en-fr) sentence pairs, with our own SentencePiece BPE tokenizer.
No pretrained translation model, language model or tokenizer is used anywhere.

The whole project is one tutorial notebook: **[`fr_en_transformer.ipynb`](fr_en_transformer.ipynb)**.
It covers data, tokenizer, model, training, decoding, sliced error analysis, test predictions and model export.

- **Model on the Hugging Face Hub:** https://huggingface.co/palashsharma048/fr-en-transformer-from-scratch
- **Training curves and config (W&B):** https://wandb.ai/palash048-geoiq/fr-en-transformer-from-scratch/runs/erhjg4l4

## Results (dev set, official `score.py`)

| slice | BLEU | chrF |
|---|---|---|
| seen (opus-100) | 26.75 | 42.49 |
| long | 33.09 | 58.87 |
| unseen_domain (opus_books, literature) | 18.06 | 40.77 |
| all | 27.47 | 45.08 |

**OVERALL = 37.17** (0.4·BLEU + 0.4·chrF + 0.2·chrF<sub>unseen</sub>). Copying the French source scores 16.1.

What the analysis found (details in Step 8 of the notebook):
- **Literature.** The gap is mostly literary grammar and register, not unknown words. Sentences in the
  passé simple score chrF 29.9 against 43.0 for the rest.
- **Long sentences.** The `long` slice score is inflated by English sources and partially-aligned references.
  The real long-input failure is repetition loops, which affect 13% of long outputs.

## Reproduce (one command, seeded)

```bash
pip install -r requirements.txt
./run.sh                 # full run: download data, train tokenizer + model, evaluate, write test_predictions.json
MT_DEBUG=1 ./run.sh      # smoke test on 20k pairs / 150 steps (minutes)
```

- `CONFIG["seed"] = 42` seeds the data sample, initialisation and batch order.
- Training resumes from `outputs/checkpoints/last.pt` if it exists. Delete that folder to train from scratch.
- Hardware used: Apple M1 laptop with 8 GB, **CPU**. The Apple GPU (MPS) produces NaN losses in
  `nn.TransformerEncoderLayer` training with PyTorch 2.8; see Step 0 of the notebook. 30k steps took about 11 h.
  On a CUDA GPU the notebook uses the GPU automatically. For a larger model, raise `d_model`, the layer counts
  and `tokens_per_batch` in `CONFIG`.

## Use the trained model

```python
import sys; sys.path.insert(0, "hf_model")
from inference import load, translate
model, sp, cfg = load("hf_model")
print(translate(model, sp, ["Ne t'inquiète pas, tout ira bien."], **cfg["decoding"]))
```

## Repository layout

| path | what it is |
|---|---|
| `fr_en_transformer.ipynb` | the project, as a step-by-step tutorial (Steps 0–10) |
| `test_predictions.json` | submission: English translations for the 330 test ids |
| `hf_model/` | weights (safetensors), tokenizer, config, standalone `inference.py`, model card |
| `outputs/` | logs (`train_log.csv`, `eval_log.csv`, `train_progress.log`), dev predictions, `results.json` |
| `outputs/run_10k/` | results of the same run at 10k steps, for comparison |
| `run.sh`, `requirements.txt` | one-command reproduction |
| `score.py`, `data/dev`, `data/test`, `sample_submission.json` | the task package (unchanged) |

## Task package notes

- Train / dev / in-domain test: `Helsinki-NLP/opus-100`, config `en-fr`. Unseen-domain test: `Helsinki-NLP/opus_books`,
  which was **never** used for training. Direction is fr → en.
- `inputs.jsonl` rows are `{id, source, length}`; `data/dev/labels.jsonl` rows are `{id, reference, slice}`.
- Score dev predictions with `python3 score.py --gold data/dev/labels.jsonl --pred my_dev_predictions.json`.
