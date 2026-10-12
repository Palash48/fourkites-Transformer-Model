# Standalone inference for the from-scratch French->English Transformer (PyTorch + sentencepiece only).
#   from inference import load, translate
#   model, sp, cfg = load(".")
#   translate(model, sp, ["Bonjour tout le monde."], beam_size=5)
import json, math, re
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

PAD, BOS, EOS = 0, 2, 3


def normalize_text(text):
    text = re.sub(r"[   ]", " ", text)
    text = re.sub(r"[’‘‛`´]", "'", text)
    text = re.sub(r"[«»“”„]", '"', text)
    text = re.sub(r"[–—]", "-", text)
    text = text.replace("…", "...")
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text):
    return [p for p in re.split(r"(?<=[.!?;:])\s+(?=[A-ZÀ-Ý0-9\"(\-])", text) if p]


def sinusoidal_positions(max_len, d_model):
    position = torch.arange(max_len).unsqueeze(1)
    freq = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    table = torch.zeros(max_len, d_model)
    table[:, 0::2] = torch.sin(position * freq)
    table[:, 1::2] = torch.cos(position * freq)
    return table


class TranslationModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_encoder_layers, n_decoder_layers, d_ff, dropout, max_positions=1024):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.register_buffer("positions", sinusoidal_positions(max_positions, d_model), persistent=False)
        self.dropout = nn.Dropout(dropout)
        self.transformer = nn.Transformer(d_model=d_model, nhead=n_heads, num_encoder_layers=n_encoder_layers,
                                          num_decoder_layers=n_decoder_layers, dim_feedforward=d_ff, dropout=dropout,
                                          batch_first=True, norm_first=True)

    def embed_tokens(self, ids):
        return self.dropout(self.embed(ids) * math.sqrt(self.d_model) + self.positions[: ids.size(1)])

    def encode(self, src):
        return self.transformer.encoder(self.embed_tokens(src), src_key_padding_mask=(src == PAD))

    def decode(self, tgt_in, memory, src):
        n = tgt_in.size(1)
        causal = torch.triu(torch.ones(n, n, dtype=torch.bool, device=tgt_in.device), diagonal=1)
        out = self.transformer.decoder(self.embed_tokens(tgt_in), memory, tgt_mask=causal, tgt_is_causal=True,
                                       tgt_key_padding_mask=(tgt_in == PAD), memory_key_padding_mask=(src == PAD))
        return out @ self.embed.weight.T


def load(model_dir, device="cpu"):
    model_dir = Path(model_dir)
    cfg = json.load(open(model_dir / "config.json"))
    model = TranslationModel(cfg["vocab_size"], cfg["d_model"], cfg["n_heads"], cfg["n_encoder_layers"],
                             cfg["n_decoder_layers"], cfg["d_ff"], cfg["dropout"])
    model.load_state_dict(load_file(str(model_dir / "model.safetensors")))
    sp = spm.SentencePieceProcessor(model_file=str(model_dir / "spm_fr_en.model"))
    return model.to(device).eval(), sp, cfg


def strip_special(ids):
    ids = ids[: ids.index(EOS)] if EOS in ids else ids
    return [i for i in ids if i not in (PAD, BOS)]


@torch.no_grad()
def beam_search(model, src, beam_size=5, alpha=0.6, max_len=100):
    memory = model.encode(src)
    beams, finished = [([BOS], 0.0)], []
    for _ in range(max_len):
        ys = torch.tensor([s for s, _ in beams], device=src.device)
        log_probs = F.log_softmax(model.decode(ys, memory.expand(len(beams), -1, -1), src.expand(len(beams), -1))[:, -1], -1)
        scores = torch.tensor([s for _, s in beams], device=src.device)[:, None] + log_probs
        top_scores, top_ids = scores.view(-1).topk(2 * beam_size)
        new_beams = []
        for sc, flat in zip(top_scores.tolist(), top_ids.tolist()):
            b, tok = divmod(flat, log_probs.size(-1))
            (finished if tok == EOS else new_beams).append((beams[b][0] + [tok], sc))
            if len(new_beams) == beam_size:
                break
        beams = new_beams
        if len(finished) >= beam_size:
            break
    best, _ = max(finished or beams, key=lambda c: c[1] / (((5 + len(c[0])) / 6) ** alpha))
    return strip_special(best)


def translate(model, sp, texts, beam_size=5, length_penalty=0.6, split_long=False, split_long_chars=150):
    device = next(model.parameters()).device
    out = []
    for text in texts:
        text = normalize_text(text)
        parts = split_sentences(text) if split_long and len(text) > split_long_chars else [text]
        pieces = []
        for part in parts:
            src = torch.tensor([sp.encode(part)[:500] + [EOS]], device=device)
            if beam_size == 1:
                memory, ys = model.encode(src), torch.tensor([[BOS]], device=device)
                for _ in range(int(1.5 * src.size(1)) + 10):
                    nxt = model.decode(ys, memory, src)[:, -1].argmax(-1, keepdim=True)
                    ys = torch.cat([ys, nxt], 1)
                    if nxt.item() == EOS:
                        break
                ids = strip_special(ys[0, 1:].tolist())
            else:
                ids = beam_search(model, src, beam_size, length_penalty, int(1.5 * src.size(1)) + 10)
            pieces.append(sp.decode(ids))
        out.append(" ".join(pieces).strip())
    return out
