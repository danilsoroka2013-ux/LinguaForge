# -*- coding: utf-8 -*-
"""Состояние проекта конструктора: корпус, токенизатор, модель, гиперпараметры."""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np

from core.config import (PROJECTS_DIR, load_checkpoint, save_checkpoint)
from core.data import load_text_file
from core.model import TinyGPT
from core.tokenizers import BaseTokenizer, build_tokenizer


def demo_corpus_path() -> str:
    base = getattr(sys, "_MEIPASS", None) or os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "assets", "corpus.txt")


class ProjectState:
    """Всё, что «живёт» между страницами приложения."""

    def __init__(self) -> None:
        self.dir = os.path.join(PROJECTS_DIR, "Мой проект")
        os.makedirs(self.dir, exist_ok=True)

        # данные
        self.text: str = ""
        self.text_path: Optional[str] = None
        self.val_frac: float = 0.1

        # токенизатор
        self.tokenizer: Optional[BaseTokenizer] = None
        self.tokenizer_kind: str = "char"     # char | bpe
        self.vocab_size: int = 512

        # модель
        self.model: Optional[TinyGPT] = None
        self.model_cfg: Dict[str, Any] = {
            "dim": 96,
            "n_layers": 3,
            "n_heads": 3,
            "block_size": 96,
            "ffn_mult": 2.667,
        }

        # обучение
        self.train_cfg: Dict[str, Any] = {
            "lr": 0.004,
            "steps": 3000,
            "batch_size": 16,
            "seq_len": 96,
            "warmup": 100,
            "weight_decay": 0.05,
            "grad_clip": 1.0,
            "min_lr": 0.0,
            "eval_every": 100,
            "ckpt_every": 500,
            "keep_checkpoints": 3,
            "seed": 42,
        }

        # журнал обучения (заполняет тренер)
        self.history: List[dict] = []

        self.trainer = None  # core.trainer.Trainer | None

    # ------------------------------------------------------------- данные
    def load_text(self, path: str) -> None:
        self.text = load_text_file(path)
        self.text_path = path

    def load_demo_corpus(self) -> None:
        self.load_text(demo_corpus_path())

    def corpus_stats(self) -> Dict[str, int]:
        t = self.text
        return {
            "chars": len(t),
            "words": len(t.split()),
            "lines": t.count("\n") + (1 if t and not t.endswith("\n") else 0),
        }

    # ------------------------------------------------------------- токенизация
    def build_tokenizer(self) -> BaseTokenizer:
        if not self.text.strip():
            raise ValueError("Сначала загрузите корпус на странице «Данные».")
        tok = build_tokenizer(self.tokenizer_kind, self.text,
                              max(8, self.vocab_size))
        self.tokenizer = tok
        return tok

    def ensure_tokenizer(self) -> BaseTokenizer:
        return self.tokenizer or self.build_tokenizer()

    def tokenize_corpus(self) -> np.ndarray:
        tok = self.ensure_tokenizer()
        return np.array(tok.encode(self.text), dtype=np.int32)

    # ------------------------------------------------------------- модель
    def build_model(self, force: bool = False) -> TinyGPT:
        vocab = self.tokenizer.vocab_size if self.tokenizer else 256
        existing = self.model
        need = force or existing is None or existing.vocab_size != vocab
        if need:
            from core.config import set_seed
            set_seed(self.train_cfg.get("seed", 42))
            self.model = TinyGPT(vocab_size=vocab, **self.model_cfg)
            self.history = []
        if self.tokenizer is not None and hasattr(self.tokenizer, "itos"):
            self.model.set_tokenizer_vocab(self.tokenizer.itos)
        return self.model

    def ensure_model_for_training(self) -> TinyGPT:
        self.ensure_tokenizer()
        return self.build_model(force=False)

    # ------------------------------------------------------------- сплит
    def train_val_ids(self) -> tuple:
        ids = self.tokenize_corpus()
        vf = float(self.val_frac)
        n_val = int(len(ids) * vf) if vf > 0 else 0
        n_val = min(n_val, max(0, len(ids) - 64))  # в train должно остаться >= 64
        if n_val > 0:
            return ids[:-n_val], ids[-n_val:]
        return ids, None

    # ------------------------------------------------------------- io модели
    def save_model(self, path: str) -> None:
        if self.model is None:
            raise ValueError("Модель ещё не собрана — соберите её на странице «Модель».")
        save_checkpoint(
            path,
            model_params={k: v.data for k, v in self.model.params().items()},
            config={**self.model.config_dict(),
                    "tokenizer": self.tokenizer.to_json() if self.tokenizer else None},
            extra={"val_frac": self.val_frac,
                   "history_tail": self.history[-64:]},
        )

    def load_model(self, path: str) -> None:
        data = load_checkpoint(path)
        cfg = dict(data["config"])
        tok_json = cfg.pop("tokenizer", None)
        self.model_cfg = {
            "dim": cfg.get("dim", 96),
            "n_layers": cfg.get("n_layers", 3),
            "n_heads": cfg.get("n_heads", 3),
            "block_size": cfg.get("block_size", 96),
            "ffn_mult": cfg.get("ffn_mult", 2.667),
        }
        if tok_json:
            from core.tokenizers import BaseTokenizer as BT
            self.tokenizer = BT.from_json(tok_json)
        self.model = TinyGPT(vocab_size=cfg["vocab_size"], **self.model_cfg)
        self.model.load_arrays(data["params"])
        if self.tokenizer is not None and hasattr(self.tokenizer, "itos"):
            self.model.set_tokenizer_vocab(self.tokenizer.itos)
        if data.get("extra") and isinstance(data["extra"].get("history_tail"), list):
            self.history = data["extra"]["history_tail"]
