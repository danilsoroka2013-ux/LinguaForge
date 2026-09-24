# -*- coding: utf-8 -*-
"""Токенизаторы: символьный и BPE (Byte Pair Encoding).

BPE обучается на корпусе за секунды для небольших текстов, кодирует быстро
через кэш слов. Оба сериализуются в JSON (внутри чекпоинта модели).
"""
from __future__ import annotations

import json
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

WORD_RE = re.compile(r"\w+|\s+|[^\w\s]", re.UNICODE)


class BaseTokenizer:
    name = "base"

    # ---- интерфейс -----------------------------------------------------
    def train(self, corpus: str, vocab_size: int) -> None:
        raise NotImplementedError

    def encode(self, text: str) -> List[int]:
        raise NotImplementedError

    def decode(self, ids: List[int]) -> str:
        raise NotImplementedError

    # ---- сериализация ---------------------------------------------------
    def to_json(self) -> Dict:
        return {"name": self.name}

    @classmethod
    def from_json(cls, d: Dict) -> "BaseTokenizer":
        kind = d.get("name")
        if kind == "char":
            t = CharTokenizer()
        elif kind == "bpe":
            t = BPETokenizer()
        else:
            raise ValueError(f"Неизвестный токенизатор: {kind}")
        t._load(d)
        return t


class CharTokenizer(BaseTokenizer):
    """Каждый символ — отдельный токен. Специальные: 0=PAD, 1=UNK, 2=BOS, 3=EOS."""

    name = "char"
    PAD, UNK, BOS, EOS = 0, 1, 2, 3

    def __init__(self) -> None:
        self.itos: List[str] = ["<pad>", "<unk>", "<bos>", "<eos>"]
        self.stoi: Dict[str, int] = {s: i for i, s in enumerate(self.itos)}

    def train(self, corpus: str, vocab_size: int) -> None:
        counts = Counter(corpus)
        # Частые символы — первыми (это удобно для наглядности, не влияет на качество).
        for ch, _cnt in sorted(counts.items(), key=lambda kv: -kv[1]):
            if ch not in self.stoi:
                self.itos.append(ch)
                self.stoi[ch] = len(self.itos) - 1
            if len(self.itos) >= max(vocab_size, 4):
                break

    def encode(self, text: str) -> List[int]:
        unk = self.stoi.get("<unk>", 1)
        return [self.stoi.get(ch, unk) for ch in text]

    def decode(self, ids: List[int]) -> str:
        out = []
        for i in ids:
            s = self.itos[i] if 0 <= i < len(self.itos) else ""
            if s in ("<pad>",):
                continue
            if s in ("<bos>", "<eos>", "<unk>"):
                continue
            out.append(s)
        return "".join(out)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def to_json(self) -> Dict:
        return {"name": self.name, "itos": self.itos}

    def _load(self, d: Dict) -> None:
        self.itos = list(d["itos"])
        self.stoi = {s: i for i, s in enumerate(self.itos)}


class BPETokenizer(BaseTokenizer):
    """BPE поверх предтокенизации regex'ом. Спецтокены: PAD/UNK/BOS/EOS."""

    name = "bpe"
    PAD, UNK, BOS, EOS = 0, 1, 2, 3

    def __init__(self) -> None:
        self.merges: Dict[Tuple[int, int], int] = {}
        self.special = ["<pad>", "<unk>", "<bos>", "<eos>"]
        self.itos: List[str] = []
        self.stoi: Dict[str, int] = {}
        self._word_cache: Dict[str, List[int]] = {}

    # ---- обучение --------------------------------------------------------
    def train(self, corpus: str, vocab_size: int) -> None:
        words = WORD_RE.findall(corpus)
        if not words:
            raise ValueError("Пустой корпус для BPE")
        counts = Counter(words)
        # Каждое слово — последовательность символов; словарь начинается с символов.
        symbol_seqs: Dict[str, List[str]] = {}
        vocab: List[str] = list(self.special)
        seen_chars = set(vocab)
        for w in counts:
            sym = []
            for ch in w:
                if ch not in seen_chars:
                    seen_chars.add(ch)
                    vocab.append(ch)
                sym.append(ch)
            symbol_seqs[w] = sym

        target = max(8, vocab_size)
        num_merges = max(0, target - len(vocab))
        for _ in range(num_merges):
            pair_counts: Counter = Counter()
            for w, freq in counts.items():
                seq = symbol_seqs[w]
                for a, b in zip(seq, seq[1:]):
                    pair_counts[(a, b)] += freq
            if not pair_counts:
                break
            best, _ = max(pair_counts.items(), key=lambda kv: kv[1])
            new_sym = best[0] + best[1]
            if new_sym in vocab:
                # уже есть (странно, но бывает) — пропускаем, чтобы не зациклиться
                merge_id = len(vocab) + len(self.merges)
            else:
                vocab.append(new_sym)
                merge_id = len(vocab) - 1
            self.merges[best] = merge_id
            for w, freq in counts.items():
                seq = symbol_seqs[w]
                if len(seq) < 2:
                    continue
                out: List[str] = []
                i = 0
                while i < len(seq):
                    if (i < len(seq) - 1 and seq[i] == best[0]
                            and seq[i + 1] == best[1]):
                        out.append(new_sym)
                        i += 2
                    else:
                        out.append(seq[i])
                        i += 1
                symbol_seqs[w] = out

        self.itos = vocab
        self.stoi = {s: i for i, s in enumerate(vocab)}
        self._word_cache = {}

    # ---- кодирование -----------------------------------------------------
    def _encode_word(self, word: str) -> List[int]:
        cached = self._word_cache.get(word)
        if cached is not None:
            return cached
        unk = self.stoi.get("<unk>", 1)
        seq: List[str] = list(word)
        while len(seq) > 1:
            best_rank = None
            best_idx = -1
            for i in range(len(seq) - 1):
                r = self.merges.get((seq[i], seq[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_idx = r, i
            if best_rank is None:
                break
            seq = seq[:best_idx] + [seq[best_idx] + seq[best_idx + 1]] + seq[best_idx + 2:]
        ids = [self.stoi.get(s, unk) for s in seq]
        self._word_cache[word] = ids
        return ids

    def encode(self, text: str) -> List[int]:
        ids: List[int] = []
        for w in WORD_RE.findall(text):
            ids.extend(self._encode_word(w))
        return ids

    def decode(self, ids: List[int]) -> str:
        out: List[str] = []
        for i in ids:
            if 0 <= i < len(self.itos):
                s = self.itos[i]
                if s in self.special:
                    continue
                out.append(s)
            else:
                out.append("")
        return "".join(out)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def to_json(self) -> Dict:
        return {
            "name": self.name,
            "itos": self.itos,
            "merges": [[a, b, i] for (a, b), i in self.merges.items()],
        }

    def _load(self, d: Dict) -> None:
        self.itos = list(d["itos"])
        self.stoi = {s: i for i, s in enumerate(self.itos)}
        self.merges = {(a, b): int(i) for a, b, i in d.get("merges", [])}
        self._word_cache = {}


def build_tokenizer(kind: str, corpus: str, vocab_size: int) -> BaseTokenizer:
    tok = CharTokenizer() if kind == "char" else BPETokenizer()
    tok.train(corpus, vocab_size)
    return tok
