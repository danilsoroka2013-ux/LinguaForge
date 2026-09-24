# -*- coding: utf-8 -*-
"""Загрузка и подготовка текстовых данных для обучения."""
from __future__ import annotations

import os
import random
from typing import Callable, List, Optional, Tuple

import numpy as np


def load_text_file(path: str) -> str:
    """Читает файл в любой разумной кодировке."""
    for enc in ("utf-8", "utf-8-sig", "cp1251"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except (UnicodeDecodeError, UnicodeError):
            continue
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def make_batches(
    ids: np.ndarray,
    batch_size: int,
    seq_len: int,
    steps: int,
    seed: int = 0,
) -> Callable[[], Tuple[np.ndarray, np.ndarray]]:
    """Возвращает бесконечный генератор случайных чанков (x, y).

    ids — плоский массив токенов. x = чанк длины seq_len, y = тот же чанк,
    сдвинутый на 1. Возвращаем функцию, чтобы шаги обучения могли идти
    бесконечно без переупорядочивания датасета.

    Оптимизация: все старты чанков батча выбираются вектором за раз,
    а (x, y) собираются одной сборкой из общего окна +1 токен.
    """
    rng = np.random.default_rng(seed)
    ids = np.asarray(ids, dtype=np.int32)
    if ids.size < seq_len + 1:
        raise ValueError(
            f"Корпус слишком короткий: {ids.size} токенов, нужно хотя бы {seq_len + 1}")

    hi = max(1, ids.size - seq_len - 1)

    def batch() -> Tuple[np.ndarray, np.ndarray]:
        starts = rng.integers(0, hi, size=batch_size)
        offs = np.arange(seq_len + 1)
        chunks = ids[starts[:, None] + offs[None, :]]      # (B, seq+1)
        return np.ascontiguousarray(chunks[:, :-1]), np.ascontiguousarray(chunks[:, 1:])

    return batch
