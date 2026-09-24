# -*- coding: utf-8 -*-
"""Общие настройки и лёгкие утилиты для движка.

Здесь живут структура конфига, сиды, папки проектов и безопасное
сохранение/загрузка чекпоинтов. Никаких тяжёлых зависимостей — только
стандартная библиотека и numpy.
"""
from __future__ import annotations

import os
import json
import time
import random
from typing import Any, Dict, List, Optional

import numpy as np

APP_NAME = "LinguaForge"
VERSION = "1.0"

# Папки ---------------------------------------------------------------
HOME_DIR = os.path.join(os.path.expanduser("~"), ".linguaforge")
PROJECTS_DIR = os.path.join(HOME_DIR, "projects")
os.makedirs(PROJECTS_DIR, exist_ok=True)


def now_str() -> str:
    return time.strftime("%d.%m %H:%M")


def set_seed(seed: int) -> None:
    """Фиксирует сиды numpy/random для повторяемости."""
    random.seed(seed)
    np.random.seed(seed % (2**32))


def clip_grad_norm(grads: Dict[str, np.ndarray], max_norm: float) -> float:
    """Мягкий клиппинг по глобальной норме. Возвращает норму до клиппинга."""
    total = 0.0
    for g in grads.values():
        total += float(np.sum(g.astype(np.float64) ** 2))
    norm = float(np.sqrt(total))
    if max_norm > 0 and norm > max_norm and norm > 0:
        scale = max_norm / norm
        for k in grads:
            grads[k] = grads[k] * scale
    return norm


class Scheduler:
    """Прогрев + линейный спад learning rate до min_lr."""

    def __init__(self, base_lr: float, warmup: int, total: int, min_lr: float = 0.0):
        self.base = float(base_lr)
        self.warmup = max(0, int(warmup))
        self.total = max(1, int(total))
        self.min_lr = float(min_lr)
        self.t = 0

    def step(self) -> float:
        self.t += 1
        t = self.t
        if t <= self.warmup and self.warmup > 0:
            return self.base * t / self.warmup
        if t >= self.total:
            return self.min_lr
        frac = 1.0 - (t - self.warmup) / max(1, (self.total - self.warmup))
        return self.min_lr + (self.base - self.min_lr) * frac


# Чекпоинты ------------------------------------------------------------


def save_checkpoint(
    path: str,
    model_params: Dict[str, np.ndarray],
    config: Dict[str, Any],
    tokenizer_data: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Сохраняет веса и метаданные в один .npz (сначала во временный файл)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    arrays = {("param__" + k): v for k, v in model_params.items()}
    meta = {
        "meta__config": json.dumps(config, ensure_ascii=False),
        "meta__version": VERSION,
        "meta__saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if tokenizer_data is not None:
        meta["meta__tokenizer"] = json.dumps(tokenizer_data, ensure_ascii=False)
    if extra is not None:
        meta["meta__extra"] = json.dumps(extra, ensure_ascii=False)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays, **meta)
    os.replace(tmp, path)


def load_checkpoint(path: str) -> Dict[str, Any]:
    """Загружает чекпоинт. Возвращает dict: params, config, tokenizer, extra."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=False)
    out: Dict[str, Any] = {"params": {}, "config": {}, "tokenizer": None, "extra": None}
    for key in data.files:
        if key.startswith("param__"):
            out["params"][key[len("param__"):]] = data[key]
        elif key == "meta__config":
            out["config"] = json.loads(str(data[key]))
        elif key == "meta__tokenizer":
            out["tokenizer"] = json.loads(str(data[key]))
        elif key == "meta__extra":
            out["extra"] = json.loads(str(data[key]))
    data.close()
    return out


def list_checkpoints(project_dir: str) -> List[str]:
    if not os.path.isdir(project_dir):
        return []
    return sorted(
        (os.path.join(project_dir, f) for f in os.listdir(project_dir) if f.endswith(".lfmodel")),
        key=os.path.getmtime,
        reverse=True,
    )


def human_size(n_bytes: float) -> str:
    for unit in ("Б", "КБ", "МБ", "ГБ"):
        if abs(n_bytes) < 1024.0:
            return f"{n_bytes:.0f} {unit}" if unit == "Б" else f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024.0
    return f"{n_bytes:.1f} ТБ"
