# -*- coding: utf-8 -*-
"""Мини-трансформер (декодер, GPT-подобный) на нашем тензоре.

Особенности:
- преднорма (pre-LN) + финальная норма — стабильно обучается без тёплых хитростей;
- RoPE (ротационные позиционные кодировки);
- SwiGLU-FFN;
- каузальное внимание через явную маску;
- вес tie: эмбеддинг входа = выходной классификатор (экономия параметров).

Всё работает в float32 (в config можно включить float64 для проверки градиентов).
"""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple

from core.tensor import Tensor
from core import ops


# ---------------------------------------------------------------- helpers

def xavier_fan_in(fan_in: int, fan_out: int) -> np.ndarray:
    limit = np.sqrt(6.0 / max(1, fan_in))
    return np.random.uniform(-limit, limit, size=(fan_in, fan_out)).astype(np.float32)


def rope_cache(seq_len: int, head_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    """Возвращает (cos, sin) формы (T, head_dim//2)."""
    half = head_dim // 2
    inv = 1.0 / (10000.0 ** (np.arange(0, half, dtype=np.float32) / max(1, half)))
    t = np.arange(seq_len, dtype=np.float32)
    freqs = np.outer(t, inv)  # (T, half)
    return np.cos(freqs), np.sin(freqs)


def apply_rope(x: Tensor, cos: np.ndarray, sin: np.ndarray) -> Tensor:
    """x: (B, T, H, D). Ротация пар размерностей внутри головы.

    cos/sin: (T, D//2). Всё через тензорные операции, чтобы градиент
    корректно возвращался в x.
    """
    half = x.shape[-1] // 2
    x1 = x[..., 0:half]
    x2 = x[..., half:]
    c = Tensor(cos[None, :, None, :].astype(np.float32))
    s = Tensor(sin[None, :, None, :].astype(np.float32))
    r1 = x1 * c - x2 * s
    r2 = x1 * s + x2 * c
    return Tensor.concatenate([r1, r2], axis=-1)


class Linear:
    def __init__(self, fan_in: int, fan_out: int, scaled: bool = True) -> None:
        scale = (1.0 / np.sqrt(fan_in)) if scaled else 1.0
        W = xavier_fan_in(fan_in, fan_out)
        W *= scale  # in-place, чтобы остаться в float32
        self.W = Tensor(W, requires_grad=True)

    def params(self) -> Dict[str, Tensor]:
        return {"W": self.W}

    def __call__(self, x: Tensor) -> Tensor:
        return x.matmul(self.W)


class Embedding:
    def __init__(self, vocab_size: int, dim: int) -> None:
        self.E = Tensor(
            np.random.randn(vocab_size, dim).astype(np.float32) * 0.02,
            requires_grad=True)

    def params(self) -> Dict[str, Tensor]:
        return {"E": self.E}

    def __call__(self, idx: np.ndarray) -> Tensor:
        return self.E.index_select(idx.reshape(-1)).reshape(
            idx.shape[0], idx.shape[1], self.E.shape[1])


class RMSNorm:
    """Предпочитает плавленую реализацию (core.ops); флаг fast=False вернёт
    наивную — она используется в градиентной проверке, где нужен float64."""

    def __init__(self, dim: int, fast: bool = True) -> None:
        self.g = Tensor(np.ones(dim, dtype=np.float32), requires_grad=True)
        self.fast = fast

    def params(self) -> Dict[str, Tensor]:
        return {"g": self.g}

    def __call__(self, x: Tensor) -> Tensor:
        if self.fast and x.data.dtype == np.float32:
            return ops.rmsnorm(x, self.g)
        ms = (x * x).mean(axis=-1, keepdims=True)
        return x * ms.rsqrt() * self.g


class CausalSelfAttention:
    def __init__(self, dim: int, n_heads: int) -> None:
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = Linear(dim, 3 * dim)
        self.proj = Linear(dim, dim)

    def params(self) -> Dict[str, Tensor]:
        p = {}
        for k, v in self.qkv.params().items():
            p["qkv." + k] = v
        for k, v in self.proj.params().items():
            p["proj." + k] = v
        return p

    def __call__(self, x: Tensor, cos: np.ndarray, sin: np.ndarray,
                 mask: np.ndarray) -> Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)  # (B,T,3C)
        # Срезы по последней оси: градиент течёт корректно через __getitem__.
        q = qkv[:, :, :C].reshape(B, T, self.n_heads, self.head_dim)
        k = qkv[:, :, C:2 * C].reshape(B, T, self.n_heads, self.head_dim)
        v = qkv[:, :, 2 * C:].reshape(B, T, self.n_heads, self.head_dim)
        # (B,H,T,hd): один физический transpose перед attn, ноль лишних копий.
        q = ops.transpose_op(q, (0, 2, 1, 3))
        k = ops.transpose_op(k, (0, 2, 1, 3))
        v = ops.transpose_op(v, (0, 2, 1, 3))
        if q.data.dtype == np.float32:
            q = ops.rope(q, cos, sin)
            k = ops.rope(k, cos, sin)
            y = ops.causal_attention(q, k, v, mask)  # (B,H,T,hd)
        else:
            # Slow-путь (float64/градчек): те же оси, наивные узлы.
            # cos/sin (T, hd/2) -> (1, 1, T, hd/2) — под оси (B, H, T, half).
            def _rope_naive(t: Tensor) -> Tensor:
                half = t.shape[-1] // 2
                t1, t2 = t[..., :half], t[..., half:]
                cN = cos[None, None]  # (1, 1, T, half)
                sN = sin[None, None]
                if t.data.dtype == np.float64:
                    cN, sN = cN.astype(np.float64), sN.astype(np.float64)
                c, s = Tensor(cN), Tensor(sN)
                return Tensor.concatenate([t1 * c - t2 * s, t1 * s + t2 * c], axis=-1)
            q = _rope_naive(q)
            k = _rope_naive(k)
            att = q.matmul(k.transpose(0, 1, 3, 2)) * (1.0 / np.sqrt(self.head_dim))
            att = att + Tensor(mask.astype(np.float32))
            att = att.softmax(axis=-1)
            y = att.matmul(v)
        # (B,H,T,hd) -> (B,T,H*hd): гибридный reshape+transpose одним узлом.
        y = ops.merge_heads(y, B, T, C)
        return self.proj(y)


class SwiGLU:
    def __init__(self, dim: int, hidden: int, fast: bool = True) -> None:
        self.fc = Linear(dim, 2 * hidden)
        self.proj = Linear(hidden, dim)
        self.fast = fast

    def params(self) -> Dict[str, Tensor]:
        p = {}
        for k, v in self.fc.params().items():
            p["fc." + k] = v
        for k, v in self.proj.params().items():
            p["proj." + k] = v
        return p

    def __call__(self, x: Tensor) -> Tensor:
        h = self.fc(x)  # (B,T,2*hidden)
        if self.fast and h.data.dtype == np.float32:
            return self.proj(ops.swiglu_gate(h))
        hidden = h.shape[-1] // 2
        a = h[..., :hidden]
        b = h[..., hidden:]
        return self.proj(a.silu() * b)


class TransformerBlock:
    def __init__(self, dim: int, n_heads: int, ffn_mult: float = 2.667,
                 fast: bool = True) -> None:
        self.ln1 = RMSNorm(dim, fast=fast)
        self.attn = CausalSelfAttention(dim, n_heads)
        self.ln2 = RMSNorm(dim, fast=fast)
        hidden = int(dim * ffn_mult)
        hidden += hidden % 2  # чётное для SwiGLU-сплита
        self.ffn = SwiGLU(dim, hidden, fast=fast)

    def params(self) -> Dict[str, Tensor]:
        p = {}
        for k, v in self.ln1.params().items():
            p["ln1." + k] = v
        for k, v in self.attn.params().items():
            p["attn." + k] = v
        for k, v in self.ln2.params().items():
            p["ln2." + k] = v
        for k, v in self.ffn.params().items():
            p["ffn." + k] = v
        return p

    def __call__(self, x: Tensor, cos: np.ndarray, sin: np.ndarray,
                 mask: np.ndarray) -> Tensor:
        x = x + self.attn(self.ln1(x), cos, sin, mask)
        x = x + self.ffn(self.ln2(x))
        return x


class TinyGPT:
    """Языковая модель. Умеет строиться из конфига и считать лосс."""

    def __init__(self, vocab_size: int, dim: int = 96, n_layers: int = 3,
                 n_heads: int = 3, block_size: int = 64, ffn_mult: float = 2.667,
                 fast: bool = True) -> None:
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.block_size = block_size
        self.ffn_mult = ffn_mult

        self.embed = Embedding(vocab_size, dim)
        self.blocks = [TransformerBlock(dim, n_heads, ffn_mult, fast=fast)
                       for _ in range(n_layers)]
        self.ln_f = RMSNorm(dim, fast=fast)
        # каузальная маска + tie: выходной слой = E^T (см. linear_tied)
        mask = np.triu(np.full((block_size, block_size), -1e9, dtype=np.float32), k=1)
        self.registered_mask = mask
        self._itos_cache: Dict[int, str] = {}

        self._rope: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    # ------------------------------------------------------------- параметры
    def params(self) -> Dict[str, Tensor]:
        p: Dict[str, Tensor] = {}
        for k, v in self.embed.params().items():
            p["embed." + k] = v
        for i, blk in enumerate(self.blocks):
            for k, v in blk.params().items():
                p[f"block{i}." + k] = v
        for k, v in self.ln_f.params().items():
            p["lnf." + k] = v
        return p

    def named_arrays(self) -> Dict[str, np.ndarray]:
        return {k: v.data for k, v in self.params().items()}

    def num_params(self) -> int:
        # Голова привязана к эмбеддингу, отдельного массива нет — считаем как есть.
        return int(sum(v.size for v in self.named_arrays().values()))

    # ------------------------------------------------------------- forward
    def _rope_for(self, T: int) -> Tuple[np.ndarray, np.ndarray]:
        r = self._rope.get(T)
        if r is None:
            r = rope_cache(T, self.head_dim())
            self._rope[T] = r
        return r

    def head_dim(self) -> int:
        return self.dim // self.n_heads

    def forward(self, idx: np.ndarray, targets: Optional[np.ndarray] = None):
        """idx (B,T) int. Возвращает (logits (B,T,V), loss или None)."""
        prev = Tensor.no_graph
        Tensor.no_graph = targets is None
        try:
            B, T = idx.shape
            if T > self.block_size:
                raise ValueError(
                    f"Длина {T} больше block_size={self.block_size}")
            cos, sin = self._rope_for(T)
            mask = self.registered_mask[:T, :T]
            x = self.embed(idx)
            for blk in self.blocks:
                x = blk(x, cos, sin, mask)
            x = self.ln_f(x)
            logits = x.linear_tied(self.embed.E)  # tie-голова: градиент идёт в E
            if targets is not None:
                if logits.data.dtype == np.float32:
                    loss = ops.fused_cross_entropy(logits, targets)
                else:
                    from core.tensor import cross_entropy
                    loss = cross_entropy(logits, targets)
            else:
                loss = None
            return logits, loss
        finally:
            Tensor.no_graph = prev

    # ------------------------------------------------------------- генерация
    @staticmethod
    def infer_mode():
        """Совместимость: использовать Tensor.infer_mode()."""
        return Tensor.infer_mode()

    def _is_newline(self, token_id: int) -> bool:
        """Определяет токен-перенос строки (для символьного токенизатора — сам '\\n')."""
        s = self._itos_cache.get(token_id)
        return s == "\n"

    def set_tokenizer_vocab(self, itos: List[str]) -> None:
        """Кэш словаря для остановки по '\\n' (необязателен для работы модели)."""
        self._itos_cache = {i: s for i, s in enumerate(itos)}

    def generate(self, idx: np.ndarray, max_new_tokens: int, temperature: float = 0.8,
                 top_k: Optional[int] = 0, top_p: float = 1.0,
                 repetition_penalty: float = 1.0,
                 stop_at_eos: Optional[int] = None, rng=None,
                 stop_at_newline: bool = False) -> List[List[int]]:
        """Авторегрессионная генерация. Возвращает список последовательностей."""
        import random as _random
        rng = rng or _random.Random()
        prev = Tensor.no_graph
        Tensor.no_graph = True
        try:
            out = [list(seq) for seq in idx]
            finished = [False] * len(out)
            for _ in range(max_new_tokens):
                # берем последние block_size токенов
                window = np.array([seq[-self.block_size:] for seq in out], dtype=np.int32)
                logits, _ = self.forward(window, targets=None)
                nexts: List[int] = []
                for b in range(len(out)):
                    if finished[b]:
                        nexts.append(self.registered_pad())
                        continue
                    row = logits.data[b, -1].astype(np.float64)
                    if repetition_penalty != 1.0:
                        seen = set(out[b][-64:])
                        for t in seen:
                            row[t] /= repetition_penalty if row[t] > 0 else 1.0 / repetition_penalty
                    if temperature <= 1e-6:
                        nxt = int(np.argmax(row))
                    else:
                        row = row / max(temperature, 1e-6)
                        row -= row.max()
                        probs = np.exp(row)
                        if top_p < 1.0:
                            order = np.argsort(-probs)
                            csum = np.cumsum(probs[order])
                            cut = int(np.searchsorted(csum, top_p * probs.sum())) + 1
                            keep = order[:cut]
                            p2 = np.zeros_like(probs)
                            p2[keep] = probs[keep]
                            probs = p2 / p2.sum()
                        if top_k and top_k > 0:
                            kth = np.partition(probs, -top_k)[-top_k]
                            probs[probs < kth] = 0.0
                            probs = probs / probs.sum()
                        nxt = int(rng.choices(np.arange(len(probs)), weights=probs)[0])
                    out[b].append(nxt)
                    if stop_at_eos is not None and nxt == stop_at_eos:
                        finished[b] = True
                    if stop_at_newline and self._is_newline(nxt):
                        finished[b] = True
                    if len(out[b]) > 4000:
                        finished[b] = True
                if all(finished):
                    break
            return out
        finally:
            Tensor.no_graph = prev

    def registered_pad(self) -> int:
        return 0

    # ------------------------------------------------------------- io
    def config_dict(self) -> Dict:
        return {
            "vocab_size": self.vocab_size,
            "dim": self.dim,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "block_size": self.block_size,
            "ffn_mult": self.ffn_mult,
        }

    @classmethod
    def from_config(cls, cfg: Dict) -> "TinyGPT":
        return cls(
            vocab_size=int(cfg["vocab_size"]),
            dim=int(cfg.get("dim", 96)),
            n_layers=int(cfg.get("n_layers", 3)),
            n_heads=int(cfg.get("n_heads", 3)),
            block_size=int(cfg.get("block_size", 64)),
            ffn_mult=float(cfg.get("ffn_mult", 2.667)),
        )

    def load_arrays(self, arrays: Dict[str, np.ndarray]) -> None:
        cur = self.params()
        for k, v in cur.items():
            if k in arrays:
                v.data[...] = arrays[k].reshape(v.data.shape).astype(v.data.dtype)
