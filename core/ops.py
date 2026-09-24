# -*- coding: utf-8 -*-
"""Плавленые (fused) numpy-операции с ручным backward.

Зачем: наивный граф из десятков мелких Tensor-операций тратит большую часть
времени на создание объектов и лишние копии массивов. Здесь каждый «большой»
кусок модели (RoPE, RMSNorm, SwiGLU, каузальное внимание, кросс-энтропия)
считается одной-двумя numpy-функциями, а градиент — аналитически за один
проход. Обучение ускоряется в несколько раз без изменения результатов.

Обратный проход совместим с core.tensor.Tensor.backward: Tensor.backward
присваивает node.grad = numpy-массив и затем вызывает node._backward(), где
node — выходной тензор узла; мы читаем градиент из node.grad.
"""
from __future__ import annotations

import numpy as np
from typing import List

from core.tensor import Tensor


def _fuse(out_data: np.ndarray, inputs: List[Tensor],
          bwd_factory) -> Tensor:
    """Выходной тензор графа; backward создаётся фабрикой bwd_factory(out).

    Возвращает None-фабрику, если ни один вход не требует градиента.
    """
    needs = [t.requires_grad for t in inputs]
    req = any(needs)
    out = Tensor(out_data, requires_grad=req,
                 _children=tuple(t for t, n in zip(inputs, needs) if n))
    if req:
        out._backward = bwd_factory(out)
    return out


# ---------------------------------------------------------------- RoPE

def rope(x: Tensor, cos: np.ndarray, sin: np.ndarray) -> Tensor:
    """Ротация половин размерностей головы. x: (B, T, H, D), cos/sin: (T, D/2).

    Один fused-узел вместо ~12 мелких. Эквивалент apply_rope из core.model.
    """
    xd = x.data
    half = xd.shape[-1] // 2
    x1, x2 = xd[..., :half], xd[..., half:]
    # x теперь (B, H, T, hd): позиционная ось T стоит на месте индекса 2.
    c = cos[None, None].astype(xd.dtype, copy=False)   # (1, 1, T, half)
    s = sin[None, None].astype(xd.dtype, copy=False)
    out = np.concatenate((x1 * c - x2 * s, x1 * s + x2 * c), axis=-1)

    def make_bwd(out_t: Tensor):
        def bwd() -> None:
            g = out_t.grad
            g1, g2 = g[..., :half], g[..., half:]
            gx = np.empty_like(g)
            gx[..., :half] = g1 * c + g2 * s
            gx[..., half:] = -g1 * s + g2 * c
            x._acc(gx)
        return bwd
    return _fuse(out, [x], make_bwd)


# ---------------------------------------------------------------- RMSNorm

def rmsnorm(x: Tensor, g: Tensor) -> Tensor:
    """RMSNorm по последней оси одним узлом: x * rsqrt(mean(x^2)+eps) * g."""
    xd = x.data
    ms = np.mean(xd * xd, axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(ms + 1e-6)
    xhat = xd * inv
    out = xhat * g.data
    N = xd.shape[-1]

    def make_bwd(out_t: Tensor):
        def bwd() -> None:
            gg = out_t.grad
            if x.requires_grad:
                gx_hat = gg * g.data
                dot = np.sum(gx_hat * xhat, axis=-1, keepdims=True)
                x._acc(inv * (gx_hat - xhat * dot / N))
            if g.requires_grad:
                gg2 = gg.reshape(-1, gg.shape[-1])
                xh2 = xhat.reshape(-1, xhat.shape[-1])
                g._acc(np.sum(gg2 * xh2, axis=0))
        return bwd
    return _fuse(out, [x, g], make_bwd)


# ---------------------------------------------------------------- формы

def transpose_op(x: Tensor, axes: tuple) -> Tensor:
    """Перестановка осей одним узлом (аналог Tensor.transpose)."""
    inv = tuple(np.argsort(axes))
    out = np.transpose(x.data, axes)

    def make_bwd(out_t: Tensor):
        def bwd() -> None:
            x._acc(np.transpose(out_t.grad, inv))
        return bwd
    return _fuse(out, [x], make_bwd)


def merge_heads(y: Tensor, B: int, T: int, C: int) -> Tensor:
    """(B, H, T, hd) -> (B, T, C) одним узлом вместо transpose+reshape.

    Backward: обратная раскладка головы без воссоздания промежуточных форм.
    """
    out = np.transpose(y.data, (0, 2, 1, 3)).reshape(B, T, C)

    def make_bwd(out_t: Tensor):
        def bwd() -> None:
            g = out_t.grad.reshape(B, T, -1)
            y._acc(np.transpose(g.reshape(B, T, -1, y.data.shape[-1]), (0, 2, 1, 3)))
        return bwd
    return _fuse(out, [y], make_bwd)


# ---------------------------------------------------------------- SwiGLU-сплит

def swiglu_gate(h: Tensor) -> Tensor:
    """h: (..., 2*hidden) -> silu(h[..., :hidden]) * h[..., hidden:].

    Один узел вместо срезов+exp+умножений; градиент течёт прямо в h.
    """
    hd = h.data
    hidden = hd.shape[-1] // 2
    a, b = hd[..., :hidden], hd[..., hidden:]
    sb = 1.0 / (1.0 + np.exp(-b))
    out = a * sb * b

    def make_bwd(out_t: Tensor):
        def bwd() -> None:
            gg = out_t.grad
            gh = np.empty_like(hd)
            gh[..., :hidden] = gg * sb
            gh[..., hidden:] = gg * a * sb * (1.0 + b * (1.0 - sb))
            h._acc(gh)
        return bwd
    return _fuse(out, [h], make_bwd)


# ---------------------------------------------------------------- attention

def causal_attention(q: Tensor, k: Tensor, v: Tensor,
                     mask: np.ndarray) -> Tensor:
    """Каузальное внимание (B, H, T, D) -> (B, H, T, D) одним узлом.

    Внутри: масштаб, маска, softmax, произведение — и весь backward сразу.
    """
    qd, kd, vd = q.data, k.data, v.data
    hd = qd.shape[-1]
    scale = 1.0 / np.sqrt(hd)
    att = np.matmul(qd, kd.transpose(0, 1, 3, 2))
    att *= scale
    att += mask
    att -= att.max(axis=-1, keepdims=True)
    ez = np.exp(att)
    probs = ez * (1.0 / ez.sum(axis=-1, keepdims=True))
    out = np.matmul(probs, vd)

    def make_bwd(out_t: Tensor):
        def bwd() -> None:
            g = out_t.grad
            if q.requires_grad or k.requires_grad:
                dp = np.matmul(g, vd.transpose(0, 1, 3, 2))
                dot = np.sum(dp * probs, axis=-1, keepdims=True)
                datt = probs * (dp - dot)
                datt *= scale
                if q.requires_grad:
                    q._acc(np.matmul(datt, kd))
                if k.requires_grad:
                    k._acc(np.matmul(datt.transpose(0, 1, 3, 2), qd))
            if v.requires_grad:
                v._acc(np.matmul(probs.transpose(0, 1, 3, 2), g))
        return bwd
    return _fuse(out, [q, k, v], make_bwd)


# ---------------------------------------------------------------- cross-entropy

def fused_cross_entropy(logits: Tensor, targets: np.ndarray) -> Tensor:
    """Кросс-энтропия: градиент вероятностей готовится заранее, backward — копия.

    Замена cross_entropy из core.tensor (результат идентичен).
    """
    x = logits.data
    B, T, V = x.shape
    tgt = targets.reshape(-1)
    flat = x.reshape(B * T, V)
    m = flat.max(axis=1, keepdims=True)
    z = flat - m
    ez = np.exp(z)
    sumz = ez.sum(axis=1, keepdims=True)
    logZ = np.log(sumz) + m
    pick = flat[np.arange(B * T), tgt].reshape(B * T, 1)
    loss_val = float((logZ - pick).mean())
    out = Tensor(loss_val, requires_grad=logits.requires_grad)

    if logits.requires_grad:
        rows = np.arange(B * T)
        probs = ez / sumz
        probs[rows, tgt] -= 1.0
        probs *= 1.0 / (B * T)
        grads = probs.reshape(x.shape).astype(logits.data.dtype, copy=False)

        def bwd() -> None:
            logits._acc(grads)
        out._backward = bwd
        out._prev = (logits,)
    return out
