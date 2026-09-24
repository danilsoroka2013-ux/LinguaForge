# -*- coding: utf-8 -*-
"""Мини-тензор с автоградиентом.

Настолько маленький, насколько возможно, и настолько быстрый, насколько нужно
для моделей на 0.3–10 млн параметров. Только numpy. Поддерживает float32
(работа) и float64 (проверка градиентов).
"""
from __future__ import annotations

import numpy as np
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple


def _as_array(data: Any) -> np.ndarray:
    a = np.asarray(data)
    if a.dtype == np.float64:
        return a  # сохраняем двойную точность для градиентной проверки
    if a.dtype.kind == "f":
        return a.astype(np.float32)
    return a.astype(np.float32)


def _unbroadcast(g: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """Сводит градиент g к форме shape (после broadcasting)."""
    if g.shape == shape:
        return g
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and g.shape[i] != 1:
            g = g.sum(axis=i, keepdims=True)
    return g


class Tensor:
    """Тензор с автоматическим дифференцированием."""

    no_graph = False  # глобальный рубильник: True — граф не строится (инференс)

    __slots__ = ("data", "grad", "requires_grad", "_backward", "_prev")

    def __init__(self, data: Any, requires_grad: bool = False,
                 _children: Tuple["Tensor", ...] = ()): 
        self.data: np.ndarray = _as_array(data)
        self.grad: Optional[np.ndarray] = None
        self.requires_grad = bool(requires_grad)
        self._backward: Callable[[], None] = lambda: None
        self._prev: Tuple[Tensor, ...] = _children if self.requires_grad else ()

    # --------------------------------------------------- контекст инференса
    # Инференс без построения графа: ощутимо быстрее и без мусора в памяти.
    # Использование:  with infer_mode():  logits, _ = model.forward(x)

    class _InferCtx:
        def __enter__(self):
            self.prev = Tensor.no_graph
            Tensor.no_graph = True
            return None

        def __exit__(self, *exc):
            Tensor.no_graph = self.prev
            return False

    @staticmethod
    def infer_mode() -> "Tensor._InferCtx":
        return Tensor._InferCtx()

    # ----------------------------------------------------------- утилиты
    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    def item(self) -> float:
        return float(self.data)

    def zero_grad(self) -> None:
        self.grad = None

    def _acc(self, g: np.ndarray) -> None:
        if self.grad is None:
            # Кэшируем на тензор, чтобы _acc переиспользовал буфер без аллокаций.
            self.grad = g
        else:
            # In-place: без лишних копирований больших градиентов.
            self.grad += g

    def _make(self, data: Any, children: Sequence["Tensor"]) -> "Tensor":
        return Tensor(data, requires_grad=True, _children=tuple(children))

    def _hook(self, out: "Tensor", bwd: Callable[[], None]) -> "Tensor":
        out._backward = bwd
        return out

    # ----------------------------------------------------------- арифметика
    def __add__(self, other: Any) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = self._make(self.data + other.data, (self, other))

        def bwd():
            g = out.grad
            if self.requires_grad:
                self._acc(_unbroadcast(g, self.data.shape))
            if other.requires_grad:
                other._acc(_unbroadcast(g, other.data.shape))
        return self._hook(out, bwd)

    def __radd__(self, other: Any) -> "Tensor":
        return self.__add__(other)

    def __neg__(self) -> "Tensor":
        out = self._make(-self.data, (self,))

        def bwd():
            if self.requires_grad:
                self._acc(-out.grad)
        return self._hook(out, bwd)

    def __sub__(self, other: Any) -> "Tensor":
        return self.__add__(-other if isinstance(other, Tensor) else Tensor(other).__neg__())

    def __rsub__(self, other: Any) -> "Tensor":
        return Tensor(other).__add__(self.__neg__())

    def __mul__(self, other: Any) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = self._make(self.data * other.data, (self, other))

        def bwd():
            g = out.grad
            if self.requires_grad:
                self._acc(_unbroadcast(g * other.data, self.data.shape))
            if other.requires_grad:
                other._acc(_unbroadcast(g * self.data, other.data.shape))
        return self._hook(out, bwd)

    def __rmul__(self, other: Any) -> "Tensor":
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self.__mul__(other.__pow__(-1.0))

    def __pow__(self, p: float) -> "Tensor":
        out = self._make(self.data ** p, (self,))

        def bwd():
            if self.requires_grad:
                self._acc(out.grad * (p * self.data ** (p - 1)))
        return self._hook(out, bwd)

    def matmul(self, other: "Tensor") -> "Tensor":
        """Матричное умножение по последним осям (2D и 3D с батчами).

        Быстрый случай «3D @ 2D-вес» (Linear) имеет отдельный backward:
        dW накапливается через (вход^T @ grad) одной GEMM без воссоздания
        промежуточных представлений.
        """
        fast_lin = (other.data.ndim == 2 and self.data.ndim == 3
                    and self.data.dtype == np.float32)
        out = self._make(np.matmul(self.data, other.data), (self, other))
        if fast_lin:
            x = self.data
            W = other.data
            B, T, C = x.shape
            M = B * T
            x2 = x.reshape(M, C)

            def bwd():
                g = out.grad
                g2 = np.ascontiguousarray(g).reshape(M, W.shape[1])
                if self.requires_grad:
                    self._acc(np.matmul(g2, W.T).reshape(x.shape))
                if other.requires_grad:
                    # dW = x^T @ g  (GEMM: (C,M)x(M,N) без копий x)
                    other._acc(np.matmul(x2.T, g2))
        else:
            def bwd():
                g = out.grad
                if self.requires_grad:
                    if other.data.ndim == 2 and self.data.ndim >= 2:
                        g2 = np.ascontiguousarray(g).reshape(-1, g.shape[-1])
                        self._acc(np.matmul(g2, other.data.T).reshape(self.data.shape))
                    else:
                        self._acc(np.matmul(g, np.swapaxes(other.data, -1, -2)))
                if other.requires_grad:
                    if other.data.ndim == 2 and self.data.ndim >= 2:
                        x2 = np.ascontiguousarray(self.data).reshape(
                            -1, self.data.shape[-1])
                        g2 = np.ascontiguousarray(g).reshape(-1, g.shape[-1])
                        other._acc(np.matmul(x2.T, g2))
                    else:
                        ga = np.matmul(np.swapaxes(self.data, -1, -2), g)
                        while ga.ndim > other.data.ndim:
                            ga = ga.sum(axis=0)
                        other._acc(ga)
        return self._hook(out, bwd)

    def __matmul__(self, other: "Tensor") -> "Tensor":
        return self.matmul(other)

    # ----------------------------------------------------------- формы
    def reshape(self, *shape: int) -> "Tensor":
        shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
        out = self._make(self.data.reshape(shape), (self,))

        def bwd():
            if self.requires_grad:
                self._acc(out.grad.reshape(self.data.shape))
        return self._hook(out, bwd)

    def transpose(self, *axes: int) -> "Tensor":
        axes = tuple(axes)
        out = self._make(np.transpose(self.data, axes), (self,))
        inv = tuple(np.argsort(axes))

        def bwd():
            if self.requires_grad:
                self._acc(np.transpose(out.grad, inv))
        return self._hook(out, bwd)

    def sum(self, axis: Optional[int] = None, keepdims: bool = False) -> "Tensor":
        out = self._make(self.data.sum(axis=axis, keepdims=keepdims), (self,))

        def bwd():
            if not self.requires_grad:
                return
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self._acc(np.broadcast_to(g, self.data.shape).copy())
        return self._hook(out, bwd)

    def mean(self, axis: Optional[int] = None, keepdims: bool = False) -> "Tensor":
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    def __getitem__(self, idx: Any) -> "Tensor":
        out = self._make(self.data[idx], (self,))

        def bwd():
            if not self.requires_grad:
                return
            g = out.grad
            gx = np.zeros_like(self.data)
            advanced = isinstance(idx, np.ndarray) or (
                isinstance(idx, tuple) and any(isinstance(i, np.ndarray) for i in idx))
            if advanced:
                np.add.at(gx, idx, g)
            else:
                gx[idx] = gx[idx] + g
            self._acc(gx)
        return self._hook(out, bwd)

    @staticmethod
    def concatenate(tensors: Sequence["Tensor"], axis: int = 0) -> "Tensor":
        """Конкатенация тензоров вдоль оси с сохранением графа."""
        datas = [t.data for t in tensors]
        any_req = any(t.requires_grad for t in tensors)
        out = Tensor(np.concatenate(datas, axis=axis),
                     requires_grad=any_req, _children=tuple(tensors))

        if any_req:
            def bwd():
                g = out.grad
                start = 0
                for t in tensors:
                    if t.requires_grad:
                        sl = [slice(None)] * g.ndim
                        sl[axis] = slice(start, start + t.data.shape[axis])
                        t._acc(g[tuple(sl)])
                    start += t.data.shape[axis]
            out._backward = bwd
        return out

    def linear_tied(self, E: "Tensor") -> "Tensor":
        """y = x @ E.T — выходной слой, привязанный к эмбеддингу E.

        Градиент честно возвращается в E: dE += g^T @ x.
        """
        x = self.data
        out = self._make(np.matmul(x, E.data.T), (self, E))

        def bwd():
            g = out.grad
            if self.requires_grad:
                self._acc(np.matmul(g, E.data))
            if E.requires_grad:
                g2 = np.ascontiguousarray(g).reshape(-1, g.shape[-1])
                x2 = x.reshape(-1, x.shape[-1])
                E._acc(np.matmul(g2.T, x2).astype(E.data.dtype))
        return self._hook(out, bwd)

    def index_select(self, idx: np.ndarray) -> "Tensor":
        """Выборка по целочисленному индексу вдоль оси 0 (эмбеддинги).

        Быстрый путь: если каждый токен батча уникален (типичный случай при
        B*T < словаря), scatter делается одной матричной операцией вместо
        медленного np.add.at.
        """
        flat_idx = idx.reshape(-1)
        out = self._make(self.data[idx], (self,))

        def bwd():
            if not self.requires_grad:
                return
            g = out.grad
            if flat_idx.size and np.unique(flat_idx).size == flat_idx.size:
                gx = np.zeros_like(self.data)
                gx[flat_idx] = g.reshape(flat_idx.size, -1)
                self._acc(gx)
            else:
                gx = np.zeros_like(self.data)
                np.add.at(gx, idx, g)
                self._acc(gx)
        return self._hook(out, bwd)

    # ----------------------------------------------------------- функции
    def exp(self) -> "Tensor":
        out = self._make(np.exp(self.data), (self,))

        def bwd():
            if self.requires_grad:
                self._acc(out.grad * out.data)
        return self._hook(out, bwd)

    def log(self) -> "Tensor":
        out = self._make(np.log(self.data), (self,))

        def bwd():
            if self.requires_grad:
                self._acc(out.grad / self.data)
        return self._hook(out, bwd)

    def sqrt(self) -> "Tensor":
        out = self._make(np.sqrt(self.data), (self,))

        def bwd():
            if self.requires_grad:
                self._acc(out.grad * 0.5 / out.data)
        return self._hook(out, bwd)

    def rsqrt(self) -> "Tensor":
        out = self._make(1.0 / np.sqrt(self.data), (self,))

        def bwd():
            if self.requires_grad:
                self._acc(out.grad * (-0.5) * out.data ** 3)
        return self._hook(out, bwd)

    def tanh(self) -> "Tensor":
        out = self._make(np.tanh(self.data), (self,))

        def bwd():
            if self.requires_grad:
                self._acc(out.grad * (1.0 - out.data ** 2))
        return self._hook(out, bwd)

    def sigmoid(self) -> "Tensor":
        x = self.data
        s = 1.0 / (1.0 + np.exp(-x))
        out = self._make(s, (self,))

        def bwd():
            if self.requires_grad:
                self._acc(out.grad * s * (1.0 - s))
        return self._hook(out, bwd)

    def silu(self) -> "Tensor":
        x = self.data
        s = 1.0 / (1.0 + np.exp(-x))
        y = x * s
        out = self._make(y, (self,))

        def bwd():
            if self.requires_grad:
                g = out.grad
                sp = s * (1.0 + x * (1.0 - s))
                np.multiply(g, sp, out=sp)  # переиспользуем буфер sp
                self._acc(sp)
        return self._hook(out, bwd)

    def softmax(self, axis: int = -1) -> "Tensor":
        x = self.data
        z = x - x.max(axis=axis, keepdims=True)
        e = np.exp(z)
        s = e / e.sum(axis=axis, keepdims=True)
        out = self._make(s, (self,))

        def bwd():
            if self.requires_grad:
                g = out.grad
                dot = (g * s).sum(axis=axis, keepdims=True)
                r = s * (g - dot)
                self._acc(r)
        return self._hook(out, bwd)

    def dropout(self, p: float, training: bool) -> "Tensor":
        if not training or p <= 0.0:
            return self
        mask = (np.random.rand(*self.data.shape) >= p).astype(self.data.dtype) / (1.0 - p)
        out = self._make(self.data * mask, (self,))

        def bwd():
            if self.requires_grad:
                self._acc(out.grad * mask)
        return self._hook(out, bwd)

    # ----------------------------------------------------------- backward
    def backward(self) -> None:
        """Обратное распространение от этого тензора (обычно от лосса)."""
        topo: List[Tensor] = []
        visited = set()

        def build(v: Tensor) -> None:
            if id(v) not in visited:
                visited.add(id(v))
                for c in v._prev:
                    build(c)
                topo.append(v)

        build(self)
        # ВАЖНО: корневому тензору присваиваем numpy-массив в .grad (не Tensor) —
        # fused-бэкенды читают node.grad напрямую как ndarray.
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()
        # Разрываем граф: backward-замыкания держат ссылки на активации, и без
        # этого циклы ссылок копятся между шагами (утечка памяти на длинных
        # прогонах). Градиенты параметров (листьев) остаются нетронутыми.
        for node in topo:
            if node._prev:
                node._prev = ()
                node._backward = None

    def detach_data(self) -> np.ndarray:
        return self.data


def cross_entropy(logits: Tensor, targets: np.ndarray) -> Tensor:
    """Средняя кросс-энтропия. logits (B,T,V) float, targets (B,T) int."""
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
        probs = ez / sumz
        d = probs
        d[np.arange(B * T), tgt] -= 1.0
        d /= (B * T)

        def bwd():
            logits._acc(d.reshape(x.shape).astype(logits.data.dtype))
        out._backward = bwd
        out._prev = (logits,)
    return out
