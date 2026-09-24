# -*- coding: utf-8 -*-
"""AdamW на numpy — аллокации вынесены из цикла, обновление in-place."""
from __future__ import annotations

import numpy as np
from typing import Dict

from core.tensor import Tensor


class AdamW:
    def __init__(self, params: Dict[str, Tensor], lr: float = 3e-3,
                 betas=(0.9, 0.95), eps: float = 1e-8, weight_decay: float = 0.05):
        self.params = list(params.values())
        self.lr = lr
        self.b0, self.b1 = betas
        self.eps = eps
        self.wd = weight_decay
        self.t = 0
        self.m = {id(p): np.zeros_like(p.data) for p in self.params}
        self.v = {id(p): np.zeros_like(p.data) for p in self.params}
        # Переиспользуемые буферы: на каждый шаг не выделяем 4 больших массива.
        self._g = {id(p): np.empty_like(p.data) for p in self.params}

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    def step(self, lr: float | None = None) -> None:
        self.t += 1
        lr = self.lr if lr is None else lr
        b0, b1 = self.b0, self.b1
        bc1 = 1.0 - b0 ** self.t
        bc2 = 1.0 - b1 ** self.t
        inv_bc1 = 1.0 / bc1
        inv_bc2 = 1.0 / bc2
        for p in self.params:
            if p.grad is None:
                continue
            pid = id(p)
            buf = self._g[pid]
            # buf = grad (приведение типов без лишней аллокации)
            np.copyto(buf, p.grad, casting="unsafe")
            g = buf
            m = self.m[pid]
            v = self.v[pid]
            # m = b0*m + (1-b0)*g
            m *= b0
            m += (1.0 - b0) * g
            # v = b1*v + (1-b1)*g^2
            v *= b1
            v += (1.0 - b1) * (g * g)
            # upd = lr * (m/bc1) / (sqrt(v/bc2) + eps)
            denom = np.sqrt(v * inv_bc2)
            denom += self.eps
            denom **= -1
            upd = m * inv_bc1 * denom
            upd *= lr
            if self.wd > 0:
                upd += lr * self.wd * p.data
            p.data -= upd
