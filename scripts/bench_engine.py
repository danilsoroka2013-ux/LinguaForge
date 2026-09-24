# -*- coding: utf-8 -*-
"""Бенчмарк скорости обучения (токенов/с) для выбранного пути вычислений."""
import argparse
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from core.config import set_seed, clip_grad_norm          # noqa: E402
from core.model import TinyGPT                            # noqa: E402
from core.data import make_batches                        # noqa: E402
from core.optimizer import AdamW                          # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", type=int, default=1)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--dim", type=int, default=96)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=3)
    ap.add_argument("--block", type=int, default=96)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--vocab", type=int, default=120)
    args = ap.parse_args()

    set_seed(0)
    V, D, L, H, T, B = (args.vocab, args.dim, args.layers, args.heads,
                        args.block, args.batch)
    m = TinyGPT(vocab_size=V, dim=D, n_layers=L, n_heads=H, block_size=T,
                fast=bool(args.fast))
    ids = np.random.randint(0, V, size=4000).astype(np.int32)
    batch = make_batches(ids, B, T, 1, seed=0)
    opt = AdamW(m.params(), lr=3e-3, weight_decay=0.05)
    params = m.params()

    def one_step() -> float:
        x, y = batch()
        _, loss = m.forward(x, y)
        opt.zero_grad()
        loss.backward()
        clip_grad_norm({k: v.grad for k, v in params.items()}, 1.0)
        opt.step(3e-3)
        return float(loss.data)

    for _ in range(3):
        one_step()
    t0 = time.time()
    for _ in range(args.steps):
        one_step()
    dt = time.time() - t0
    tok_s = args.steps * B * T / dt
    print(f"{tok_s:,.0f}")


if __name__ == "__main__":
    main()
