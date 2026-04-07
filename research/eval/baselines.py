"""Baselines for matmul tensor rank <2,4,5>."""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from research.eval.evaluator import (  # noqa: E402
    BENCHMARK,
    N,
    M,
    P,
    build_target_tensor,
    naive_r40_decomposition,
    verify_tensor_decomposition,
)


def run_naive_baseline():
    """Return the explicit rank-40 decomposition."""
    U, V, W = naive_r40_decomposition()
    return U, V, W, 40, 0.0


def run_gradient_baseline(rank: int = 35, num_restarts: int = 3, steps: int = 5000, lr: float = 0.05, seed: int = 0):
    """Simple numpy-based gradient descent CP decomposition with random restarts."""
    T = build_target_tensor()
    d1, d2, d3 = T.shape
    rng = np.random.default_rng(seed)
    best = None
    for restart in range(num_restarts):
        U = rng.standard_normal((d1, rank)).astype(np.float32) * 0.1
        V = rng.standard_normal((d2, rank)).astype(np.float32) * 0.1
        W = rng.standard_normal((d3, rank)).astype(np.float32) * 0.1
        for step in range(steps):
            R = np.einsum("ir,jr,kr->ijk", U, V, W) - T
            gU = np.einsum("ijk,jr,kr->ir", R, V, W)
            gV = np.einsum("ijk,ir,kr->jr", R, U, W)
            gW = np.einsum("ijk,ir,jr->kr", R, U, V)
            U -= lr * gU
            V -= lr * gV
            W -= lr * gW
            if step % 1000 == 0:
                loss = float(np.sum(R * R))
                if not np.isfinite(loss):
                    break
        loss = float(np.sum((np.einsum("ir,jr,kr->ijk", U, V, W) - T) ** 2))
        if best is None or loss < best[3]:
            best = (U.copy(), V.copy(), W.copy(), loss)
    U, V, W, loss = best
    return U, V, W, rank, loss


def _report(name, U, V, W, claimed_rank, loss):
    is_valid, rank, max_err = verify_tensor_decomposition(U, V, W)
    score = BENCHMARK / rank if is_valid else 0.0
    print(f"[{name}]")
    print(f"  rank={rank} valid={is_valid} max_err={max_err:.3e} loss={loss:.3e} score={score:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Baselines for matmul <2,4,5>.")
    parser.add_argument("--run", action="store_true", help="Run all baselines.")
    parser.add_argument("--run-naive", action="store_true", help="Run only the naive R=40 baseline.")
    parser.add_argument("--run-gradient", action="store_true", help="Run only the gradient baseline.")
    parser.add_argument("--rank", type=int, default=35)
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--steps", type=int, default=5000)
    args = parser.parse_args()

    if not (args.run or args.run_naive or args.run_gradient):
        parser.print_help()
        return

    if args.run or args.run_naive:
        U, V, W, r, loss = run_naive_baseline()
        _report("naive_r40", U, V, W, r, loss)

    if args.run or args.run_gradient:
        U, V, W, r, loss = run_gradient_baseline(
            rank=args.rank, num_restarts=args.restarts, steps=args.steps
        )
        _report(f"gradient_r{args.rank}", U, V, W, r, loss)


if __name__ == "__main__":
    main()
