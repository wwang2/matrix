"""Evaluator for matrix multiplication tensor rank <2,4,5> research campaign.

Adapted from openevolve/examples/alphaevolve_math_problems/matmul/evaluator.py.
"""
import argparse
import os
import sys
import time
from importlib import __import__

import numpy as np

BENCHMARK = 32
N, M, P = 2, 4, 5
TENSOR_SHAPE = (N * M, M * P, N * P)  # (8, 20, 10)


def build_target_tensor(n: int = N, m: int = M, p: int = P) -> np.ndarray:
    """Construct the matmul tensor <n, m, p>."""
    T = np.zeros((n * m, m * p, n * p), dtype=np.float32)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    return T


def verify_tensor_decomposition(U, V, W, n: int = N, m: int = M, p: int = P):
    """Verify a CP decomposition (U, V, W) of the matmul tensor <n, m, p>.

    Returns:
        (is_valid, rank, max_error)
    """
    # Reject complex inputs before casting (imaginary part would be silently dropped)
    if np.iscomplexobj(U) or np.iscomplexobj(V) or np.iscomplexobj(W):
        return False, -1, float("inf")

    try:
        U = np.asarray(U, dtype=np.float32)
        V = np.asarray(V, dtype=np.float32)
        W = np.asarray(W, dtype=np.float32)
    except Exception:
        return False, -1, float("inf")

    if U.ndim != 2 or V.ndim != 2 or W.ndim != 2:
        return False, -1, float("inf")

    if not (U.shape[1] == V.shape[1] == W.shape[1]):
        return False, -1, float("inf")

    rank = int(U.shape[1])

    if U.shape != (n * m, rank) or V.shape != (m * p, rank) or W.shape != (n * p, rank):
        return False, rank, float("inf")

    # Guard against NaN/Inf in factors (would produce NaN max_err, not inf)
    if not (np.all(np.isfinite(U)) and np.all(np.isfinite(V)) and np.all(np.isfinite(W))):
        return False, rank, float("inf")

    T = build_target_tensor(n, m, p)
    constructed = np.einsum("ir,jr,kr->ijk", U, V, W)
    max_err = float(np.max(np.abs(constructed - T)))
    is_valid = bool(np.array_equal(constructed, T))
    return is_valid, rank, max_err


def evaluate_solution(program_path: str) -> dict:
    """Evaluate a candidate program. The program must expose `run()` returning
    ((U, V, W), n, m, p, loss, rank).
    """
    try:
        abs_program_path = os.path.abspath(program_path)
        program_dir = os.path.dirname(abs_program_path)
        module_name = os.path.splitext(os.path.basename(program_path))[0]
        try:
            sys.path.insert(0, program_dir)
            if module_name in sys.modules:
                del sys.modules[module_name]
            program = __import__(module_name)
            start = time.time()
            decomposition, n, m, p, loss, rank = program.run()
            eval_time = time.time() - start
        finally:
            if program_dir in sys.path:
                sys.path.remove(program_dir)

        U, V, W = decomposition
        is_valid, verified_rank, max_err = verify_tensor_decomposition(U, V, W, n, m, p)
        if not is_valid:
            return {
                "combined_score": 0.0,
                "error": f"invalid decomposition (max_err={max_err:.3e})",
                "rank": verified_rank,
                "loss": float(loss),
            }

        return {
            "combined_score": float(BENCHMARK) / float(verified_rank),
            "loss": float(loss),
            "rank": int(verified_rank),
            "max_error": max_err,
            "eval_time": float(eval_time),
        }
    except Exception as e:
        return {"combined_score": 0.0, "error": str(e)}


def naive_r40_decomposition():
    """Build the explicit naive rank-40 decomposition of the <2,4,5> matmul tensor."""
    n, m, p = N, M, P
    R = n * m * p  # 40
    U = np.zeros((n * m, R), dtype=np.float32)
    V = np.zeros((m * p, R), dtype=np.float32)
    W = np.zeros((n * p, R), dtype=np.float32)
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + i, r] = 1.0
                r += 1
    return U, V, W


def _run_sanity_checks() -> int:
    results = []

    # 1) Trivial bad: all zeros, R=1
    U = np.zeros((8, 1), dtype=np.float32)
    V = np.zeros((20, 1), dtype=np.float32)
    W = np.zeros((10, 1), dtype=np.float32)
    is_valid, rank, err = verify_tensor_decomposition(U, V, W)
    ok = (not is_valid)
    results.append(("trivial_bad", ok, f"is_valid={is_valid} rank={rank} err={err:.2e}"))

    # 2) Naive R=40
    U, V, W = naive_r40_decomposition()
    is_valid, rank, err = verify_tensor_decomposition(U, V, W)
    ok = is_valid and rank == 40 and err == 0.0
    results.append(("naive_r40", ok, f"is_valid={is_valid} rank={rank} err={err:.2e}"))

    # 3) Known R=32 placeholder: just verify score formula
    score = BENCHMARK / 32
    ok = abs(score - 1.0) < 1e-12
    results.append(("known_r32_score", ok, f"score={score}"))

    # 4) Determinism — write a temp program and evaluate twice
    import tempfile
    prog = (
        "import numpy as np\n"
        "from research.eval.evaluator import naive_r40_decomposition\n"
        "def run():\n"
        "    U, V, W = naive_r40_decomposition()\n"
        "    return (U, V, W), 2, 4, 5, 0.0, 40\n"
    )
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "candidate_prog.py")
        with open(path, "w") as f:
            f.write(prog)
        # Ensure repo root in path so import works
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        r1 = evaluate_solution(path)
        r2 = evaluate_solution(path)
    ok = r1.get("combined_score") == r2.get("combined_score") and r1.get("rank") == r2.get("rank")
    results.append(("determinism", ok, f"r1={r1.get('combined_score')} r2={r2.get('combined_score')}"))

    print("Sanity check results:")
    all_ok = True
    for name, ok, detail in results:
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_ok = False
        print(f"  [{status}] {name}: {detail}")
    return 0 if all_ok else 1


def main():
    parser = argparse.ArgumentParser(description="Evaluator for matmul tensor rank <2,4,5>.")
    parser.add_argument("--sanity-check", action="store_true", help="Run built-in sanity checks.")
    parser.add_argument("--evaluate", type=str, default=None, help="Path to candidate program.py.")
    args = parser.parse_args()

    if args.sanity_check:
        sys.exit(_run_sanity_checks())
    if args.evaluate:
        print(evaluate_solution(args.evaluate))
        return
    parser.print_help()


if __name__ == "__main__":
    main()
