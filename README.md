# matrix — Minimal Tensor Rank R(2,4,5)

Find the smallest R such that the (2,4,5) matrix multiplication tensor admits an exact rank-R decomposition.

## Problem

The matrix multiplication tensor T ∈ ℝ^{8×20×10} encodes the bilinear map for computing a 2×4 times 4×5 product. Its rank is the minimum number of scalar multiplications in any bilinear algorithm. The naive algorithm uses 40; Strassen-style decompositions can do better. We want the minimum.

## Metric

| Metric | Direction | Naive | SOTA | Lower Bound |
|--------|-----------|-------|------|-------------|
| rank R | minimize | 40 | **32** (AlphaEvolve 2025) | 20 (exact) |

Score = 32/R (score > 1.0 means beating SOTA). Significance threshold: 1 (integer metric).

## Quick Start

```bash
uv sync
python research/eval/evaluator.py --sanity-check
```

## Validity

A decomposition (U, V, W) is valid iff:

```python
np.array_equal(np.einsum("ir,jr,kr->ijk", U, V, W), T)  # exact float32 equality
```

Factors should be half-integer-valued (STE rounding from continuous optimization). The rank is read from `U.shape[1]`.

## Structure

```
research/
  problem.md          # Problem definition and known results
  eval/               # Frozen evaluation harness (eval-v1)
    evaluator.py      # verify_tensor_decomposition(), evaluate_solution()
    baselines.py      # Naive R=40 and gradient baselines
    config.yaml       # Metric config, sanity check definitions
  figures/            # Visualizations
  style.md            # Plotting conventions
orbits/               # Agent experiment branches (created by /research)
docs/                 # GitHub Pages
```

## Campaign

Tracked on GitHub Issues: [Campaign #1](https://github.com/wwang2/matrix/issues/1)

Eval frozen at tag `eval-v1`. Any changes require an `[Eval Change]` issue.

Run `/research` to start the autonomous search loop.
