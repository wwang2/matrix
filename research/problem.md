# Research Problem: Minimal Tensor Rank of ⟨2,4,5⟩

## Question

What is the minimal tensor rank R(2,4,5) — the smallest number of scalar multiplications needed to compute the product of a 2×4 matrix with a 4×5 matrix using a bilinear algorithm?

Concretely: find the smallest integer R such that there exist real matrices U ∈ ℝ^{8×R}, V ∈ ℝ^{20×R}, W ∈ ℝ^{10×R} with

    T_ijk = Σ_{r=1}^{R} U_{ir} V_{jr} W_{kr}

where T ∈ ℝ^{8×20×10} is the standard matrix multiplication tensor for (n=2, m=4, p=5).

## Background

The **matrix multiplication tensor** T_{n,m,p} encodes the bilinear map computing C = A·B for A ∈ ℝ^{n×m}, B ∈ ℝ^{m×p}. Its rank (the minimum number of rank-1 terms needed to decompose it) equals the minimum number of scalar multiplications in any bilinear algorithm for this matrix product.

The naive algorithm uses n·m·p = 40 multiplications for (2,4,5). Finding a rank-R decomposition for R < 40 directly gives a faster algorithm. This is the generalization of Strassen's 1969 discovery that 2×2 matrix multiplication can be done in 7 (not 8) multiplications.

**Why (2,4,5)?**
- It is a non-square format with a substantial gap between the naive upper bound (40) and SOTA (32)
- It relates to ⟨2,4,4⟩ and ⟨2,5,5⟩ via dimension augmentation
- By cyclic permutation symmetry: R(2,4,5) = R(4,5,2) = R(5,2,4)
- Improvements here can contribute to bounds on the asymptotic matrix multiplication exponent ω

**Key identities:**
- T_{n,m,p} has shape (nm) × (mp) × (np) = 8 × 20 × 10
- T[i·m+j, j·p+k, k·n+i] = 1 for 0 ≤ i < n, 0 ≤ j < m, 0 ≤ k < p (all other entries 0)

## Known Results

| Bound | Value | Source | Year |
|-------|-------|--------|------|
| Naive upper bound | 40 | Standard | — |
| SOTA upper bound | **32** | AlphaEvolve (Novikov et al.) | 2025 |
| Previous upper bound (ternary coefficients) | 33 | Meta flip graph (Kauers et al.) | 2024 |
| Trivial lower bound (max flattening) | 20 | Classical | — |
| Best known lower bound | ~20 | (no dedicated bound published) | — |

**Gap:** Lower bound ≈ 20, upper bound = 32. Values R = 31, 30, ..., 21 are all open.

**Related tensors:**

| Format | SOTA Rank | Source |
|--------|-----------|--------|
| ⟨2,2,2⟩ | 7 (exact) | Strassen 1969 |
| ⟨2,4,4⟩ | 26 | Hopcroft & Kerr 1971 |
| ⟨2,5,5⟩ | 40 | Hopcroft & Kerr 1971 |
| ⟨3,4,5⟩ | 47 | AlphaTensor 2022 |
| ⟨4,4,4⟩ complex | 48 | AlphaEvolve 2025 |

## Success Criteria

**Metric:** `rank` (integer, minimize)  
**Direction:** minimize  
**Current SOTA:** 32  
**Baseline (evaluator default):** 32  

**Victory conditions (ordered):**
1. **Beat SOTA**: Find a valid decomposition with R ≤ 31 → breakthrough result
2. **Match SOTA**: Reproduce R = 32 reliably with a novel algorithm → strong result
3. **Near-SOTA**: Reliably achieve R ≤ 34 with a new algorithmic approach → useful

**Validity requirement (hard constraint):**
The reconstructed tensor from the factor matrices must be **exactly equal** to T (not approximately):

    np.array_equal(np.einsum("ir,jr,kr->ijk", U, V, W), T)   # T stored as float32 {0,1} entries

Any decomposition failing this check is invalid regardless of rank.

**Coefficient types:** Factor matrices U, V, W may have real-valued entries, **but must be representable in float32 such that einsum produces exact {0,1} output**. In practice, this means factors whose entries are half-integers (multiples of 0.5) or small integers/rationals with powers-of-2 denominators. The standard approach is continuous optimization followed by **Straight-Through Estimator (STE) rounding to the nearest half-integer** (as in the reference `initial_program.py`). A solution is valid iff it passes the exact-equality check after rounding.

**Lower bound (exact):** The trivial max-flattening lower bound gives exactly **R(2,4,5) ≥ 20** (since max(nm, mp, np) = max(8, 20, 10) = 20). No tighter dedicated lower bound is published for this specific format. The range R ∈ {20, ..., 31} is entirely open.

**Significance threshold:** R is an integer — any decrease of ≥ 1 is a genuine, significant improvement.

## Algorithmic Landscape

**Main approaches in the literature:**

1. **Gradient + rounding (continuous relaxation):** Minimize ||T - Σ u_r ⊗ v_r ⊗ w_r||² over ℝ, then round to exact. Used in Smirnov (2013+), initial_program.py baseline. Difficulty: non-convex landscape, rounding rarely preserves exactness.

2. **Evolutionary / LLM-driven (AlphaEvolve):** Evolve the optimization procedure itself. Achieves SOTA R=32. This is essentially the upper bound we're trying to beat.

3. **Discrete search / flip graphs:** Local neighborhood search over exact rational schemes. Ternary {-1,0,1} coefficients give rank 33. Full rational achieves 32.

4. **SAT / backtracking:** Prove lower bounds or enumerate exact solutions over finite fields. Computationally intensive; best for smaller formats.

5. **Algebraic / symmetry-based:** Exploit cyclic/transpose symmetries to reduce search space. Combined with any of the above.

## References

- Novikov et al. (2025), "AlphaEvolve: A coding agent for scientific and algorithmic discovery" — https://arxiv.org/abs/2506.13131
- Fawzi et al. (2022), "Discovering faster matrix multiplication algorithms with reinforcement learning" — Nature, https://www.nature.com/articles/s41586-022-05172-4
- Kauers et al. (2024), "Fast Matrix Multiplication via Ternary Meta Flip Graphs" — https://arxiv.org/abs/2511.20317
- Sedoglavic fast matrix multiplication catalog — https://fmm.univ-lille.fr/
- OpenEvolve evaluator: `openevolve/examples/alphaevolve_math_problems/matmul/evaluator.py`
- Strassen (1969), "Gaussian elimination is not optimal"
