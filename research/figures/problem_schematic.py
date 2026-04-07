"""
problem_schematic.py

Generates two figures for the R(2,4,5) matrix multiplication tensor rank research campaign:
  1. rank_landscape.png  — horizontal bar chart of known bounds
  2. algorithm_approaches.png — grouped chart of algorithm approaches
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ---------------------------------------------------------------------------
# Figure 1: rank_landscape.png
# ---------------------------------------------------------------------------
fig1, ax1 = plt.subplots(figsize=(10, 5))

# Bar data (label, rank, color)
bars = [
    ("Naive algorithm",            40, "#999999"),
    ("Ternary coefficients",       33, "#90caf9"),  # light blue
    ("SOTA (AlphaEvolve 2025)",    32, "#1565c0"),  # blue
]

y_positions = np.arange(len(bars))
labels      = [b[0] for b in bars]
ranks       = [b[1] for b in bars]
colors      = [b[2] for b in bars]

bar_height = 0.5
hbars = ax1.barh(y_positions, ranks, height=bar_height, color=colors, zorder=3)

# Annotate bar values
for bar, rank in zip(hbars, ranks):
    ax1.text(
        bar.get_width() + 0.4,
        bar.get_y() + bar.get_height() / 2,
        f"R = {rank}",
        va="center", ha="left", fontsize=12, fontweight="bold",
    )

# Shaded open-gap region (rank 20 to 32)
ax1.axvspan(20, 32, alpha=0.18, color="#ff9800", zorder=1, label="Open gap (R = 21..31)")
ax1.text(
    26, -0.65,
    "Open gap (R = 21..31)",
    ha="center", va="center", fontsize=11, color="#e65100", fontweight="bold",
)

# Trivial lower bound dashed line
ax1.axvline(x=20, color="#c62828", linewidth=2, linestyle="--", zorder=4, label="Trivial lower bound (R >= 20)")
ax1.text(
    20, len(bars) - 0.05,
    "  Lower bound\n  R >= 20",
    ha="left", va="top", fontsize=10, color="#c62828",
)

# Axis formatting
ax1.set_xlim(0, 42)
ax1.set_yticks(y_positions)
ax1.set_yticklabels(labels, fontsize=12)
ax1.set_xlabel("Rank R", fontsize=13)
ax1.set_title("R(2,4,5) Matrix Multiplication Tensor Rank: Known Bounds", fontsize=14, pad=14)

# Only x gridlines
ax1.xaxis.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)
ax1.set_axisbelow(True)
ax1.yaxis.grid(False)

# Legend
legend_handles = [
    mpatches.Patch(color="#ff9800", alpha=0.35, label="Open gap (R = 21..31)"),
    plt.Line2D([0], [0], color="#c62828", linewidth=2, linestyle="--", label="Trivial lower bound (R >= 20)"),
]
ax1.legend(handles=legend_handles, loc="lower right", fontsize=11)

fig1.tight_layout()
fig1.savefig("/Users/wujiewang/code/matrix/research/figures/rank_landscape.png", dpi=150)
print("Saved rank_landscape.png")
plt.close(fig1)


# ---------------------------------------------------------------------------
# Figure 2: algorithm_approaches.png
# ---------------------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(9, 6))

algo_labels = [
    "Gradient +\nSTE rounding",
    "Evolutionary\n(AlphaEvolve)",
    "Flip graph\n(ternary)",
    "SAT /\nbacktracking",
]
best_ranks  = [32, 32, 33, None]   # None = no upper bound achieved
reliabilities = ["medium", "high", "high", "N/A (lower\nbounds only)"]

x_pos = np.arange(len(algo_labels))
bar_colors = ["#42a5f5", "#1565c0", "#90caf9", "#bdbdbd"]

# Plot bars where rank is available
rank_values = [r if r is not None else 0 for r in best_ranks]
bars2 = ax2.bar(x_pos, rank_values, color=bar_colors, width=0.55, zorder=3)

# Annotate bars
for i, (bar, rank, rel) in enumerate(zip(bars2, best_ranks, reliabilities)):
    if rank is not None:
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"R = {rank}",
            ha="center", va="bottom", fontsize=12, fontweight="bold",
        )
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            1.5,
            f"reliability:\n{rel}",
            ha="center", va="bottom", fontsize=9.5, color="#333333",
        )
    else:
        # SAT/backtracking bar is 0 — add special annotation
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            1.5,
            f"{rel}",
            ha="center", va="bottom", fontsize=9.5, color="#555555",
        )
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            0.3,
            "—",
            ha="center", va="bottom", fontsize=14, color="#777777",
        )

# SOTA reference line at R=32
ax2.axhline(y=32, color="#c62828", linewidth=2, linestyle="--", zorder=4, label="SOTA: R = 32")
ax2.text(
    len(algo_labels) - 0.45, 32.35,
    "SOTA (R = 32)",
    ha="right", va="bottom", fontsize=10.5, color="#c62828", fontweight="bold",
)

# Axis formatting
ax2.set_xticks(x_pos)
ax2.set_xticklabels(algo_labels, fontsize=12)
ax2.set_ylabel("Best achieved rank R", fontsize=13)
ax2.set_ylim(0, 42)
ax2.set_title("Algorithm Approaches for R(2,4,5)", fontsize=14, pad=14)

# Only y gridlines
ax2.yaxis.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)
ax2.set_axisbelow(True)
ax2.xaxis.grid(False)

ax2.legend(fontsize=11)

fig2.tight_layout()
fig2.savefig("/Users/wujiewang/code/matrix/research/figures/algorithm_approaches.png", dpi=150)
print("Saved algorithm_approaches.png")
plt.close(fig2)

print("Done.")
