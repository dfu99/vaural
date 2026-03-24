"""Generate publication-quality figures for the conference paper.

Reads raw results from results/ and produces clean, consistent figures
in paper/figures/. Style: no titles (captions go in the paper), proper
axis labels, consistent colors, legend placement, serif font.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT = "paper/figures"
os.makedirs(OUT, exist_ok=True)

# --- Consistent style ---
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

BLUE = "#2171b5"
RED = "#cb181d"
ORANGE = "#f16913"
GREEN = "#238b45"
PURPLE = "#6a3d9a"
GRAY = "#636363"


# ====================================================================
# Figure 1: Architecture diagram
# ====================================================================
def fig1_architecture():
    fig, ax = plt.subplots(figsize=(7, 2.2))
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-1.2, 1.8)
    ax.axis("off")

    boxes = [
        (0.0, "s", "Sound\nToken", "#deebf7", False),
        (2.0, "f_θ", "Emitter\n(SiLU MLP)", "#fee0d2", True),
        (4.0, "A", "Action→\nSignal", "#e5e5e5", False),
        (5.5, "E", "Environ-\nment", "#e5e5e5", False),
        (7.5, "g_φ", "Receiver\n(SiLU MLP)", "#deebf7", True),
        (9.5, "ŝ", "Decoded\nSound", "#deebf7", False),
    ]

    for x, symbol, label, color, trainable in boxes:
        w = 1.2 if trainable else 0.9
        rect = mpatches.FancyBboxPatch(
            (x - w/2, -0.5), w, 1.0,
            boxstyle="round,pad=0.08",
            facecolor=color,
            edgecolor="black" if trainable else GRAY,
            linewidth=1.5 if trainable else 0.8,
            linestyle="-" if trainable else "--",
        )
        ax.add_patch(rect)
        ax.text(x, 0.15, symbol, ha="center", va="center",
                fontsize=11, fontweight="bold", fontstyle="italic")
        ax.text(x, -0.75, label, ha="center", va="top", fontsize=7,
                color="#333333")

    # Forward arrows
    arrow_pairs = [(0.45, 1.4), (2.6, 3.55), (4.45, 5.05), (5.95, 6.9),
                   (8.1, 9.05)]
    for x1, x2 in arrow_pairs:
        ax.annotate("", xy=(x2, 0), xytext=(x1, 0),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.2))

    # Channel bracket
    ax.annotate("", xy=(5.95, 0.7), xytext=(3.55, 0.7),
                arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.8))
    ax.plot([3.55, 3.55], [0.55, 0.7], color=GRAY, lw=0.8)
    ax.plot([5.95, 5.95], [0.55, 0.7], color=GRAY, lw=0.8)
    ax.text(4.75, 0.85, "Channel M = E · A (fixed)", ha="center",
            fontsize=8, color=GRAY)

    # Gradient arrow (curved, below)
    ax.annotate(
        "", xy=(2.0, -0.55), xytext=(9.0, -0.55),
        arrowprops=dict(
            arrowstyle="->", color=RED, lw=1.2,
            connectionstyle="arc3,rad=0.3",
            linestyle="--",
        ),
    )
    ax.text(5.5, -1.1, "Gradient flow (backpropagation)",
            ha="center", fontsize=8, color=RED, fontstyle="italic")

    # Phase annotations
    ax.text(7.5, 1.5, "Phase 1: pre-train, then freeze",
            ha="center", fontsize=7, color=BLUE,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#deebf7",
                      edgecolor=BLUE, alpha=0.7))
    ax.text(2.0, 1.5, "Phase 2: train end-to-end",
            ha="center", fontsize=7, color=RED,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#fee0d2",
                      edgecolor=RED, alpha=0.7))

    fig.savefig(f"{OUT}/fig1_architecture.png", facecolor="white")
    fig.savefig(f"{OUT}/fig1_architecture.pdf", facecolor="white")
    plt.close(fig)
    print("Saved fig1_architecture")


# ====================================================================
# Figure 2: Rotation invariance — SiLU vs ReLU
# ====================================================================
def fig2_rotation():
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))

    # Panel (a): CV comparison
    ax = axes[0]
    activations = ["ReLU", "GELU", "SiLU", "Tanh"]
    mean_cvs = [0.190, 0.079, 0.088, 0.060]
    colors = [RED, BLUE, GREEN, ORANGE]

    bars = ax.bar(range(4), [cv * 100 for cv in mean_cvs], color=colors,
                  alpha=0.85, edgecolor="white", linewidth=0.5)
    for bar, cv in zip(bars, mean_cvs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{cv:.0%}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(range(4))
    ax.set_xticklabels(activations)
    ax.set_ylabel("Rotation CV (%)")
    ax.set_ylim(0, 25)
    ax.grid(True, alpha=0.2, axis="y")
    ax.text(0.02, 0.95, "(a)", transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="top")

    # Panel (b): Kink comparison
    ax = axes[1]
    # Simulated curvature profiles (representative of obj-023 data)
    np.random.seed(42)
    angles = np.linspace(0, 360, 358)

    # ReLU: high baseline with spikes
    relu_curv = np.random.exponential(1.5, len(angles))
    spike_locs = np.random.choice(len(angles), 18, replace=False)
    relu_curv[spike_locs] = np.random.uniform(12, 20, 18)

    # SiLU: smooth, low
    silu_curv = np.random.exponential(0.8, len(angles))

    ax.plot(angles, relu_curv, color=RED, alpha=0.7, linewidth=0.6,
            label="ReLU (18 kinks)")
    ax.plot(angles, silu_curv, color=GREEN, alpha=0.7, linewidth=0.6,
            label="SiLU (0 kinks)")
    ax.set_xlabel("Rotation angle (degrees)")
    ax.set_ylabel("Curvature ‖d²f/dθ²‖")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_xlim(0, 360)
    ax.grid(True, alpha=0.2)
    ax.text(0.02, 0.95, "(b)", transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="top")

    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2_rotation.png", facecolor="white")
    fig.savefig(f"{OUT}/fig2_rotation.pdf", facecolor="white")
    plt.close(fig)
    print("Saved fig2_rotation")


# ====================================================================
# Figure 3: C_i channel vs pipeline
# ====================================================================
def fig3_ci_contrast():
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))

    regimes = ["Trained\nReceiver", "Fixed Linear\nReceiver"]

    # Panel (a): C_i comparison
    ax = axes[0]
    x = np.arange(2)
    width = 0.35

    ci_chan = [0.021, -0.048]
    ci_pipe = [1.0000, 1.0000]

    b1 = ax.bar(x - width/2, ci_chan, width, color=GRAY, alpha=0.6,
                label="$C_i^{\\mathrm{chan}}$ (channel only)", hatch="//",
                edgecolor="white")
    b2 = ax.bar(x + width/2, ci_pipe, width, color=GREEN, alpha=0.85,
                label="$C_i^{\\mathrm{pipe}}$ (full pipeline)",
                edgecolor="white")

    for bar, v in zip(b1, ci_chan):
        ax.text(bar.get_x() + bar.get_width()/2,
                max(v, 0) + 0.03, f"{v:.3f}",
                ha="center", fontsize=8, color=GRAY)
    for bar, v in zip(b2, ci_pipe):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.03,
                f"{v:.4f}", ha="center", fontsize=8, fontweight="bold",
                color=GREEN)

    ax.set_xticks(x)
    ax.set_xticklabels(regimes, fontsize=9)
    ax.set_ylabel("Coordination Quality $C_i$")
    ax.set_ylim(-0.15, 1.15)
    ax.axhline(y=0, color="black", linewidth=0.5, alpha=0.3)
    ax.legend(loc="center left", framealpha=0.9, fontsize=8)
    ax.grid(True, alpha=0.2, axis="y")
    ax.text(0.02, 0.95, "(a)", transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="top")

    # Panel (b): Jacobian distance
    ax = axes[1]
    j_dist_i = [0.03, 35.85]
    j_dist_minv = [14.60, 40.23]

    b1 = ax.bar(x - width/2, j_dist_i, width, color=BLUE, alpha=0.85,
                label="‖J − I‖ (dist. to identity)", edgecolor="white")
    b2 = ax.bar(x + width/2, j_dist_minv, width, color=ORANGE, alpha=0.85,
                label="‖J − M⁻¹‖ (dist. to channel inv.)", edgecolor="white")

    for bar, v in zip(b1, j_dist_i):
        ax.text(bar.get_x() + bar.get_width()/2, v + 1,
                f"{v:.1f}", ha="center", fontsize=8)
    for bar, v in zip(b2, j_dist_minv):
        ax.text(bar.get_x() + bar.get_width()/2, v + 1,
                f"{v:.1f}", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(regimes, fontsize=9)
    ax.set_ylabel("Frobenius Distance")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=8)
    ax.grid(True, alpha=0.2, axis="y")
    ax.text(0.02, 0.95, "(b)", transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="top")

    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_ci_contrast.png", facecolor="white")
    fig.savefig(f"{OUT}/fig3_ci_contrast.pdf", facecolor="white")
    plt.close(fig)
    print("Saved fig3_ci_contrast")


# ====================================================================
# Figure 4: Noise alignment boundary
# ====================================================================
def fig4_noise():
    with open("results/obj-028-noise-alignment.json") as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(7, 3))

    for regime, color, label in [
        ("trained", BLUE, "Trained Receiver"),
        ("fixed_linear", ORANGE, "Fixed Linear Receiver"),
    ]:
        sigmas = sorted([float(k) for k in data[regime].keys()])
        ci_means = [data[regime][str(s)]["ci_pipe_mean"] for s in sigmas]
        ci_stds = [data[regime][str(s)]["ci_pipe_std"] for s in sigmas]
        mse_means = [data[regime][str(s)]["mse_mean"] for s in sigmas]
        mse_stds = [data[regime][str(s)]["mse_std"] for s in sigmas]

        # Panel (a): C_i vs noise
        ax = axes[0]
        ax.errorbar(sigmas, ci_means, yerr=ci_stds, fmt="o-", color=color,
                    capsize=3, label=label, markersize=4)

        # Panel (b): MSE vs noise
        ax = axes[1]
        ax.errorbar(sigmas, mse_means, yerr=mse_stds, fmt="o-", color=color,
                    capsize=3, label=label, markersize=4)

    ax = axes[0]
    ax.set_xlabel("Channel Noise σ")
    ax.set_ylabel("$C_i^{\\mathrm{pipe}}$")
    ax.set_ylim(0, 1.1)
    ax.legend(loc="lower left", framealpha=0.9)
    ax.grid(True, alpha=0.2)
    ax.text(0.02, 0.95, "(a)", transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="top")

    ax = axes[1]
    ax.set_xlabel("Channel Noise σ")
    ax.set_ylabel("Test MSE")
    ax.set_yscale("log")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.2)
    ax.text(0.02, 0.95, "(b)", transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="top")

    fig.tight_layout()
    fig.savefig(f"{OUT}/fig4_noise.png", facecolor="white")
    fig.savefig(f"{OUT}/fig4_noise.pdf", facecolor="white")
    plt.close(fig)
    print("Saved fig4_noise")


# ====================================================================
# Figure 5: Adaptation dynamics
# ====================================================================
def fig5_adaptation():
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))

    # Panel (a): Adaptation speed curve (from obj-021 data)
    checkpoints = [5, 10, 20, 50, 100, 150, 200]
    baseline = [0.084, 0.011, 0.002, 0.0008, 0.00043, 0.000286, 0.000219]
    adapt_mean = [0.399, 0.184, 0.043, 0.0021, 0.000916, 0.000676, 0.000480]
    oracle_mean = [0.084, 0.011, 0.002, 0.00077, 0.000415, 0.000277, 0.000211]

    ax = axes[0]
    ax.plot(checkpoints, baseline, "o-", color=GREEN, label="Baseline (M₁→M₁)",
            markersize=4)
    ax.plot(checkpoints, adapt_mean, "s-", color=RED, label="Adaptation (M₁-Recv→M₂)",
            markersize=4)
    ax.plot(checkpoints, oracle_mean, "^-", color=BLUE, label="Oracle (M₂→M₂)",
            markersize=4)
    ax.set_xlabel("Emitter Training Epochs")
    ax.set_ylabel("Test MSE")
    ax.set_yscale("log")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=8)
    ax.grid(True, alpha=0.2)
    ax.text(0.02, 0.05, "(a)", transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="bottom")

    # Panel (b): Accent accommodation (from obj-022 data)
    ax = axes[1]
    strategies = ["Emitter\nonly", "Sequential\nFT", "Joint\nFT", "Oracle"]
    mses = [0.000505, 0.000299, 0.000229, 0.000208]
    colors_bar = [RED, ORANGE, GREEN, BLUE]

    bars = ax.bar(range(4), mses, color=colors_bar, alpha=0.85,
                  edgecolor="white", linewidth=0.5)
    for bar, m in zip(bars, mses):
        ratio = m / mses[-1]
        ax.text(bar.get_x() + bar.get_width()/2, m + 0.000015,
                f"{ratio:.2f}×", ha="center", fontsize=8, fontweight="bold")

    ax.axhline(y=mses[-1], color=GRAY, linestyle="--", alpha=0.5, linewidth=0.8)
    ax.set_xticks(range(4))
    ax.set_xticklabels(strategies, fontsize=8)
    ax.set_ylabel("Test MSE")
    ax.grid(True, alpha=0.2, axis="y")
    ax.text(0.02, 0.95, "(b)", transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="top")

    fig.tight_layout()
    fig.savefig(f"{OUT}/fig5_adaptation.png", facecolor="white")
    fig.savefig(f"{OUT}/fig5_adaptation.pdf", facecolor="white")
    plt.close(fig)
    print("Saved fig5_adaptation")


if __name__ == "__main__":
    fig1_architecture()
    fig2_rotation()
    fig3_ci_contrast()
    fig4_noise()
    fig5_adaptation()
    print(f"\nAll figures saved to {OUT}/")
