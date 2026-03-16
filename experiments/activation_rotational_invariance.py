"""Experiment: Does Activation Function Cause Rotational Sensitivity?

obj-015 found the system is NOT rotationally invariant (CV 13-23%) and
hypothesized that ReLU axis-alignment is the cause. ReLU(x) = max(0, x)
operates element-wise, creating decision boundaries aligned with coordinate
hyperplanes. This creates preferred directions — when channel rotations
align "hard" directions with these boundaries, learning quality varies.

This experiment directly tests the hypothesis by comparing activations:
  - ReLU: axis-aligned boundaries (the hypothesized cause of rotation bias)
  - GELU: smooth, approximately Gaussian-gated — no sharp axis boundaries
  - SiLU (Swish): smooth, self-gated — no axis boundaries
  - Tanh: smooth, bounded — no axis boundaries but saturates

Prediction: If the ReLU hypothesis is correct, smooth activations should
show lower rotation CV (more rotationally invariant) than ReLU.

Design: Same as obj-015 — fix 3 spectra (κ=1, 10, 100), vary 8 random
rotations per spectrum, train sequential pipeline for each. Compare
rotation CV across activation functions.
"""

import os
import sys

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from components import ActionToSignal, Environment, Pipeline
from train import pretrain_receiver, train_emitter
from experiments.pure_rotational_invariance import make_orthogonal, make_channel_from_svd


class FlexibleMLP(nn.Module):
    """3-layer MLP with configurable activation function."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int,
                 activation: str = "relu"):
        super().__init__()
        act_fn = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "tanh": nn.Tanh(),
        }[activation]

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_and_evaluate(a2s, env, cfg, activation, n_test=2000, seed=42):
    """Train pipeline with specified activation and return test MSE."""
    torch.manual_seed(seed)

    # Build Receiver and Emitter with the specified activation
    receiver = FlexibleMLP(cfg.signal_dim, cfg.sound_dim, cfg.hidden_dim,
                           activation=activation)
    # Pretrain receiver (same interface as train.pretrain_receiver)
    recv_losses = pretrain_receiver(a2s, env, receiver, cfg)
    receiver.requires_grad_(False)

    emitter = FlexibleMLP(cfg.sound_dim, cfg.action_dim, cfg.hidden_dim,
                          activation=activation)
    pipeline = Pipeline(emitter, a2s, env, receiver)
    emit_losses = train_emitter(pipeline, cfg)

    # Evaluate
    torch.manual_seed(99)
    sounds = torch.randn(n_test, cfg.sound_dim)
    pipeline.eval()
    with torch.no_grad():
        reconstructed = pipeline(sounds)
        mse = (reconstructed - sounds).pow(2).mean().item()

    return mse, recv_losses, emit_losses


def run_experiment(dim=8, n_rotations=8, epochs_recv=400, epochs_emit=500,
                   n_samples=2000):
    cfg = Config(
        sound_dim=dim, action_dim=dim, signal_dim=dim,
        hidden_dim=64,
        receiver_lr=1e-3, receiver_epochs=epochs_recv,
        receiver_samples=n_samples, receiver_batch_size=64,
        emitter_lr=1e-3, emitter_epochs=epochs_emit,
        emitter_samples=n_samples, emitter_batch_size=64,
        plot_every=200,
    )

    activations = ["relu", "gelu", "silu", "tanh"]

    spectra = {
        "flat (κ=1)": torch.ones(dim),
        "moderate (κ=10)": torch.logspace(0, -1, dim),
        "steep (κ=100)": torch.logspace(0, -2, dim),
    }

    results = {}

    for act_name in activations:
        results[act_name] = {}
        for spec_name, sigmas in spectra.items():
            kappa = (sigmas[0] / sigmas[-1]).item()
            print(f"\n{'='*60}")
            print(f"Activation: {act_name.upper()}, Spectrum: {spec_name}")
            print(f"{'='*60}")

            mses = []
            for rot_idx in range(n_rotations):
                seed_u = 1000 + rot_idx * 7
                seed_v = 2000 + rot_idx * 13

                a2s, env, M = make_channel_from_svd(dim, sigmas, seed_u, seed_v)

                mse, _, _ = train_and_evaluate(
                    a2s, env, cfg, act_name, seed=42
                )
                mses.append(mse)
                print(f"  Rot {rot_idx+1}/{n_rotations}: MSE={mse:.6f}")

            mses = np.array(mses)
            cv = mses.std() / mses.mean() if mses.mean() > 0 else 0
            results[act_name][spec_name] = {
                "kappa": kappa,
                "mses": mses,
                "mean": mses.mean(),
                "std": mses.std(),
                "cv": cv,
                "min": mses.min(),
                "max": mses.max(),
            }

            print(f"  → Mean={mses.mean():.6f} ± {mses.std():.6f} "
                  f"(CV={cv:.1%}, max/min={mses.max()/max(mses.min(),1e-10):.2f}×)")

    return results, dim


def plot_results(results, dim, output_path):
    activations = list(results.keys())
    spec_names = list(results[activations[0]].keys())
    n_acts = len(activations)
    n_specs = len(spec_names)

    act_colors = {
        "relu": "#E91E63",
        "gelu": "#2196F3",
        "silu": "#4CAF50",
        "tanh": "#FF9800",
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # --- Row 1: CV comparison (the key result) ---
    # Panel 1: CV grouped by spectrum
    ax = axes[0, 0]
    x = np.arange(n_specs)
    width = 0.8 / n_acts
    for i, act in enumerate(activations):
        cvs = [results[act][s]["cv"] for s in spec_names]
        offset = (i - (n_acts - 1) / 2) * width
        bars = ax.bar(x + offset, cvs, width, color=act_colors[act],
                      alpha=0.8, label=act.upper())
        for bar, cv in zip(bars, cvs):
            ax.text(bar.get_x() + bar.get_width() / 2, cv + 0.005,
                    f"{cv:.0%}", ha="center", va="bottom", fontsize=7,
                    fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"κ={results[activations[0]][s]['kappa']:.0f}"
                        for s in spec_names])
    ax.set_ylabel("Coefficient of Variation")
    ax.set_title("Rotation Sensitivity by Activation\n(lower = more rotationally invariant)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 2: Mean MSE grouped by spectrum
    ax = axes[0, 1]
    for i, act in enumerate(activations):
        means = [results[act][s]["mean"] for s in spec_names]
        stds = [results[act][s]["std"] for s in spec_names]
        offset = (i - (n_acts - 1) / 2) * width
        ax.bar(x + offset, means, width, yerr=stds, color=act_colors[act],
               alpha=0.8, label=act.upper(), capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"κ={results[activations[0]][s]['kappa']:.0f}"
                        for s in spec_names])
    ax.set_ylabel("Mean Test MSE")
    ax.set_yscale("log")
    ax.set_title("Reconstruction Quality by Activation\n(error bars = ±1σ across rotations)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 3: Max/min ratio (worst-case spread)
    ax = axes[0, 2]
    for i, act in enumerate(activations):
        ratios = [results[act][s]["max"] / max(results[act][s]["min"], 1e-10)
                  for s in spec_names]
        offset = (i - (n_acts - 1) / 2) * width
        ax.bar(x + offset, ratios, width, color=act_colors[act],
               alpha=0.8, label=act.upper())
    ax.set_xticks(x)
    ax.set_xticklabels([f"κ={results[activations[0]][s]['kappa']:.0f}"
                        for s in spec_names])
    ax.set_ylabel("Max/Min MSE Ratio")
    ax.set_title("Worst-Case Rotation Spread\n(1.0 = perfectly invariant)")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # --- Row 2: Per-spectrum scatter plots ---
    for col, spec_name in enumerate(spec_names):
        ax = axes[1, col]
        kappa = results[activations[0]][spec_name]["kappa"]

        for i, act in enumerate(activations):
            r = results[act][spec_name]
            mses = r["mses"]
            jitter = np.random.default_rng(42 + i).uniform(-0.15, 0.15,
                                                            len(mses))
            ax.scatter(np.full_like(mses, i) + jitter, mses,
                       color=act_colors[act], s=60, alpha=0.7,
                       edgecolors="white", linewidth=0.5, zorder=3)
            ax.errorbar(i, r["mean"], yerr=r["std"], fmt="D",
                        color="black", markersize=6, capsize=6,
                        capthick=1.5, zorder=4)

        ax.set_xticks(range(n_acts))
        ax.set_xticklabels([a.upper() for a in activations], fontsize=9)
        ax.set_ylabel("Test MSE")
        ax.set_title(f"κ={kappa:.0f}: MSE Distribution per Rotation")
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"Activation Function × Rotational Invariance (dim={dim})\n"
        f"Does smooth activation reduce rotation sensitivity?",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved: {output_path}")


def print_summary(results):
    activations = list(results.keys())
    spec_names = list(results[activations[0]].keys())

    print("\n" + "=" * 80)
    print("SUMMARY: Activation × Rotational Invariance")
    print("=" * 80)

    # Table header
    header = f"  {'Activation':<10}"
    for s in spec_names:
        kappa = results[activations[0]][s]["kappa"]
        header += f" | {'κ=' + str(int(kappa)) + ' MSE':>12} {'CV':>6} {'max/min':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for act in activations:
        row = f"  {act.upper():<10}"
        for s in spec_names:
            r = results[act][s]
            ratio = r["max"] / max(r["min"], 1e-10)
            row += f" | {r['mean']:>12.6f} {r['cv']:>5.1%} {ratio:>7.2f}×"
        print(row)

    # Identify best activation for rotation invariance
    print("\n  Best rotation invariance (lowest mean CV across spectra):")
    mean_cvs = {}
    for act in activations:
        cvs = [results[act][s]["cv"] for s in spec_names]
        mean_cvs[act] = np.mean(cvs)
    for act in sorted(mean_cvs, key=mean_cvs.get):
        print(f"    {act.upper()}: mean CV = {mean_cvs[act]:.1%}")

    print("\n  Best reconstruction (lowest mean MSE across spectra):")
    mean_mses = {}
    for act in activations:
        mses_all = [results[act][s]["mean"] for s in spec_names]
        mean_mses[act] = np.mean(mses_all)
    for act in sorted(mean_mses, key=mean_mses.get):
        print(f"    {act.upper()}: mean MSE = {mean_mses[act]:.6f}")

    # Test the hypothesis
    relu_cv = mean_cvs["relu"]
    smooth_cvs = {k: v for k, v in mean_cvs.items() if k != "relu"}
    all_smooth_lower = all(v < relu_cv for v in smooth_cvs.values())
    any_smooth_lower = any(v < relu_cv for v in smooth_cvs.values())

    print("\n  HYPOTHESIS TEST: ReLU axis-alignment causes rotation sensitivity")
    if all_smooth_lower:
        print("  → SUPPORTED: All smooth activations have lower rotation CV than ReLU")
    elif any_smooth_lower:
        winners = [k for k, v in smooth_cvs.items() if v < relu_cv]
        print(f"  → PARTIALLY SUPPORTED: {', '.join(w.upper() for w in winners)} "
              f"beat ReLU on rotation CV")
    else:
        print("  → NOT SUPPORTED: No smooth activation beats ReLU on rotation CV")
        print("    Rotation sensitivity may not be primarily caused by activation shape")


def main():
    os.makedirs("results", exist_ok=True)

    print("=" * 60)
    print("EXPERIMENT: Activation Function × Rotational Invariance")
    print("Does smooth activation reduce rotation sensitivity?")
    print("=" * 60)

    results, dim = run_experiment(
        dim=8, n_rotations=8, epochs_recv=400, epochs_emit=500, n_samples=2000
    )

    plot_results(results, dim,
                 "results/obj-017-activation-rotational-invariance.png")
    print_summary(results)


if __name__ == "__main__":
    main()
