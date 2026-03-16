"""Experiment: Diagnosing and Reducing Residual Rotation Sensitivity.

obj-017 showed SiLU reduces rotation CV from 19% (ReLU) to 8.8%, but it's
still not zero. This experiment investigates WHY residual sensitivity persists
and tests three interventions to reduce it further:

Questions:
  Q1. Is the 8.8% CV reproducible across training seeds, or is it noise?
      → Run multiple seeds per rotation to decompose variance into
        "rotation variance" vs "training variance"
  Q2. Does more capacity (wider hidden dim) reduce rotation sensitivity?
      → Test hidden_dim=128, 256 with SiLU
  Q3. Does normalization (LayerNorm) reduce rotation sensitivity?
      → LayerNorm normalizes activations, potentially removing residual
        axis-dependent scaling effects that survive smooth activations

Design:
  - All use SiLU activation (established best from obj-017)
  - 4 configurations: baseline (h=64), wide (h=128), wider (h=256),
    baseline+LayerNorm (h=64)
  - 3 spectra: κ=1, 10, 100
  - 8 rotations per spectrum
  - For Q1: 3 training seeds per rotation for the baseline config
  - Total: ~108 pipeline trainings
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
from experiments.pure_rotational_invariance import make_channel_from_svd


class SiLUMLP(nn.Module):
    """3-layer MLP with SiLU activation."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SiLUMLPLayerNorm(nn.Module):
    """3-layer MLP with SiLU + LayerNorm after each hidden layer."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_and_evaluate(a2s, env, cfg, mlp_class, hidden_dim, n_test=2000, seed=42):
    """Train pipeline with specified MLP class and return test MSE."""
    torch.manual_seed(seed)

    receiver = mlp_class(cfg.signal_dim, cfg.sound_dim, hidden_dim)
    recv_losses = pretrain_receiver(a2s, env, receiver, cfg)
    receiver.requires_grad_(False)

    emitter = mlp_class(cfg.sound_dim, cfg.action_dim, hidden_dim)
    pipeline = Pipeline(emitter, a2s, env, receiver)
    emit_losses = train_emitter(pipeline, cfg)

    torch.manual_seed(99)
    sounds = torch.randn(n_test, cfg.sound_dim)
    pipeline.eval()
    with torch.no_grad():
        reconstructed = pipeline(sounds)
        mse = (reconstructed - sounds).pow(2).mean().item()

    return mse, recv_losses, emit_losses


def run_seed_variance_test(dim=8, n_rotations=8, n_seeds=3,
                           epochs_recv=400, epochs_emit=500, n_samples=2000):
    """Q1: Decompose variance into rotation vs training seed components."""
    print("\n" + "=" * 70)
    print("Q1: Is residual rotation CV from training noise or rotation structure?")
    print("=" * 70)

    cfg = Config(
        sound_dim=dim, action_dim=dim, signal_dim=dim, hidden_dim=64,
        receiver_lr=1e-3, receiver_epochs=epochs_recv,
        receiver_samples=n_samples, receiver_batch_size=64,
        emitter_lr=1e-3, emitter_epochs=epochs_emit,
        emitter_samples=n_samples, emitter_batch_size=64,
        plot_every=9999,
    )

    # Use κ=10 where rotation effects are most visible
    sigmas = torch.logspace(0, -1, dim)
    kappa = (sigmas[0] / sigmas[-1]).item()
    print(f"Spectrum: κ={kappa:.0f}")

    # For each rotation, run n_seeds different training seeds
    rotation_means = []
    rotation_stds = []
    all_mses = []  # (n_rotations, n_seeds)

    for rot_idx in range(n_rotations):
        seed_u = 1000 + rot_idx * 7
        seed_v = 2000 + rot_idx * 13
        a2s, env, M = make_channel_from_svd(dim, sigmas, seed_u, seed_v)

        seed_mses = []
        for s in range(n_seeds):
            train_seed = 42 + s * 100
            mse, _, _ = train_and_evaluate(
                a2s, env, cfg, SiLUMLP, 64, seed=train_seed
            )
            seed_mses.append(mse)
            print(f"  Rot {rot_idx+1}/{n_rotations}, seed {s+1}/{n_seeds}: MSE={mse:.6f}")

        seed_mses = np.array(seed_mses)
        rotation_means.append(seed_mses.mean())
        rotation_stds.append(seed_mses.std())
        all_mses.append(seed_mses)

    rotation_means = np.array(rotation_means)
    rotation_stds = np.array(rotation_stds)
    all_mses = np.array(all_mses)

    # Decompose variance
    # Total variance = between-rotation variance + within-rotation (seed) variance
    grand_mean = all_mses.mean()
    between_var = rotation_means.var()
    within_var = (rotation_stds ** 2).mean()
    total_var = all_mses.var()

    rotation_cv = rotation_means.std() / rotation_means.mean()
    mean_seed_cv = np.mean(rotation_stds / rotation_means)

    results = {
        "all_mses": all_mses,
        "rotation_means": rotation_means,
        "rotation_stds": rotation_stds,
        "between_var": between_var,
        "within_var": within_var,
        "total_var": total_var,
        "variance_ratio": between_var / max(total_var, 1e-20),
        "rotation_cv": rotation_cv,
        "mean_seed_cv": mean_seed_cv,
        "kappa": kappa,
    }

    print(f"\n  Between-rotation variance: {between_var:.2e} ({results['variance_ratio']:.0%} of total)")
    print(f"  Within-rotation (seed) variance: {within_var:.2e} ({within_var/max(total_var,1e-20):.0%} of total)")
    print(f"  Rotation CV: {rotation_cv:.1%}")
    print(f"  Mean seed CV: {mean_seed_cv:.1%}")

    return results


def run_capacity_test(dim=8, n_rotations=8, epochs_recv=400, epochs_emit=500,
                      n_samples=2000):
    """Q2 + Q3: Test wider networks and LayerNorm."""
    print("\n" + "=" * 70)
    print("Q2+Q3: Does more capacity or normalization reduce rotation CV?")
    print("=" * 70)

    configs = {
        "SiLU h=64": (SiLUMLP, 64),
        "SiLU h=128": (SiLUMLP, 128),
        "SiLU h=256": (SiLUMLP, 256),
        "SiLU+LN h=64": (SiLUMLPLayerNorm, 64),
    }

    spectra = {
        "κ=1": torch.ones(dim),
        "κ=10": torch.logspace(0, -1, dim),
    }

    results = {}
    for config_name, (mlp_class, hidden_dim) in configs.items():
        results[config_name] = {}
        cfg = Config(
            sound_dim=dim, action_dim=dim, signal_dim=dim, hidden_dim=hidden_dim,
            receiver_lr=1e-3, receiver_epochs=epochs_recv,
            receiver_samples=n_samples, receiver_batch_size=64,
            emitter_lr=1e-3, emitter_epochs=epochs_emit,
            emitter_samples=n_samples, emitter_batch_size=64,
            plot_every=9999,
        )

        for spec_name, sigmas in spectra.items():
            kappa = (sigmas[0] / sigmas[-1]).item()
            print(f"\n  {config_name}, {spec_name}")

            mses = []
            for rot_idx in range(n_rotations):
                seed_u = 1000 + rot_idx * 7
                seed_v = 2000 + rot_idx * 13
                a2s, env, M = make_channel_from_svd(dim, sigmas, seed_u, seed_v)

                mse, _, _ = train_and_evaluate(
                    a2s, env, cfg, mlp_class, hidden_dim, seed=42
                )
                mses.append(mse)

            mses = np.array(mses)
            cv = mses.std() / mses.mean() if mses.mean() > 0 else 0
            results[config_name][spec_name] = {
                "kappa": kappa,
                "mses": mses,
                "mean": mses.mean(),
                "std": mses.std(),
                "cv": cv,
                "min": mses.min(),
                "max": mses.max(),
            }
            print(f"    Mean={mses.mean():.6f} ± {mses.std():.6f} (CV={cv:.1%})")

    return results


def plot_results(seed_results, capacity_results, dim, output_path):
    """Generate comprehensive visualization."""
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.35)

    config_colors = {
        "SiLU h=64": "#4CAF50",
        "SiLU h=128": "#2196F3",
        "SiLU h=256": "#9C27B0",
        "SiLU+LN h=64": "#FF9800",
    }

    # --- Panel 1: Seed variance decomposition ---
    ax = fig.add_subplot(gs[0, 0:2])
    all_mses = seed_results["all_mses"]
    n_rot, n_seeds = all_mses.shape
    for rot_idx in range(n_rot):
        jitter = np.random.default_rng(rot_idx).uniform(-0.12, 0.12, n_seeds)
        ax.scatter(
            np.full(n_seeds, rot_idx) + jitter,
            all_mses[rot_idx] * 1e6,
            s=50, alpha=0.7, color="#2196F3", edgecolors="white", linewidth=0.5
        )
        ax.errorbar(
            rot_idx, seed_results["rotation_means"][rot_idx] * 1e6,
            yerr=seed_results["rotation_stds"][rot_idx] * 1e6,
            fmt="D", color="black", markersize=6, capsize=5, capthick=1.5, zorder=4
        )
    ax.set_xlabel("Rotation index")
    ax.set_ylabel("Test MSE (×10⁻⁶)")
    ax.set_title(
        f"Q1: Rotation vs Seed Variance (κ={seed_results['kappa']:.0f})\n"
        f"Between-rotation: {seed_results['variance_ratio']:.0%} of total variance",
        fontsize=11
    )
    ax.grid(True, alpha=0.3, axis="y")

    # --- Panel 2: Variance decomposition pie ---
    ax = fig.add_subplot(gs[0, 2])
    between_pct = seed_results["variance_ratio"] * 100
    within_pct = 100 - between_pct
    ax.pie(
        [between_pct, within_pct],
        labels=[f"Rotation\n{between_pct:.0f}%", f"Training seed\n{within_pct:.0f}%"],
        colors=["#E91E63", "#78909C"],
        autopct="", startangle=90, textprops={"fontsize": 11}
    )
    ax.set_title("Variance Decomposition", fontsize=11)

    # --- Panel 3: Summary stats for seed test ---
    ax = fig.add_subplot(gs[0, 3])
    ax.axis("off")
    stats_text = (
        f"Seed Variance Analysis (κ={seed_results['kappa']:.0f})\n"
        f"{'─' * 35}\n"
        f"Rotation CV:     {seed_results['rotation_cv']:.1%}\n"
        f"Mean seed CV:    {seed_results['mean_seed_cv']:.1%}\n"
        f"Between-rot var: {seed_results['between_var']:.2e}\n"
        f"Within-rot var:  {seed_results['within_var']:.2e}\n"
        f"Ratio (rot/total): {seed_results['variance_ratio']:.0%}\n\n"
    )
    if seed_results['variance_ratio'] > 0.5:
        stats_text += "→ Rotation is the dominant\n  source of MSE variance.\n  NOT training noise."
    else:
        stats_text += "→ Training noise is dominant.\n  Rotation effect is small."
    ax.text(0.1, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5", alpha=0.8))

    # --- Row 2: CV comparison across configurations ---
    configs = list(capacity_results.keys())
    spec_names = list(capacity_results[configs[0]].keys())
    n_configs = len(configs)
    n_specs = len(spec_names)

    ax = fig.add_subplot(gs[1, 0:2])
    x = np.arange(n_specs)
    width = 0.8 / n_configs
    for i, cfg_name in enumerate(configs):
        cvs = [capacity_results[cfg_name][s]["cv"] for s in spec_names]
        offset = (i - (n_configs - 1) / 2) * width
        bars = ax.bar(x + offset, cvs, width, color=config_colors[cfg_name],
                      alpha=0.85, label=cfg_name)
        for bar, cv in zip(bars, cvs):
            ax.text(bar.get_x() + bar.get_width() / 2, cv + 0.005,
                    f"{cv:.0%}", ha="center", va="bottom", fontsize=7,
                    fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(spec_names)
    ax.set_ylabel("Coefficient of Variation")
    ax.set_title("Q2+Q3: Rotation CV by Configuration\n(lower = more invariant)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")

    # --- Mean MSE comparison ---
    ax = fig.add_subplot(gs[1, 2:4])
    for i, cfg_name in enumerate(configs):
        means = [capacity_results[cfg_name][s]["mean"] for s in spec_names]
        stds = [capacity_results[cfg_name][s]["std"] for s in spec_names]
        offset = (i - (n_configs - 1) / 2) * width
        ax.bar(x + offset, means, width, yerr=stds, color=config_colors[cfg_name],
               alpha=0.85, label=cfg_name, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(spec_names)
    ax.set_ylabel("Mean Test MSE")
    ax.set_yscale("log")
    ax.set_title("Reconstruction Quality by Configuration\n(error bars = ±1σ across rotations)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")

    # --- Row 3: Per-spectrum scatter for each config ---
    for col, spec_name in enumerate(spec_names):
        ax = fig.add_subplot(gs[2, col])

        for i, cfg_name in enumerate(configs):
            r = capacity_results[cfg_name][spec_name]
            mses = r["mses"]
            jitter = np.random.default_rng(42 + i).uniform(-0.15, 0.15, len(mses))
            ax.scatter(
                np.full_like(mses, i) + jitter, mses,
                color=config_colors[cfg_name], s=50, alpha=0.7,
                edgecolors="white", linewidth=0.5, zorder=3
            )
            ax.errorbar(i, r["mean"], yerr=r["std"], fmt="D",
                        color="black", markersize=5, capsize=5,
                        capthick=1.5, zorder=4)

        ax.set_xticks(range(n_configs))
        ax.set_xticklabels([c.replace(" ", "\n") for c in configs], fontsize=7)
        ax.set_ylabel("Test MSE")
        ax.set_title(f"{spec_name}: Per-Rotation MSE")
        ax.grid(True, alpha=0.3, axis="y")

    # --- Summary table ---
    ax = fig.add_subplot(gs[2, 3])
    ax.axis("off")
    table_data = []
    for cfg_name in configs:
        mean_cv = np.mean([capacity_results[cfg_name][s]["cv"] for s in spec_names])
        mean_mse = np.mean([capacity_results[cfg_name][s]["mean"] for s in spec_names])
        table_data.append([
            cfg_name,
            f"{mean_cv:.1%}",
            f"{mean_mse:.6f}",
        ])
    table = ax.table(
        cellText=table_data,
        colLabels=["Config", "Mean CV", "Mean MSE"],
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Highlight best CV
    cvs_all = [float(row[1].strip('%')) for row in table_data]
    best_idx = np.argmin(cvs_all)
    for col_idx in range(3):
        table[best_idx + 1, col_idx].set_facecolor("#E8F5E9")

    ax.set_title("Overall Summary\n(green = best rotation invariance)", fontsize=10)

    fig.suptitle(
        f"Diagnosing Residual Rotation Sensitivity (dim={dim}, SiLU)\n"
        f"Can wider networks or normalization achieve true rotational invariance?",
        fontsize=14, fontweight="bold",
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved: {output_path}")


def print_summary(seed_results, capacity_results):
    configs = list(capacity_results.keys())
    spec_names = list(capacity_results[configs[0]].keys())

    print("\n" + "=" * 80)
    print("FINAL SUMMARY: Residual Rotation Sensitivity Investigation")
    print("=" * 80)

    print(f"\nQ1: Variance Decomposition (κ={seed_results['kappa']:.0f})")
    print(f"  Between-rotation variance: {seed_results['variance_ratio']:.0%} of total")
    print(f"  Within-rotation (seed) variance: {1 - seed_results['variance_ratio']:.0%} of total")
    if seed_results['variance_ratio'] > 0.5:
        print("  → Rotation is the DOMINANT source of variance (NOT training noise)")
    else:
        print("  → Training noise is dominant; rotation effect is small")

    print(f"\nQ2+Q3: Configuration Comparison")
    header = f"  {'Config':<18}"
    for s in spec_names:
        header += f" | {s:>10} CV  {s:>10} MSE"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for cfg_name in configs:
        row = f"  {cfg_name:<18}"
        for s in spec_names:
            r = capacity_results[cfg_name][s]
            row += f" | {r['cv']:>10.1%}  {r['mean']:>13.6f}"
        print(row)

    # Overall rankings
    print("\n  Overall Rankings (mean across spectra):")
    rankings = []
    for cfg_name in configs:
        mean_cv = np.mean([capacity_results[cfg_name][s]["cv"] for s in spec_names])
        mean_mse = np.mean([capacity_results[cfg_name][s]["mean"] for s in spec_names])
        rankings.append((cfg_name, mean_cv, mean_mse))

    print("    By rotation invariance (lowest CV):")
    for name, cv, mse in sorted(rankings, key=lambda x: x[1]):
        print(f"      {name:<18} CV={cv:.1%}  MSE={mse:.6f}")

    print("    By reconstruction quality (lowest MSE):")
    for name, cv, mse in sorted(rankings, key=lambda x: x[2]):
        print(f"      {name:<18} MSE={mse:.6f}  CV={cv:.1%}")

    # Conclusions
    best_cv_name = min(rankings, key=lambda x: x[1])
    best_mse_name = min(rankings, key=lambda x: x[2])
    baseline_cv = [r for r in rankings if r[0] == "SiLU h=64"][0][1]

    print("\n  CONCLUSIONS:")
    print(f"    Best rotation invariance: {best_cv_name[0]} (CV={best_cv_name[1]:.1%})")
    print(f"    Best reconstruction: {best_mse_name[0]} (MSE={best_mse_name[2]:.6f})")

    wider_cvs = [r for r in rankings if "h=128" in r[0] or "h=256" in r[0]]
    if wider_cvs and all(r[1] < baseline_cv for r in wider_cvs):
        print("    → More capacity DOES reduce rotation sensitivity")
    elif wider_cvs and any(r[1] < baseline_cv for r in wider_cvs):
        print("    → More capacity has MIXED effect on rotation sensitivity")
    else:
        print("    → More capacity does NOT reduce rotation sensitivity")

    ln_cv = [r for r in rankings if "LN" in r[0]]
    if ln_cv and ln_cv[0][1] < baseline_cv:
        print(f"    → LayerNorm REDUCES rotation CV ({ln_cv[0][1]:.1%} vs {baseline_cv:.1%} baseline)")
    elif ln_cv:
        print(f"    → LayerNorm does NOT help ({ln_cv[0][1]:.1%} vs {baseline_cv:.1%} baseline)")


def main():
    os.makedirs("results", exist_ok=True)
    dim = 8

    print("=" * 70)
    print("EXPERIMENT: Diagnosing Residual Rotation Sensitivity")
    print("SiLU gets CV to 8.8% — can we reach ~0%?")
    print("=" * 70)

    # Q1: Seed variance test (4 rotations × 3 seeds = 12 runs)
    seed_results = run_seed_variance_test(
        dim=dim, n_rotations=4, n_seeds=3,
        epochs_recv=100, epochs_emit=200, n_samples=1000
    )

    # Q2+Q3: Capacity and normalization test (4 configs × 2 spectra × 4 rotations = 32 runs)
    capacity_results = run_capacity_test(
        dim=dim, n_rotations=4,
        epochs_recv=100, epochs_emit=200, n_samples=1000
    )

    plot_results(seed_results, capacity_results, dim,
                 "results/obj-019-residual-rotation-sensitivity.png")
    print_summary(seed_results, capacity_results)


if __name__ == "__main__":
    main()
