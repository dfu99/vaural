"""Experiment: SiLU + Joint Training — Near-Perfect Rotational Invariance?

Combines the two best strategies from the rotational invariance investigation:
  - obj-017: SiLU activation reduces rotation CV from 19% → 8.8%
  - obj-016: Joint training halves rotation CV at moderate κ (23% → 10.5%)

Question: Does combining both achieve near-perfect rotational invariance?

Design: 4 conditions (2×2 factorial):
  - ReLU × Sequential (baseline from obj-015/016)
  - ReLU × Joint (from obj-016)
  - SiLU × Sequential (from obj-017)
  - SiLU × Joint (NEW — the key test)

8 random rotations × 3 spectra (κ=1, 10, 100) × 4 conditions = 96 trainings.
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
from experiments.joint_training import train_joint
from experiments.pure_rotational_invariance import make_channel_from_svd
from experiments.activation_rotational_invariance import FlexibleMLP


def train_sequential(a2s, env, cfg, activation, seed=42):
    torch.manual_seed(seed)
    receiver = FlexibleMLP(cfg.signal_dim, cfg.sound_dim, cfg.hidden_dim,
                           activation=activation)
    pretrain_receiver(a2s, env, receiver, cfg)
    receiver.requires_grad_(False)
    emitter = FlexibleMLP(cfg.sound_dim, cfg.action_dim, cfg.hidden_dim,
                          activation=activation)
    pipeline = Pipeline(emitter, a2s, env, receiver)
    train_emitter(pipeline, cfg)
    return pipeline


def train_joint_mode(a2s, env, cfg, activation, total_epochs, seed=42):
    torch.manual_seed(seed)
    receiver = FlexibleMLP(cfg.signal_dim, cfg.sound_dim, cfg.hidden_dim,
                           activation=activation)
    emitter = FlexibleMLP(cfg.sound_dim, cfg.action_dim, cfg.hidden_dim,
                          activation=activation)
    pipeline = Pipeline(emitter, a2s, env, receiver)
    train_joint(pipeline, cfg, total_epochs)
    return pipeline


def evaluate(pipeline, dim, n_test=2000, seed=99):
    torch.manual_seed(seed)
    sounds = torch.randn(n_test, dim)
    pipeline.eval()
    with torch.no_grad():
        reconstructed = pipeline(sounds)
        mse = (reconstructed - sounds).pow(2).mean().item()
    return mse


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
    total_epochs = epochs_recv + epochs_emit

    conditions = [
        ("ReLU+Seq", "relu", "sequential"),
        ("ReLU+Joint", "relu", "joint"),
        ("SiLU+Seq", "silu", "sequential"),
        ("SiLU+Joint", "silu", "joint"),
    ]

    spectra = {
        "flat (κ=1)": torch.ones(dim),
        "moderate (κ=10)": torch.logspace(0, -1, dim),
        "steep (κ=100)": torch.logspace(0, -2, dim),
    }

    results = {}

    for cond_name, activation, mode in conditions:
        results[cond_name] = {}
        for spec_name, sigmas in spectra.items():
            kappa = (sigmas[0] / sigmas[-1]).item()
            print(f"\n{'='*60}")
            print(f"{cond_name}, Spectrum: {spec_name}")
            print(f"{'='*60}")

            mses = []
            for rot_idx in range(n_rotations):
                seed_u = 1000 + rot_idx * 7
                seed_v = 2000 + rot_idx * 13

                a2s, env, M = make_channel_from_svd(dim, sigmas, seed_u, seed_v)

                if mode == "sequential":
                    pipeline = train_sequential(a2s, env, cfg, activation)
                else:
                    # Rebuild channel for joint (fresh components)
                    a2s, env, M = make_channel_from_svd(dim, sigmas, seed_u, seed_v)
                    pipeline = train_joint_mode(a2s, env, cfg, activation,
                                               total_epochs)

                mse = evaluate(pipeline, dim)
                mses.append(mse)
                print(f"  Rot {rot_idx+1}/{n_rotations}: MSE={mse:.6f}")

            mses = np.array(mses)
            cv = mses.std() / mses.mean() if mses.mean() > 0 else 0
            results[cond_name][spec_name] = {
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
    cond_names = list(results.keys())
    spec_names = list(results[cond_names[0]].keys())
    n_conds = len(cond_names)
    n_specs = len(spec_names)

    cond_colors = {
        "ReLU+Seq": "#E91E63",
        "ReLU+Joint": "#FF9800",
        "SiLU+Seq": "#4CAF50",
        "SiLU+Joint": "#2196F3",
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # --- Panel 1: CV comparison (key result) ---
    ax = axes[0, 0]
    x = np.arange(n_specs)
    width = 0.8 / n_conds
    for i, cond in enumerate(cond_names):
        cvs = [results[cond][s]["cv"] for s in spec_names]
        offset = (i - (n_conds - 1) / 2) * width
        bars = ax.bar(x + offset, cvs, width, color=cond_colors[cond],
                      alpha=0.8, label=cond)
        for bar, cv in zip(bars, cvs):
            ax.text(bar.get_x() + bar.get_width() / 2, cv + 0.005,
                    f"{cv:.0%}", ha="center", va="bottom", fontsize=6,
                    fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"κ={results[cond_names[0]][s]['kappa']:.0f}"
                        for s in spec_names])
    ax.set_ylabel("Coefficient of Variation")
    ax.set_title("Rotation Sensitivity\n(lower = more invariant)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # --- Panel 2: Mean MSE comparison ---
    ax = axes[0, 1]
    for i, cond in enumerate(cond_names):
        means = [results[cond][s]["mean"] for s in spec_names]
        stds = [results[cond][s]["std"] for s in spec_names]
        offset = (i - (n_conds - 1) / 2) * width
        ax.bar(x + offset, means, width, yerr=stds, color=cond_colors[cond],
               alpha=0.8, label=cond, capsize=2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"κ={results[cond_names[0]][s]['kappa']:.0f}"
                        for s in spec_names])
    ax.set_ylabel("Mean Test MSE")
    ax.set_yscale("log")
    ax.set_title("Reconstruction Quality\n(error bars = ±1σ across rotations)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # --- Panel 3: Interaction plot (activation × training mode) ---
    ax = axes[0, 2]
    for spec_idx, spec_name in enumerate(spec_names):
        kappa = results[cond_names[0]][spec_name]["kappa"]
        # Plot lines connecting seq→joint for each activation
        for act, seq_cond, jnt_cond, color, marker in [
            ("ReLU", "ReLU+Seq", "ReLU+Joint", "#E91E63", "o"),
            ("SiLU", "SiLU+Seq", "SiLU+Joint", "#4CAF50", "s"),
        ]:
            seq_cv = results[seq_cond][spec_name]["cv"]
            jnt_cv = results[jnt_cond][spec_name]["cv"]
            label = f"{act} κ={kappa:.0f}" if spec_idx == 0 or True else None
            ax.plot([0, 1], [seq_cv, jnt_cv], f"{marker}-", color=color,
                    alpha=0.5 + 0.15 * spec_idx, linewidth=1.5, markersize=6,
                    label=f"{act} κ={kappa:.0f}")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Sequential", "Joint"])
    ax.set_ylabel("Rotation CV")
    ax.set_title("Interaction: Activation × Training Mode")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3, axis="y")

    # --- Row 2: Per-spectrum scatter plots ---
    for col, spec_name in enumerate(spec_names):
        ax = axes[1, col]
        kappa = results[cond_names[0]][spec_name]["kappa"]

        for i, cond in enumerate(cond_names):
            r = results[cond][spec_name]
            mses = r["mses"]
            jitter = np.random.default_rng(42 + i).uniform(-0.15, 0.15,
                                                            len(mses))
            ax.scatter(np.full_like(mses, i) + jitter, mses,
                       color=cond_colors[cond], s=50, alpha=0.7,
                       edgecolors="white", linewidth=0.5, zorder=3)
            ax.errorbar(i, r["mean"], yerr=r["std"], fmt="D",
                        color="black", markersize=5, capsize=5,
                        capthick=1.5, zorder=4)

        ax.set_xticks(range(n_conds))
        ax.set_xticklabels([c.replace("+", "\n+") for c in cond_names],
                           fontsize=7)
        ax.set_ylabel("Test MSE")
        ax.set_title(f"κ={kappa:.0f}: Per-Rotation MSE")
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"SiLU + Joint Training: Near-Perfect Rotational Invariance? (dim={dim})\n"
        f"2×2 factorial: (ReLU vs SiLU) × (Sequential vs Joint)",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved: {output_path}")


def print_summary(results):
    cond_names = list(results.keys())
    spec_names = list(results[cond_names[0]].keys())

    print("\n" + "=" * 80)
    print("SUMMARY: 2×2 Factorial — Activation × Training Mode")
    print("=" * 80)

    header = f"  {'Condition':<14}"
    for s in spec_names:
        kappa = results[cond_names[0]][s]["kappa"]
        header += f" | {'κ=' + str(int(kappa)) + ' MSE':>12} {'CV':>6}"
    header += " | {'Mean CV':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for cond in cond_names:
        row = f"  {cond:<14}"
        cvs = []
        for s in spec_names:
            r = results[cond][s]
            row += f" | {r['mean']:>12.6f} {r['cv']:>5.1%}"
            cvs.append(r['cv'])
        row += f" | {np.mean(cvs):>7.1%}"
        print(row)

    # Find best combination
    print("\n  Ranking by mean CV (rotation invariance):")
    mean_cvs = {}
    for cond in cond_names:
        cvs = [results[cond][s]["cv"] for s in spec_names]
        mean_cvs[cond] = np.mean(cvs)
    for cond in sorted(mean_cvs, key=mean_cvs.get):
        print(f"    {cond}: mean CV = {mean_cvs[cond]:.1%}")

    print("\n  Ranking by mean MSE (reconstruction quality):")
    mean_mses = {}
    for cond in cond_names:
        mses = [results[cond][s]["mean"] for s in spec_names]
        mean_mses[cond] = np.mean(mses)
    for cond in sorted(mean_mses, key=mean_mses.get):
        print(f"    {cond}: mean MSE = {mean_mses[cond]:.6f}")

    # Test: does SiLU+Joint beat both individual improvements?
    sj_cv = mean_cvs.get("SiLU+Joint", float("inf"))
    ss_cv = mean_cvs.get("SiLU+Seq", float("inf"))
    rj_cv = mean_cvs.get("ReLU+Joint", float("inf"))
    rs_cv = mean_cvs.get("ReLU+Seq", float("inf"))

    print(f"\n  INTERACTION TEST:")
    print(f"    ReLU+Seq (baseline):  CV = {rs_cv:.1%}")
    print(f"    SiLU alone:           CV = {ss_cv:.1%} ({(1-ss_cv/rs_cv)*100:+.0f}% vs baseline)")
    print(f"    Joint alone:          CV = {rj_cv:.1%} ({(1-rj_cv/rs_cv)*100:+.0f}% vs baseline)")
    print(f"    SiLU+Joint combined:  CV = {sj_cv:.1%} ({(1-sj_cv/rs_cv)*100:+.0f}% vs baseline)")

    if sj_cv < min(ss_cv, rj_cv):
        print("    → SYNERGISTIC: Combined is better than either alone")
    elif sj_cv < max(ss_cv, rj_cv):
        print("    → PARTIAL: Combined beats one but not the other")
    else:
        print("    → NO SYNERGY: Combined is not better than the best individual")


def main():
    os.makedirs("results", exist_ok=True)

    print("=" * 60)
    print("EXPERIMENT: SiLU + Joint Training × Rotational Invariance")
    print("2×2 factorial: (ReLU vs SiLU) × (Sequential vs Joint)")
    print("=" * 60)

    results, dim = run_experiment(
        dim=8, n_rotations=8, epochs_recv=400, epochs_emit=500, n_samples=2000
    )

    plot_results(results, dim,
                 "results/obj-018-silu-joint-rotation.png")
    print_summary(results)


if __name__ == "__main__":
    main()
