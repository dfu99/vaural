"""Experiment: Does Joint Training Reduce Rotation Sensitivity?

Combines findings from:
  obj-013: Channel κ dominates sequential training
  obj-014: Joint training reduces channel sensitivity by 94%
  obj-015: System is not rotationally invariant (CV 13-23% across rotations)

Key question: Does joint training reduce the ROTATION-specific variance
that obj-015 revealed? If so, joint training doesn't just help with the
spectrum — it makes the system more truly rotation-invariant.

Design: At κ=10 (where obj-015 found highest CV=23.4%):
  - 8 random rotation pairs (U, V), same spectrum
  - Sequential vs joint training for each
  - Compare CV across rotations for each training mode
"""

import os
import sys

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from components import Emitter, ActionToSignal, Environment, Receiver, Pipeline
from train import pretrain_receiver, train_emitter
from experiments.joint_training import train_joint
from experiments.pure_rotational_invariance import make_orthogonal, make_channel_from_svd


def train_sequential(a2s, env, cfg, seed=42):
    torch.manual_seed(seed)
    receiver = Receiver(cfg.signal_dim, cfg.sound_dim, cfg.hidden_dim)
    recv_losses = pretrain_receiver(a2s, env, receiver, cfg)
    receiver.requires_grad_(False)
    emitter = Emitter(cfg.sound_dim, cfg.action_dim, cfg.hidden_dim)
    pipeline = Pipeline(emitter, a2s, env, receiver)
    emit_losses = train_emitter(pipeline, cfg)
    return pipeline


def train_joint_mode(a2s, env, cfg, total_epochs, seed=42):
    torch.manual_seed(seed)
    receiver = Receiver(cfg.signal_dim, cfg.sound_dim, cfg.hidden_dim)
    emitter = Emitter(cfg.sound_dim, cfg.action_dim, cfg.hidden_dim)
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

    spectra = {
        "flat (κ=1)": torch.ones(dim),
        "moderate (κ=10)": torch.logspace(0, -1, dim),
        "steep (κ=100)": torch.logspace(0, -2, dim),
    }

    results = {}
    for spec_name, sigmas in spectra.items():
        kappa = (sigmas[0] / sigmas[-1]).item()
        print(f"\n{'='*60}")
        print(f"Spectrum: {spec_name}")
        print(f"{'='*60}")

        seq_mses = []
        jnt_mses = []

        for rot_idx in range(n_rotations):
            seed_u = 1000 + rot_idx * 7
            seed_v = 2000 + rot_idx * 13

            print(f"\n  Rotation {rot_idx+1}/{n_rotations}")

            # Build channel
            a2s, env, M = make_channel_from_svd(dim, sigmas, seed_u, seed_v)

            # Sequential
            print(f"    Sequential...", end=" ", flush=True)
            pipeline_seq = train_sequential(a2s, env, cfg)
            mse_seq = evaluate(pipeline_seq, dim)
            seq_mses.append(mse_seq)
            print(f"MSE={mse_seq:.6f}")

            # Rebuild channel (fresh components)
            a2s, env, M = make_channel_from_svd(dim, sigmas, seed_u, seed_v)

            # Joint
            print(f"    Joint...", end=" ", flush=True)
            pipeline_jnt = train_joint_mode(a2s, env, cfg, total_epochs)
            mse_jnt = evaluate(pipeline_jnt, dim)
            jnt_mses.append(mse_jnt)
            print(f"MSE={mse_jnt:.6f}")

        seq_mses = np.array(seq_mses)
        jnt_mses = np.array(jnt_mses)

        results[spec_name] = {
            "kappa": kappa,
            "seq_mses": seq_mses,
            "jnt_mses": jnt_mses,
            "seq_mean": seq_mses.mean(),
            "seq_std": seq_mses.std(),
            "seq_cv": seq_mses.std() / seq_mses.mean() if seq_mses.mean() > 0 else 0,
            "jnt_mean": jnt_mses.mean(),
            "jnt_std": jnt_mses.std(),
            "jnt_cv": jnt_mses.std() / jnt_mses.mean() if jnt_mses.mean() > 0 else 0,
        }

        print(f"\n  Sequential: {seq_mses.mean():.6f} ± {seq_mses.std():.6f} (CV={results[spec_name]['seq_cv']:.1%})")
        print(f"  Joint:      {jnt_mses.mean():.6f} ± {jnt_mses.std():.6f} (CV={results[spec_name]['jnt_cv']:.1%})")

    return results, dim


def plot_results(results, dim, output_path):
    spec_names = list(results.keys())
    n_specs = len(spec_names)
    colors_seq = "#2196F3"
    colors_jnt = "#E91E63"

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # --- Panel 1: Paired MSE comparison across rotations (κ=10) ---
    ax = axes[0, 0]
    # Pick the middle spectrum for detailed view
    mid_key = spec_names[1]
    r = results[mid_key]
    x = np.arange(len(r["seq_mses"]))
    width = 0.35
    ax.bar(x - width/2, r["seq_mses"], width, color=colors_seq, alpha=0.8, label="Sequential")
    ax.bar(x + width/2, r["jnt_mses"], width, color=colors_jnt, alpha=0.8, label="Joint")
    ax.set_xlabel("Rotation index")
    ax.set_ylabel("Test MSE")
    ax.set_title(f"Per-Rotation MSE: Sequential vs Joint ({mid_key})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # --- Panel 2: CV comparison (the key result) ---
    ax = axes[0, 1]
    x = np.arange(n_specs)
    width = 0.35
    seq_cvs = [results[k]["seq_cv"] for k in spec_names]
    jnt_cvs = [results[k]["jnt_cv"] for k in spec_names]
    bars_s = ax.bar(x - width/2, seq_cvs, width, color=colors_seq, alpha=0.8, label="Sequential")
    bars_j = ax.bar(x + width/2, jnt_cvs, width, color=colors_jnt, alpha=0.8, label="Joint")
    ax.set_xticks(x)
    ax.set_xticklabels([f"κ={results[k]['kappa']:.0f}" for k in spec_names])
    ax.set_ylabel("Coefficient of Variation")
    ax.set_title("Rotation Sensitivity: Sequential vs Joint\n(lower = more rotationally invariant)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    for bar, cv in zip(list(bars_s) + list(bars_j), seq_cvs + jnt_cvs):
        ax.text(bar.get_x() + bar.get_width()/2, cv + max(seq_cvs + jnt_cvs) * 0.02,
                f"{cv:.1%}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # --- Panel 3: Mean MSE comparison ---
    ax = axes[1, 0]
    seq_means = [results[k]["seq_mean"] for k in spec_names]
    jnt_means = [results[k]["jnt_mean"] for k in spec_names]
    seq_stds = [results[k]["seq_std"] for k in spec_names]
    jnt_stds = [results[k]["jnt_std"] for k in spec_names]

    x = np.arange(n_specs)
    ax.bar(x - width/2, seq_means, width, yerr=seq_stds, color=colors_seq, alpha=0.8,
           label="Sequential", capsize=4)
    ax.bar(x + width/2, jnt_means, width, yerr=jnt_stds, color=colors_jnt, alpha=0.8,
           label="Joint", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels([f"κ={results[k]['kappa']:.0f}" for k in spec_names])
    ax.set_ylabel("Mean Test MSE")
    ax.set_yscale("log")
    ax.set_title("Mean MSE with Rotation Variance\n(error bars = ±1σ across rotations)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # --- Panel 4: Summary table ---
    ax = axes[1, 1]
    ax.axis("off")
    table_data = []
    for k in spec_names:
        r = results[k]
        improvement = r["seq_mean"] / max(r["jnt_mean"], 1e-10)
        cv_reduction = (1 - r["jnt_cv"] / max(r["seq_cv"], 1e-10)) * 100 if r["seq_cv"] > 0 else 0
        table_data.append([
            f"κ={r['kappa']:.0f}",
            f"{r['seq_mean']:.6f}",
            f"{r['seq_cv']:.1%}",
            f"{r['jnt_mean']:.6f}",
            f"{r['jnt_cv']:.1%}",
            f"{improvement:.1f}×",
            f"{cv_reduction:+.0f}%",
        ])
    table = ax.table(
        cellText=table_data,
        colLabels=["κ", "Seq MSE", "Seq CV", "Joint MSE", "Joint CV", "MSE Gain", "CV Change"],
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    ax.set_title("Summary: Joint Training Effect on Rotation Sensitivity",
                 fontsize=11, fontweight="bold", pad=20)

    fig.suptitle(
        f"Does Joint Training Reduce Rotation Sensitivity? (dim={dim})\n"
        f"Fixed spectrum, varying rotations — sequential vs joint",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved: {output_path}")


def main():
    os.makedirs("results", exist_ok=True)

    print("=" * 60)
    print("EXPERIMENT: Joint Training vs Rotation Sensitivity")
    print("=" * 60)

    results, dim = run_experiment(
        dim=8, n_rotations=8, epochs_recv=400, epochs_emit=500, n_samples=2000
    )

    plot_results(results, dim, "results/obj-016-rotation-sensitivity-joint-vs-seq.png")

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    for name, r in results.items():
        improvement = r["seq_mean"] / max(r["jnt_mean"], 1e-10)
        print(f"\n  {name}:")
        print(f"    Sequential: MSE={r['seq_mean']:.6f} ± {r['seq_std']:.6f} (CV={r['seq_cv']:.1%})")
        print(f"    Joint:      MSE={r['jnt_mean']:.6f} ± {r['jnt_std']:.6f} (CV={r['jnt_cv']:.1%})")
        print(f"    MSE improvement: {improvement:.1f}×")
        if r["seq_cv"] > 0:
            cv_change = (r["jnt_cv"] / r["seq_cv"] - 1) * 100
            print(f"    CV change: {cv_change:+.0f}%")


if __name__ == "__main__":
    main()
