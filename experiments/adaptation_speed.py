"""Experiment: Adaptation Speed Curve.

How quickly can the Emitter compensate for a channel rotation?

Setup: Pre-train Receiver on channel M₁ (κ=10, dim=8). Then rotate the
channel to M₂ (same spectrum, different U,V). Train a fresh Emitter on M₂
with the frozen M₁-Receiver, checkpointing MSE at several epoch milestones.

This measures the "adaptation speed" — how many epochs of Emitter training
are needed to reach functional communication through a rotated channel.

Comparison:
  - Adaptation curve: Emitter learns to compensate for rotation mismatch
  - Oracle curve: Full pipeline retrained on M₂ (upper bound on quality)
  - Baseline: Pipeline trained and tested on M₁ (no rotation)

Uses SiLU activation (the project default after obj-017/obj-020 findings).
"""

import os
import sys

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from components import Emitter, Receiver, ActionToSignal, Environment, Pipeline
from train import pretrain_receiver, train_emitter
from experiments.pure_rotational_invariance import make_channel_from_svd


def evaluate_pipeline(pipeline, sound_dim, n_test=2000, seed=99):
    """Evaluate pipeline MSE on test sounds."""
    torch.manual_seed(seed)
    sounds = torch.randn(n_test, sound_dim)
    pipeline.eval()
    with torch.no_grad():
        reconstructed = pipeline(sounds)
        mse = (reconstructed - sounds).pow(2).mean().item()
    pipeline.train()
    return mse


def train_emitter_with_checkpoints(pipeline, cfg, checkpoints):
    """Train emitter, recording MSE at specified epoch checkpoints."""
    pipeline.receiver.requires_grad_(False)
    pipeline.action_to_signal.requires_grad_(False)
    pipeline.environment.requires_grad_(False)

    sounds = torch.randn(cfg.emitter_samples, cfg.sound_dim)
    optimizer = torch.optim.Adam(pipeline.emitter.parameters(), lr=cfg.emitter_lr)
    loss_fn = nn.MSELoss()

    checkpoint_mses = {}
    checkpoint_set = set(checkpoints)

    # Record MSE at epoch 0 (before any training)
    if 0 in checkpoint_set:
        checkpoint_mses[0] = evaluate_pipeline(pipeline, cfg.sound_dim)

    for epoch in range(cfg.emitter_epochs):
        perm = torch.randperm(sounds.size(0))
        for i in range(0, sounds.size(0), cfg.emitter_batch_size):
            idx = perm[i:i + cfg.emitter_batch_size]
            batch = sounds[idx]
            decoded = pipeline(batch)
            loss = loss_fn(decoded, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        ep = epoch + 1
        if ep in checkpoint_set:
            mse = evaluate_pipeline(pipeline, cfg.sound_dim)
            checkpoint_mses[ep] = mse
            print(f"    Epoch {ep}: MSE={mse:.6f}")

    return checkpoint_mses


def run_experiment(dim=8, n_rotations=4):
    """Run adaptation speed experiment."""
    print("=" * 60)
    print("EXPERIMENT: Adaptation Speed Curve")
    print("How quickly can the Emitter adapt to a rotated channel?")
    print("=" * 60)

    # Use moderate conditioning where rotation effects are visible
    sigmas = torch.logspace(0, -1, dim)  # κ=10
    kappa = (sigmas[0] / sigmas[-1]).item()
    print(f"Channel: κ={kappa:.0f}, dim={dim}")

    max_epochs = 200
    checkpoints = [0, 5, 10, 20, 50, 100, 150, 200]

    cfg_recv = Config(
        sound_dim=dim, action_dim=dim, signal_dim=dim, hidden_dim=64,
        receiver_lr=1e-3, receiver_epochs=100,
        receiver_samples=1000, receiver_batch_size=64,
        emitter_lr=1e-3, emitter_epochs=max_epochs,
        emitter_samples=1000, emitter_batch_size=64,
        plot_every=9999,
    )

    # Fixed M₁ channel
    seed_u1, seed_v1 = 1000, 2000
    a2s1, env1, M1 = make_channel_from_svd(dim, sigmas, seed_u1, seed_v1)

    # Pre-train Receiver on M₁
    print(f"\nPre-training Receiver on M₁ (100 epochs)...")
    torch.manual_seed(42)
    receiver_m1 = Receiver(cfg_recv.signal_dim, cfg_recv.sound_dim, cfg_recv.hidden_dim)
    pretrain_receiver(a2s1, env1, receiver_m1, cfg_recv)
    receiver_m1.requires_grad_(False)

    # Baseline: train Emitter on M₁ with M₁-Receiver
    print(f"\nBaseline: Emitter on M₁ with M₁-Receiver")
    torch.manual_seed(42)
    emitter_base = Emitter(cfg_recv.sound_dim, cfg_recv.action_dim, cfg_recv.hidden_dim)
    pipeline_base = Pipeline(emitter_base, a2s1, env1, receiver_m1)
    baseline_mses = train_emitter_with_checkpoints(pipeline_base, cfg_recv, checkpoints)

    # For each rotation: adaptation + oracle curves
    adaptation_curves = []
    oracle_curves = []

    for rot_idx in range(n_rotations):
        seed_u2 = 3000 + rot_idx * 11
        seed_v2 = 4000 + rot_idx * 17
        a2s2, env2, M2 = make_channel_from_svd(dim, sigmas, seed_u2, seed_v2)

        print(f"\n--- Rotation {rot_idx+1}/{n_rotations} ---")

        # Adaptation: fresh Emitter on M₂ channel, frozen M₁-Receiver
        print(f"  Adaptation (M₁-Receiver, M₂ channel):")
        torch.manual_seed(42)
        emitter_adapt = Emitter(cfg_recv.sound_dim, cfg_recv.action_dim, cfg_recv.hidden_dim)
        pipeline_adapt = Pipeline(emitter_adapt, a2s2, env2, receiver_m1)
        adapt_mses = train_emitter_with_checkpoints(pipeline_adapt, cfg_recv, checkpoints)
        adaptation_curves.append(adapt_mses)

        # Oracle: full retrain on M₂
        print(f"  Oracle (M₂-Receiver, M₂ channel):")
        torch.manual_seed(42)
        receiver_m2 = Receiver(cfg_recv.signal_dim, cfg_recv.sound_dim, cfg_recv.hidden_dim)
        pretrain_receiver(a2s2, env2, receiver_m2, cfg_recv)
        receiver_m2.requires_grad_(False)

        torch.manual_seed(42)
        emitter_oracle = Emitter(cfg_recv.sound_dim, cfg_recv.action_dim, cfg_recv.hidden_dim)
        pipeline_oracle = Pipeline(emitter_oracle, a2s2, env2, receiver_m2)
        oracle_mses = train_emitter_with_checkpoints(pipeline_oracle, cfg_recv, checkpoints)
        oracle_curves.append(oracle_mses)

    return {
        "checkpoints": checkpoints,
        "baseline": baseline_mses,
        "adaptation_curves": adaptation_curves,
        "oracle_curves": oracle_curves,
        "n_rotations": n_rotations,
        "kappa": kappa,
        "dim": dim,
    }


def plot_results(results, output_path):
    checkpoints = results["checkpoints"]
    baseline = results["baseline"]
    adaptation_curves = results["adaptation_curves"]
    oracle_curves = results["oracle_curves"]
    n_rot = results["n_rotations"]
    kappa = results["kappa"]
    dim = results["dim"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Colors
    baseline_color = "#4CAF50"
    adapt_color = "#E91E63"
    oracle_color = "#2196F3"

    # --- Panel 1: All curves ---
    ax = axes[0]
    base_vals = [baseline[e] for e in checkpoints]
    ax.plot(checkpoints, base_vals, "o-", color=baseline_color, linewidth=2.5,
            markersize=6, label="Baseline (M₁→M₁)", zorder=5)

    for i, (adapt, oracle) in enumerate(zip(adaptation_curves, oracle_curves)):
        adapt_vals = [adapt[e] for e in checkpoints]
        oracle_vals = [oracle[e] for e in checkpoints]
        label_a = "Adaptation (M₁-Recv→M₂)" if i == 0 else None
        label_o = "Oracle (M₂→M₂)" if i == 0 else None
        ax.plot(checkpoints, adapt_vals, "s--", color=adapt_color, linewidth=1.5,
                markersize=4, alpha=0.5, label=label_a)
        ax.plot(checkpoints, oracle_vals, "^:", color=oracle_color, linewidth=1.5,
                markersize=4, alpha=0.5, label=label_o)

    ax.set_xlabel("Emitter Training Epochs")
    ax.set_ylabel("Test MSE")
    ax.set_yscale("log")
    ax.set_title("Adaptation Speed: All Curves")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Mean curves with error bands ---
    ax = axes[1]
    ax.plot(checkpoints, base_vals, "o-", color=baseline_color, linewidth=2.5,
            markersize=6, label="Baseline", zorder=5)

    adapt_matrix = np.array([[c[e] for e in checkpoints] for c in adaptation_curves])
    oracle_matrix = np.array([[c[e] for e in checkpoints] for c in oracle_curves])

    adapt_mean = adapt_matrix.mean(axis=0)
    adapt_std = adapt_matrix.std(axis=0)
    oracle_mean = oracle_matrix.mean(axis=0)
    oracle_std = oracle_matrix.std(axis=0)

    ax.plot(checkpoints, adapt_mean, "s-", color=adapt_color, linewidth=2.5,
            markersize=6, label="Adaptation (mean)")
    ax.fill_between(checkpoints, adapt_mean - adapt_std, adapt_mean + adapt_std,
                     color=adapt_color, alpha=0.15)

    ax.plot(checkpoints, oracle_mean, "^-", color=oracle_color, linewidth=2.5,
            markersize=6, label="Oracle (mean)")
    ax.fill_between(checkpoints, oracle_mean - oracle_std, oracle_mean + oracle_std,
                     color=oracle_color, alpha=0.15)

    ax.set_xlabel("Emitter Training Epochs")
    ax.set_ylabel("Test MSE")
    ax.set_yscale("log")
    ax.set_title("Mean ± 1σ Across Rotations")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panel 3: Adaptation ratio over time ---
    ax = axes[2]
    # Ratio = adaptation MSE / oracle MSE (how much worse is adaptation?)
    ratio_matrix = adapt_matrix / np.maximum(oracle_matrix, 1e-10)
    ratio_mean = ratio_matrix.mean(axis=0)
    ratio_std = ratio_matrix.std(axis=0)

    # Skip epoch 0 where both are untrained
    valid = [i for i, e in enumerate(checkpoints) if e > 0]
    valid_epochs = [checkpoints[i] for i in valid]
    valid_ratio = ratio_mean[valid]
    valid_ratio_std = ratio_std[valid]

    ax.plot(valid_epochs, valid_ratio, "D-", color="#9C27B0", linewidth=2.5,
            markersize=7)
    ax.fill_between(valid_epochs, valid_ratio - valid_ratio_std,
                     valid_ratio + valid_ratio_std, color="#9C27B0", alpha=0.15)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Oracle parity")

    for i, (ep, r) in enumerate(zip(valid_epochs, valid_ratio)):
        ax.annotate(f"{r:.1f}×", (ep, r), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9, fontweight="bold")

    ax.set_xlabel("Emitter Training Epochs")
    ax.set_ylabel("Adaptation / Oracle MSE Ratio")
    ax.set_title("Adaptation Cost Over Training\n(1.0 = oracle-quality)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    fig.suptitle(
        f"Adaptation Speed Curve (dim={dim}, κ={kappa:.0f}, SiLU)\n"
        f"How quickly does the Emitter compensate for a channel rotation?",
        fontsize=13, fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved: {output_path}")


def print_summary(results):
    checkpoints = results["checkpoints"]
    baseline = results["baseline"]
    adaptation_curves = results["adaptation_curves"]
    oracle_curves = results["oracle_curves"]

    adapt_matrix = np.array([[c[e] for e in checkpoints] for c in adaptation_curves])
    oracle_matrix = np.array([[c[e] for e in checkpoints] for c in oracle_curves])

    print("\n" + "=" * 70)
    print("SUMMARY: Adaptation Speed Curve")
    print("=" * 70)

    print(f"\n  {'Epochs':>8} | {'Baseline':>12} | {'Adaptation':>12} | {'Oracle':>12} | {'Adapt/Oracle':>12}")
    print(f"  {'-'*8}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")

    for i, ep in enumerate(checkpoints):
        base = baseline[ep]
        adapt = adapt_matrix[:, i].mean()
        oracle = oracle_matrix[:, i].mean()
        ratio = adapt / max(oracle, 1e-10) if ep > 0 else float('nan')
        print(f"  {ep:>8} | {base:>12.6f} | {adapt:>12.6f} | {oracle:>12.6f} | {ratio:>11.1f}×")

    # Find when adaptation reaches within 2× of oracle
    final_adapt = adapt_matrix[:, -1].mean()
    final_oracle = oracle_matrix[:, -1].mean()
    final_baseline = baseline[checkpoints[-1]]

    print(f"\n  Final adaptation MSE: {final_adapt:.6f}")
    print(f"  Final oracle MSE: {final_oracle:.6f}")
    print(f"  Final baseline MSE: {final_baseline:.6f}")
    print(f"  Adaptation penalty: {final_adapt/max(final_oracle,1e-10):.1f}× oracle")
    print(f"  Adaptation vs baseline: {final_adapt/max(final_baseline,1e-10):.1f}× baseline")


def main():
    os.makedirs("results", exist_ok=True)

    results = run_experiment(dim=8, n_rotations=4)
    plot_results(results, "results/obj-021-adaptation-speed.png")
    print_summary(results)


if __name__ == "__main__":
    main()
