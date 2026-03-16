"""Experiment: Accent Accommodation — Can Receiver Fine-Tuning Close the Gap?

obj-021 showed the Emitter adapts to a rotated channel within ~50 epochs but
hits a 2.3× "accent" penalty vs oracle. The penalty exists because the
M₁-Receiver has channel-specific biases baked in during pre-training.

This experiment tests whether fine-tuning the Receiver on the new channel
can close that gap. Three strategies:

  1. Emitter-only adaptation (obj-021 baseline): Train fresh Emitter, frozen Receiver
  2. Joint fine-tuning: Unfreeze Receiver during Emitter training on M₂
  3. Sequential fine-tuning: First fine-tune Receiver on M₂ (with identity mapping),
     then train fresh Emitter

The question: does "accent accommodation" (Receiver adapting to a new speaker)
work, and which strategy is best?

Biological analogy: (1) = speaker adapts to listener's expectations (accent
modification), (2) = both adapt simultaneously (conversation convergence),
(3) = listener first adapts to the new accent, then speaker refines.
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


def evaluate(pipeline, sound_dim, n_test=2000, seed=99):
    torch.manual_seed(seed)
    sounds = torch.randn(n_test, sound_dim)
    pipeline.eval()
    with torch.no_grad():
        mse = (pipeline(sounds) - sounds).pow(2).mean().item()
    pipeline.train()
    return mse


def fine_tune_receiver(a2s, env, receiver, cfg, epochs=100):
    """Fine-tune an existing Receiver on a new channel (identity mapping)."""
    receiver.requires_grad_(True)
    sounds = torch.randn(cfg.receiver_samples, cfg.sound_dim)
    with torch.no_grad():
        signals = a2s(sounds)
        received = env(signals)

    optimizer = torch.optim.Adam(receiver.parameters(), lr=cfg.receiver_lr)
    loss_fn = nn.MSELoss()
    losses = []

    for epoch in range(epochs):
        perm = torch.randperm(sounds.size(0))
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, sounds.size(0), cfg.receiver_batch_size):
            idx = perm[i:i + cfg.receiver_batch_size]
            decoded = receiver(received[idx])
            loss = loss_fn(decoded, sounds[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        losses.append(epoch_loss / n_batches)

    return losses


def train_joint(pipeline, cfg, epochs=200):
    """Joint training: both Emitter and Receiver update together."""
    pipeline.receiver.requires_grad_(True)
    pipeline.action_to_signal.requires_grad_(False)
    pipeline.environment.requires_grad_(False)

    sounds = torch.randn(cfg.emitter_samples, cfg.sound_dim)
    params = list(pipeline.emitter.parameters()) + list(pipeline.receiver.parameters())
    optimizer = torch.optim.Adam(params, lr=cfg.emitter_lr)
    loss_fn = nn.MSELoss()
    losses = []

    for epoch in range(epochs):
        perm = torch.randperm(sounds.size(0))
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, sounds.size(0), cfg.emitter_batch_size):
            idx = perm[i:i + cfg.emitter_batch_size]
            decoded = pipeline(sounds[idx])
            loss = loss_fn(decoded, sounds[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        losses.append(epoch_loss / n_batches)

    return losses


def run_experiment(dim=8, n_rotations=4, recv_epochs=100, emit_epochs=200,
                   n_samples=1000):
    print("=" * 60)
    print("EXPERIMENT: Accent Accommodation")
    print("Can Receiver fine-tuning close the 2.3× adaptation gap?")
    print("=" * 60)

    sigmas = torch.logspace(0, -1, dim)  # κ=10
    kappa = (sigmas[0] / sigmas[-1]).item()

    cfg = Config(
        sound_dim=dim, action_dim=dim, signal_dim=dim, hidden_dim=64,
        receiver_lr=1e-3, receiver_epochs=recv_epochs,
        receiver_samples=n_samples, receiver_batch_size=64,
        emitter_lr=1e-3, emitter_epochs=emit_epochs,
        emitter_samples=n_samples, emitter_batch_size=64,
        plot_every=9999,
    )

    # Fixed M₁ channel — pre-train Receiver
    seed_u1, seed_v1 = 1000, 2000
    a2s1, env1, M1 = make_channel_from_svd(dim, sigmas, seed_u1, seed_v1)

    print(f"\nPre-training Receiver on M₁ (κ={kappa:.0f})...")
    torch.manual_seed(42)
    receiver_m1 = Receiver(cfg.signal_dim, cfg.sound_dim, cfg.hidden_dim)
    pretrain_receiver(a2s1, env1, receiver_m1, cfg)

    # Baseline: M₁ pipeline
    receiver_m1_frozen = Receiver(cfg.signal_dim, cfg.sound_dim, cfg.hidden_dim)
    receiver_m1_frozen.load_state_dict(receiver_m1.state_dict())
    receiver_m1_frozen.requires_grad_(False)

    torch.manual_seed(42)
    emitter_base = Emitter(cfg.sound_dim, cfg.action_dim, cfg.hidden_dim)
    pipeline_base = Pipeline(emitter_base, a2s1, env1, receiver_m1_frozen)
    train_emitter(pipeline_base, cfg)
    baseline_mse = evaluate(pipeline_base, cfg.sound_dim)
    print(f"\nBaseline (M₁→M₁): MSE={baseline_mse:.6f}")

    results_per_rotation = []

    for rot_idx in range(n_rotations):
        seed_u2 = 3000 + rot_idx * 11
        seed_v2 = 4000 + rot_idx * 17
        a2s2, env2, M2 = make_channel_from_svd(dim, sigmas, seed_u2, seed_v2)

        print(f"\n--- Rotation {rot_idx+1}/{n_rotations} ---")
        rot_results = {}

        # Strategy 1: Emitter-only (frozen M₁-Receiver)
        print("  Strategy 1: Emitter-only adaptation")
        recv1 = Receiver(cfg.signal_dim, cfg.sound_dim, cfg.hidden_dim)
        recv1.load_state_dict(receiver_m1.state_dict())
        recv1.requires_grad_(False)
        torch.manual_seed(42)
        emit1 = Emitter(cfg.sound_dim, cfg.action_dim, cfg.hidden_dim)
        pipe1 = Pipeline(emit1, a2s2, env2, recv1)
        train_emitter(pipe1, cfg)
        rot_results["emitter_only"] = evaluate(pipe1, cfg.sound_dim)
        print(f"    MSE={rot_results['emitter_only']:.6f}")

        # Strategy 2: Joint fine-tuning (both adapt on M₂)
        print("  Strategy 2: Joint fine-tuning")
        recv2 = Receiver(cfg.signal_dim, cfg.sound_dim, cfg.hidden_dim)
        recv2.load_state_dict(receiver_m1.state_dict())
        torch.manual_seed(42)
        emit2 = Emitter(cfg.sound_dim, cfg.action_dim, cfg.hidden_dim)
        pipe2 = Pipeline(emit2, a2s2, env2, recv2)
        train_joint(pipe2, cfg, epochs=emit_epochs)
        rot_results["joint"] = evaluate(pipe2, cfg.sound_dim)
        print(f"    MSE={rot_results['joint']:.6f}")

        # Strategy 3: Sequential fine-tuning (Receiver first, then Emitter)
        print("  Strategy 3: Sequential fine-tuning")
        recv3 = Receiver(cfg.signal_dim, cfg.sound_dim, cfg.hidden_dim)
        recv3.load_state_dict(receiver_m1.state_dict())
        fine_tune_receiver(a2s2, env2, recv3, cfg, epochs=recv_epochs)
        recv3.requires_grad_(False)
        torch.manual_seed(42)
        emit3 = Emitter(cfg.sound_dim, cfg.action_dim, cfg.hidden_dim)
        pipe3 = Pipeline(emit3, a2s2, env2, recv3)
        train_emitter(pipe3, cfg)
        rot_results["sequential_ft"] = evaluate(pipe3, cfg.sound_dim)
        print(f"    MSE={rot_results['sequential_ft']:.6f}")

        # Oracle: full retrain on M₂
        print("  Oracle: full retrain")
        torch.manual_seed(42)
        recv_o = Receiver(cfg.signal_dim, cfg.sound_dim, cfg.hidden_dim)
        pretrain_receiver(a2s2, env2, recv_o, cfg)
        recv_o.requires_grad_(False)
        torch.manual_seed(42)
        emit_o = Emitter(cfg.sound_dim, cfg.action_dim, cfg.hidden_dim)
        pipe_o = Pipeline(emit_o, a2s2, env2, recv_o)
        train_emitter(pipe_o, cfg)
        rot_results["oracle"] = evaluate(pipe_o, cfg.sound_dim)
        print(f"    MSE={rot_results['oracle']:.6f}")

        results_per_rotation.append(rot_results)

    return {
        "baseline_mse": baseline_mse,
        "rotations": results_per_rotation,
        "n_rotations": n_rotations,
        "kappa": kappa,
        "dim": dim,
    }


def plot_results(results, output_path):
    n_rot = results["n_rotations"]
    baseline = results["baseline_mse"]

    strategies = ["emitter_only", "joint", "sequential_ft", "oracle"]
    labels = ["Emitter-only\n(frozen Recv)", "Joint\nfine-tuning",
              "Sequential\nfine-tuning", "Oracle\n(full retrain)"]
    colors = ["#E91E63", "#FF9800", "#4CAF50", "#2196F3"]

    # Collect data
    data = {s: [r[s] for r in results["rotations"]] for s in strategies}
    means = {s: np.mean(data[s]) for s in strategies}
    stds = {s: np.std(data[s]) for s in strategies}

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # --- Panel 1: MSE comparison ---
    ax = axes[0]
    x = np.arange(len(strategies))
    bars = ax.bar(x, [means[s] for s in strategies], yerr=[stds[s] for s in strategies],
                  color=colors, alpha=0.85, capsize=6)
    ax.axhline(y=baseline, color="gray", linestyle="--", alpha=0.7, label=f"Baseline (M₁→M₁)")
    for bar, s in zip(bars, strategies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + stds[s] + 0.00002,
                f"{means[s]:.5f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Test MSE")
    ax.set_title("Adaptation Strategy Comparison")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # --- Panel 2: Ratio to oracle ---
    ax = axes[1]
    oracle_mean = means["oracle"]
    ratios = {s: np.array(data[s]) / np.maximum(np.array(data["oracle"]), 1e-10)
              for s in strategies}
    ratio_means = {s: ratios[s].mean() for s in strategies}
    ratio_stds = {s: ratios[s].std() for s in strategies}

    bars = ax.bar(x, [ratio_means[s] for s in strategies],
                  yerr=[ratio_stds[s] for s in strategies],
                  color=colors, alpha=0.85, capsize=6)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.7, label="Oracle parity")
    baseline_ratio = baseline / oracle_mean
    ax.axhline(y=baseline_ratio, color="gray", linestyle=":", alpha=0.5,
               label=f"Baseline ratio ({baseline_ratio:.1f}×)")
    for bar, s in zip(bars, strategies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + ratio_stds[s] + 0.05,
                f"{ratio_means[s]:.2f}×", ha="center", va="bottom", fontsize=10,
                fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("MSE / Oracle MSE")
    ax.set_title("Adaptation Cost (Ratio to Oracle)\n(1.0 = perfect)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # --- Panel 3: Per-rotation scatter ---
    ax = axes[2]
    for i, s in enumerate(strategies):
        mses = np.array(data[s])
        jitter = np.random.default_rng(42 + i).uniform(-0.15, 0.15, len(mses))
        ax.scatter(np.full_like(mses, i) + jitter, mses, color=colors[i],
                   s=70, alpha=0.7, edgecolors="white", linewidth=0.5, zorder=3)
        ax.errorbar(i, means[s], yerr=stds[s], fmt="D", color="black",
                    markersize=6, capsize=6, capthick=1.5, zorder=4)
    ax.axhline(y=baseline, color="gray", linestyle="--", alpha=0.5)
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Test MSE")
    ax.set_title("Per-Rotation Results")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"Accent Accommodation (dim={results['dim']}, κ={results['kappa']:.0f}, SiLU)\n"
        f"Can Receiver fine-tuning close the adaptation gap?",
        fontsize=13, fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved: {output_path}")


def print_summary(results):
    baseline = results["baseline_mse"]
    strategies = ["emitter_only", "joint", "sequential_ft", "oracle"]
    labels = ["Emitter-only", "Joint FT", "Sequential FT", "Oracle"]
    data = {s: [r[s] for r in results["rotations"]] for s in strategies}

    print("\n" + "=" * 70)
    print("SUMMARY: Accent Accommodation")
    print("=" * 70)
    print(f"  Baseline (M₁→M₁): MSE={baseline:.6f}")
    print()

    print(f"  {'Strategy':<18} | {'Mean MSE':>12} | {'Std':>10} | {'vs Oracle':>10} | {'vs Baseline':>12}")
    print(f"  {'-'*18}-+-{'-'*12}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}")

    oracle_mean = np.mean(data["oracle"])
    for s, label in zip(strategies, labels):
        mean = np.mean(data[s])
        std = np.std(data[s])
        vs_oracle = mean / max(oracle_mean, 1e-10)
        vs_baseline = mean / max(baseline, 1e-10)
        print(f"  {label:<18} | {mean:>12.6f} | {std:>10.6f} | {vs_oracle:>9.2f}× | {vs_baseline:>11.2f}×")

    # Determine winner
    non_oracle = ["emitter_only", "joint", "sequential_ft"]
    best = min(non_oracle, key=lambda s: np.mean(data[s]))
    best_label = {"emitter_only": "Emitter-only", "joint": "Joint FT",
                  "sequential_ft": "Sequential FT"}[best]
    best_ratio = np.mean(data[best]) / max(oracle_mean, 1e-10)

    print(f"\n  BEST ADAPTATION STRATEGY: {best_label} ({best_ratio:.2f}× oracle)")

    seq_ft_ratio = np.mean(data["sequential_ft"]) / max(oracle_mean, 1e-10)
    emit_ratio = np.mean(data["emitter_only"]) / max(oracle_mean, 1e-10)
    gap_closed = (emit_ratio - seq_ft_ratio) / (emit_ratio - 1.0) * 100 if emit_ratio > 1.0 else 0

    print(f"  Gap closed by Receiver fine-tuning: {gap_closed:.0f}%")
    print(f"    Emitter-only: {emit_ratio:.2f}× oracle")
    print(f"    Sequential FT: {seq_ft_ratio:.2f}× oracle")
    print(f"    Remaining gap: {seq_ft_ratio - 1.0:.2f}× above oracle")


def main():
    os.makedirs("results", exist_ok=True)
    results = run_experiment(
        dim=8, n_rotations=4, recv_epochs=100, emit_epochs=200, n_samples=1000
    )
    plot_results(results, "results/obj-022-accent-accommodation.png")
    print_summary(results)


if __name__ == "__main__":
    main()
