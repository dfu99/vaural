"""Experiment: Joint vs Sequential training.

Tests the sensorimotor coupling hypothesis by comparing:
1. Sequential (baseline): Pre-train Receiver, freeze it, then train Emitter
2. Joint: Train Emitter and Receiver simultaneously from scratch

Both use the same hyperparameters, data, and random seeds for fair comparison.
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


def train_joint(
    pipeline: Pipeline,
    cfg: Config,
    total_epochs: int,
) -> dict:
    """Train Emitter and Receiver jointly from scratch.

    Both modules' parameters are updated simultaneously using the
    end-to-end reconstruction loss.
    """
    # Both Emitter and Receiver are trainable
    params = list(pipeline.emitter.parameters()) + list(pipeline.receiver.parameters())
    optimizer = torch.optim.Adam(params, lr=cfg.emitter_lr)
    loss_fn = nn.MSELoss()

    sounds = torch.randn(cfg.emitter_samples, cfg.sound_dim)

    losses = []
    for epoch in range(total_epochs):
        perm = torch.randperm(sounds.size(0))
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, sounds.size(0), cfg.emitter_batch_size):
            idx = perm[i : i + cfg.emitter_batch_size]
            batch = sounds[idx]

            decoded = pipeline(batch)
            loss = loss_fn(decoded, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        if (epoch + 1) % 100 == 0:
            print(f"  Joint epoch {epoch + 1}/{total_epochs}, loss: {avg_loss:.6f}")

    return {"losses": losses}


def run_sequential(cfg: Config, seed: int) -> dict:
    """Run the standard sequential training pipeline."""
    torch.manual_seed(seed)

    action_to_signal = ActionToSignal(cfg.action_dim, cfg.signal_dim, seed=100)
    environment = Environment(cfg.signal_dim, seed=200)

    # Phase 1: Pre-train Receiver
    receiver = Receiver(cfg.signal_dim, cfg.sound_dim, cfg.hidden_dim)
    recv_losses = pretrain_receiver(action_to_signal, environment, receiver, cfg)

    # Freeze
    receiver.requires_grad_(False)
    receiver.eval()

    # Phase 2: Train Emitter
    emitter = Emitter(cfg.sound_dim, cfg.action_dim, cfg.hidden_dim)
    pipeline = Pipeline(emitter, action_to_signal, environment, receiver)
    emit_losses = train_emitter(pipeline, cfg)

    # Evaluate
    pipeline.eval()
    test_sounds = torch.randn(500, cfg.sound_dim)
    with torch.no_grad():
        decoded = pipeline(test_sounds)
        test_mse = nn.functional.mse_loss(decoded, test_sounds).item()

    return {
        "recv_losses": recv_losses,
        "emit_losses": emit_losses,
        "test_mse": test_mse,
        "pipeline": pipeline,
    }


def run_joint(cfg: Config, seed: int, total_epochs: int) -> dict:
    """Run joint training from scratch."""
    torch.manual_seed(seed)

    action_to_signal = ActionToSignal(cfg.action_dim, cfg.signal_dim, seed=100)
    environment = Environment(cfg.signal_dim, seed=200)
    receiver = Receiver(cfg.signal_dim, cfg.sound_dim, cfg.hidden_dim)
    emitter = Emitter(cfg.sound_dim, cfg.action_dim, cfg.hidden_dim)
    pipeline = Pipeline(emitter, action_to_signal, environment, receiver)

    result = train_joint(pipeline, cfg, total_epochs)

    # Evaluate
    pipeline.eval()
    test_sounds = torch.randn(500, cfg.sound_dim)
    with torch.no_grad():
        decoded = pipeline(test_sounds)
        test_mse = nn.functional.mse_loss(decoded, test_sounds).item()

    result["test_mse"] = test_mse
    result["pipeline"] = pipeline
    return result


def plot_comparison(seq_result, joint_result, total_epochs, output_path):
    """Plot sequential vs joint training comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: Loss curves over total epochs
    ax = axes[0]
    # Sequential: concat receiver + emitter losses
    seq_all = seq_result["recv_losses"] + seq_result["emit_losses"]
    joint_all = joint_result["losses"]

    ax.plot(range(1, len(seq_all) + 1), seq_all, label="Sequential", linewidth=1.5, color="#2196F3")
    ax.plot(range(1, len(joint_all) + 1), joint_all, label="Joint", linewidth=1.5, color="#E91E63")
    ax.set_yscale("log")
    ax.set_xlabel("Total Epochs")
    ax.set_ylabel("MSE Loss (log)")
    ax.set_title("Training Loss Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Vertical line at phase boundary for sequential
    n_recv = len(seq_result["recv_losses"])
    ax.axvline(x=n_recv, color="#2196F3", linestyle="--", alpha=0.5, label=f"Recv→Emit ({n_recv})")

    # Panel 2: Final MSE bar chart
    ax = axes[1]
    methods = ["Sequential", "Joint"]
    mses = [seq_result["test_mse"], joint_result["test_mse"]]
    colors = ["#2196F3", "#E91E63"]
    bars = ax.bar(methods, mses, color=colors, alpha=0.7, edgecolor=colors, linewidth=2)
    ax.set_ylabel("Test MSE")
    ax.set_title("Final Test MSE")
    ax.set_yscale("log")
    for bar, mse in zip(bars, mses):
        ax.text(bar.get_x() + bar.get_width() / 2, mse * 1.3, f"{mse:.6f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 3: Convergence speed (epochs to reach various thresholds)
    ax = axes[2]
    thresholds = [1.0, 0.1, 0.01, 0.001]
    seq_epochs_to = []
    joint_epochs_to = []
    for t in thresholds:
        seq_ep = next((i + 1 for i, l in enumerate(seq_all) if l < t), len(seq_all))
        joint_ep = next((i + 1 for i, l in enumerate(joint_all) if l < t), len(joint_all))
        seq_epochs_to.append(seq_ep)
        joint_epochs_to.append(joint_ep)

    x = np.arange(len(thresholds))
    width = 0.35
    ax.bar(x - width / 2, seq_epochs_to, width, label="Sequential", color="#2196F3", alpha=0.7)
    ax.bar(x + width / 2, joint_epochs_to, width, label="Joint", color="#E91E63", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"<{t}" for t in thresholds])
    ax.set_xlabel("MSE Threshold")
    ax.set_ylabel("Epochs to Reach")
    ax.set_title("Convergence Speed")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Sequential vs Joint Training — Sensorimotor Coupling Hypothesis", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    # Fast CPU experiment: small samples, moderate epochs
    cfg = Config(
        sound_dim=8,
        action_dim=8,
        signal_dim=8,
        hidden_dim=64,
        receiver_lr=1e-3,
        receiver_epochs=300,
        receiver_samples=2000,
        emitter_lr=1e-3,
        emitter_epochs=300,
        emitter_samples=2000,
        emitter_batch_size=64,
        receiver_batch_size=64,
    )

    total_epochs = cfg.receiver_epochs + cfg.emitter_epochs
    seed = 42

    os.makedirs("results", exist_ok=True)

    print("=" * 60)
    print("EXPERIMENT: Sequential vs Joint Training")
    print(f"Config: dim={cfg.sound_dim}, h={cfg.hidden_dim}, total_epochs={total_epochs}")
    print("=" * 60)

    print("\n--- Sequential Training (dim=8) ---")
    seq_result = run_sequential(cfg, seed)
    print(f"  Sequential test MSE: {seq_result['test_mse']:.6f}")

    print("\n--- Joint Training (dim=8) ---")
    joint_result = run_joint(cfg, seed, total_epochs)
    print(f"  Joint test MSE: {joint_result['test_mse']:.6f}")

    # dim=16 with same fast settings
    cfg16 = Config(
        sound_dim=16,
        action_dim=16,
        signal_dim=16,
        hidden_dim=64,
        receiver_lr=1e-3,
        receiver_epochs=300,
        receiver_samples=2000,
        emitter_lr=1e-3,
        emitter_epochs=300,
        emitter_samples=2000,
        emitter_batch_size=64,
        receiver_batch_size=64,
    )
    total_16 = cfg16.receiver_epochs + cfg16.emitter_epochs

    print("\n" + "=" * 60)
    print(f"EXPERIMENT: dim=16, total_epochs={total_16}")
    print("=" * 60)

    print("\n--- Sequential Training (dim=16) ---")
    seq16 = run_sequential(cfg16, seed)
    print(f"  Sequential test MSE: {seq16['test_mse']:.6f}")

    print("\n--- Joint Training (dim=16) ---")
    joint16 = run_joint(cfg16, seed, total_16)
    print(f"  Joint test MSE: {joint16['test_mse']:.6f}")

    # Generate comparison plots
    plot_comparison(seq_result, joint_result, total_epochs,
                    "results/obj-011-joint-vs-sequential-dim8.png")
    plot_comparison(seq16, joint16, total_16,
                    "results/obj-011-joint-vs-sequential-dim16.png")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  dim=8:  Sequential MSE={seq_result['test_mse']:.6f}  |  Joint MSE={joint_result['test_mse']:.6f}")
    print(f"  dim=16: Sequential MSE={seq16['test_mse']:.6f}  |  Joint MSE={joint16['test_mse']:.6f}")

    winner8 = "Joint" if joint_result["test_mse"] < seq_result["test_mse"] else "Sequential"
    winner16 = "Joint" if joint16["test_mse"] < seq16["test_mse"] else "Sequential"
    print(f"  dim=8 winner:  {winner8}")
    print(f"  dim=16 winner: {winner16}")


if __name__ == "__main__":
    main()
