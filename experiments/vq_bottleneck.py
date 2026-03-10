"""Experiment: VQ Bottleneck — Emergent Discrete Units.

Tests whether a vector quantization bottleneck between the Emitter and the
channel produces emergent discrete "phoneme-like" codes. Compares:
1. Continuous baseline (standard Emitter)
2. VQ bottleneck (Emitter → VQ → Channel → Receiver)

Analyzes codebook utilization and whether discrete codes cluster meaningfully.
"""

import os
import sys

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from components import (
    Emitter, ActionToSignal, Environment, Receiver, Pipeline, VectorQuantizer,
)
from train import pretrain_receiver


class VQPipeline(nn.Module):
    """Pipeline with VQ bottleneck: Emitter → VQ → ActionToSignal → Environment → Receiver."""

    def __init__(self, emitter, vq, action_to_signal, environment, receiver):
        super().__init__()
        self.emitter = emitter
        self.vq = vq
        self.action_to_signal = action_to_signal
        self.environment = environment
        self.receiver = receiver

    def forward(self, x):
        z = self.emitter(x)
        z_q, vq_loss, indices = self.vq(z)
        signal = self.action_to_signal(z_q)
        received = self.environment(signal)
        decoded = self.receiver(received)
        return decoded, vq_loss, indices


def train_vq_emitter(pipeline, cfg, epochs):
    """Train Emitter + VQ jointly with reconstruction + VQ loss."""
    pipeline.receiver.requires_grad_(False)
    pipeline.action_to_signal.requires_grad_(False)
    pipeline.environment.requires_grad_(False)

    params = list(pipeline.emitter.parameters()) + list(pipeline.vq.parameters())
    optimizer = torch.optim.Adam(params, lr=cfg.emitter_lr)
    recon_fn = nn.MSELoss()

    sounds = torch.randn(cfg.emitter_samples, cfg.sound_dim)

    losses = {"recon": [], "vq": [], "total": []}
    all_indices = []

    for epoch in range(epochs):
        perm = torch.randperm(sounds.size(0))
        ep_recon, ep_vq, ep_total = 0.0, 0.0, 0.0
        n = 0
        epoch_indices = []

        for i in range(0, sounds.size(0), cfg.emitter_batch_size):
            idx = perm[i : i + cfg.emitter_batch_size]
            batch = sounds[idx]

            decoded, vq_loss, indices = pipeline(batch)
            recon_loss = recon_fn(decoded, batch)
            total_loss = recon_loss + vq_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            ep_recon += recon_loss.item()
            ep_vq += vq_loss.item()
            ep_total += total_loss.item()
            n += 1
            epoch_indices.append(indices.detach())

        losses["recon"].append(ep_recon / n)
        losses["vq"].append(ep_vq / n)
        losses["total"].append(ep_total / n)
        all_indices.append(torch.cat(epoch_indices))

        if (epoch + 1) % 100 == 0:
            print(f"  VQ epoch {epoch + 1}/{epochs}, "
                  f"recon={ep_recon/n:.6f}, vq={ep_vq/n:.6f}")

    return losses, all_indices


def analyze_codebook(vq, all_indices, num_codes, output_prefix):
    """Analyze and visualize codebook utilization."""
    # Final epoch usage
    final_indices = all_indices[-1].numpy()
    usage_counts = np.bincount(final_indices, minlength=num_codes)
    active_codes = (usage_counts > 0).sum()
    entropy = -np.sum(
        (usage_counts / usage_counts.sum()) * np.log2(usage_counts / usage_counts.sum() + 1e-10)
    )

    print(f"\n  Codebook analysis:")
    print(f"    Active codes: {active_codes}/{num_codes}")
    print(f"    Usage entropy: {entropy:.2f} bits (max {np.log2(num_codes):.2f})")
    print(f"    Most used code: {usage_counts.max()} times")
    print(f"    Least used active code: {usage_counts[usage_counts > 0].min()} times")

    # Plot 1: Codebook usage histogram
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    ax.bar(range(num_codes), usage_counts, color="#2196F3", alpha=0.7)
    ax.set_xlabel("Code Index")
    ax.set_ylabel("Usage Count")
    ax.set_title(f"Codebook Utilization ({active_codes}/{num_codes} active)")
    ax.grid(True, alpha=0.3, axis="y")

    # Plot 2: Usage over training (track active codes per epoch)
    ax = axes[1]
    active_per_epoch = []
    for ep_indices in all_indices:
        ep_counts = np.bincount(ep_indices.numpy(), minlength=num_codes)
        active_per_epoch.append((ep_counts > 0).sum())
    ax.plot(range(1, len(active_per_epoch) + 1), active_per_epoch, color="#E91E63", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Active Codes")
    ax.set_title("Codebook Utilization Over Training")
    ax.set_ylim(0, num_codes + 1)
    ax.grid(True, alpha=0.3)

    # Plot 3: Codebook vectors heatmap
    ax = axes[2]
    codebook = vq.codebook.detach().cpu().numpy()
    # Sort by usage
    sorted_idx = np.argsort(-usage_counts)
    codebook_sorted = codebook[sorted_idx]
    im = ax.imshow(codebook_sorted, cmap="RdBu_r", aspect="auto",
                   vmin=-codebook_sorted.max(), vmax=codebook_sorted.max())
    ax.set_xlabel("Code Dimension")
    ax.set_ylabel("Code Index (sorted by usage)")
    ax.set_title("Learned Codebook Vectors")
    fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("VQ Bottleneck — Emergent Discrete Codes", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{output_prefix}-codebook.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {output_prefix}-codebook.png")

    return {
        "active_codes": int(active_codes),
        "entropy": float(entropy),
        "max_entropy": float(np.log2(num_codes)),
    }


def plot_vq_vs_continuous(cont_losses, vq_losses, cont_mse, vq_mse, codebook_stats, output_path):
    """Compare VQ and continuous training."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Loss curves
    ax = axes[0]
    ax.plot(range(1, len(cont_losses) + 1), cont_losses, label="Continuous", linewidth=1.5, color="#2196F3")
    ax.plot(range(1, len(vq_losses["recon"]) + 1), vq_losses["recon"], label="VQ (recon only)", linewidth=1.5, color="#E91E63")
    ax.plot(range(1, len(vq_losses["total"]) + 1), vq_losses["total"], label="VQ (total)", linewidth=1.5, color="#E91E63", linestyle="--", alpha=0.5)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (log)")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Final MSE comparison
    ax = axes[1]
    methods = ["Continuous", "VQ"]
    mses = [cont_mse, vq_mse]
    colors = ["#2196F3", "#E91E63"]
    bars = ax.bar(methods, mses, color=colors, alpha=0.7, edgecolor=colors, linewidth=2)
    ax.set_ylabel("Test MSE")
    ax.set_title("Final Test MSE")
    for bar, mse in zip(bars, mses):
        ax.text(bar.get_x() + bar.get_width() / 2, mse + 0.001, f"{mse:.6f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # VQ loss components
    ax = axes[2]
    ax.plot(range(1, len(vq_losses["recon"]) + 1), vq_losses["recon"], label="Reconstruction", linewidth=1.5)
    ax.plot(range(1, len(vq_losses["vq"]) + 1), vq_losses["vq"], label="VQ (commitment+codebook)", linewidth=1.5)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log)")
    ax.set_title("VQ Loss Components")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Continuous vs VQ Bottleneck | Codebook: {codebook_stats['active_codes']} active, "
                 f"entropy={codebook_stats['entropy']:.1f}/{codebook_stats['max_entropy']:.1f} bits",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_capacity_curve(results, cont_mse, output_path):
    """Plot codebook size vs MSE — the capacity curve."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    sizes = [r["num_codes"] for r in results]
    mses = [r["test_mse"] for r in results]
    active = [r["active_codes"] for r in results]
    entropies = [r["entropy"] for r in results]
    max_entropies = [r["max_entropy"] for r in results]

    # Panel 1: MSE vs codebook size
    ax = axes[0]
    ax.plot(sizes, mses, "o-", color="#E91E63", linewidth=2, markersize=8, label="VQ")
    ax.axhline(y=cont_mse, color="#2196F3", linestyle="--", linewidth=2, label=f"Continuous ({cont_mse:.4f})")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Codebook Size")
    ax.set_ylabel("Test MSE (log)")
    ax.set_title("Capacity Curve: Codes vs Reconstruction")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(sizes)
    ax.set_xticklabels(sizes)

    # Panel 2: Active codes / total
    ax = axes[1]
    ax.bar(range(len(sizes)), active, color="#4CAF50", alpha=0.7, label="Active")
    ax.bar(range(len(sizes)), [s - a for s, a in zip(sizes, active)], bottom=active,
           color="#ccc", alpha=0.5, label="Unused")
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels(sizes)
    ax.set_xlabel("Codebook Size")
    ax.set_ylabel("Code Count")
    ax.set_title("Codebook Utilization")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 3: Entropy utilization
    ax = axes[2]
    ax.bar(range(len(sizes)), entropies, color="#FF9800", alpha=0.7, label="Actual")
    ax.bar(range(len(sizes)), [m - e for m, e in zip(max_entropies, entropies)], bottom=entropies,
           color="#ccc", alpha=0.5, label="Unused capacity")
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels(sizes)
    ax.set_xlabel("Codebook Size")
    ax.set_ylabel("Entropy (bits)")
    ax.set_title("Information Utilization")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("VQ Bottleneck Capacity Curve — How Many Discrete Codes Do You Need?",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    codebook_sizes = [4, 8, 16, 32, 64, 128, 256]
    epochs = 300
    seed = 42

    cfg = Config(
        sound_dim=8,
        action_dim=8,
        signal_dim=8,
        hidden_dim=64,
        receiver_lr=1e-3,
        receiver_epochs=300,
        receiver_samples=2000,
        emitter_lr=1e-3,
        emitter_epochs=epochs,
        emitter_samples=2000,
        emitter_batch_size=64,
        receiver_batch_size=64,
    )

    os.makedirs("results", exist_ok=True)

    print("=" * 60)
    print(f"EXPERIMENT: VQ Codebook Size Sweep (dim={cfg.sound_dim})")
    print(f"Codebook sizes: {codebook_sizes}")
    print("=" * 60)

    # Shared setup: pre-train receiver once
    torch.manual_seed(seed)
    action_to_signal = ActionToSignal(cfg.action_dim, cfg.signal_dim, seed=100)
    environment = Environment(cfg.signal_dim, seed=200)
    receiver = Receiver(cfg.signal_dim, cfg.sound_dim, cfg.hidden_dim)
    recv_losses = pretrain_receiver(action_to_signal, environment, receiver, cfg)
    receiver.requires_grad_(False)
    receiver.eval()

    test_sounds = torch.randn(500, cfg.sound_dim)

    # Continuous baseline
    print("\n--- Continuous Baseline ---")
    torch.manual_seed(seed + 1)
    emitter_cont = Emitter(cfg.sound_dim, cfg.action_dim, cfg.hidden_dim)
    pipeline_cont = Pipeline(emitter_cont, action_to_signal, environment, receiver)
    from train import train_emitter
    cont_losses = train_emitter(pipeline_cont, cfg)
    pipeline_cont.eval()
    with torch.no_grad():
        cont_mse = nn.functional.mse_loss(pipeline_cont(test_sounds), test_sounds).item()
    print(f"  Continuous test MSE: {cont_mse:.6f}")

    # Sweep codebook sizes
    results = []
    for num_codes in codebook_sizes:
        print(f"\n--- VQ Bottleneck (codes={num_codes}) ---")
        torch.manual_seed(seed + 1)
        emitter_vq = Emitter(cfg.sound_dim, cfg.action_dim, cfg.hidden_dim)
        vq = VectorQuantizer(num_codes, cfg.action_dim, commitment_cost=0.1)

        # Initialize codebook from encoder outputs
        with torch.no_grad():
            warmup_z = emitter_vq(torch.randn(2000, cfg.sound_dim))
            perm = torch.randperm(warmup_z.size(0))[:num_codes]
            vq.codebook.copy_(warmup_z[perm])
            vq.ema_sum.copy_(warmup_z[perm])
            vq.ema_count.fill_(1.0)

        vq_pipeline = VQPipeline(emitter_vq, vq, action_to_signal, environment, receiver)
        vq_losses, all_indices = train_vq_emitter(vq_pipeline, cfg, epochs)

        vq_pipeline.eval()
        with torch.no_grad():
            decoded_vq, _, final_indices = vq_pipeline(test_sounds)
            vq_mse = nn.functional.mse_loss(decoded_vq, test_sounds).item()

        final_idx = all_indices[-1].numpy()
        usage = np.bincount(final_idx, minlength=num_codes)
        active = int((usage > 0).sum())
        entropy = -np.sum((usage / usage.sum()) * np.log2(usage / usage.sum() + 1e-10))

        print(f"  MSE={vq_mse:.6f}, active={active}/{num_codes}, entropy={entropy:.2f}/{np.log2(num_codes):.2f}")

        results.append({
            "num_codes": num_codes,
            "test_mse": vq_mse,
            "active_codes": active,
            "entropy": entropy,
            "max_entropy": np.log2(num_codes),
            "vq_losses": vq_losses,
            "all_indices": all_indices,
            "vq": vq,
        })

    # Analyze best codebook
    best = min(results, key=lambda r: r["test_mse"])
    print(f"\n  Best: codes={best['num_codes']}, MSE={best['test_mse']:.6f}")
    analyze_codebook(best["vq"], best["all_indices"], best["num_codes"],
                     "results/obj-012-vq-bottleneck")

    # Capacity curve
    plot_capacity_curve(results, cont_mse, "results/obj-012-vq-capacity-curve.png")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY — Codebook Size Sweep")
    print("=" * 60)
    print(f"  {'Codes':>6} | {'MSE':>10} | {'Active':>8} | {'Entropy':>12}")
    print(f"  {'------':>6} | {'----------':>10} | {'--------':>8} | {'------------':>12}")
    for r in results:
        print(f"  {r['num_codes']:>6} | {r['test_mse']:>10.6f} | {r['active_codes']:>3}/{r['num_codes']:<4} | {r['entropy']:.2f}/{r['max_entropy']:.2f} bits")
    print(f"  {'cont':>6} | {cont_mse:>10.6f} | {'--':>8} | {'--':>12}")
    print(f"\n  Continuous baseline: {cont_mse:.6f}")
    print(f"  Best VQ: codes={best['num_codes']}, MSE={best['test_mse']:.6f}")


if __name__ == "__main__":
    main()
