"""Experiment: Structured VQ — Phoneme Emergence from Clustered Inputs.

Tests whether VQ codes discover meaningful categories when input has natural
cluster structure (mixture of Gaussians), unlike random continuous inputs where
VQ provides no benefit.

Compares:
1. Random inputs + VQ (baseline — known to be poor)
2. Structured inputs + VQ (should show cluster-code alignment)
3. Structured inputs + continuous (upper bound)

Key metric: Normalized Mutual Information (NMI) between true cluster labels
and VQ code assignments. High NMI = VQ codes discovered the input categories.
"""

import os
import sys

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import normalized_mutual_info_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from components import (
    Emitter, ActionToSignal, Environment, Receiver, Pipeline, VectorQuantizer,
)
from train import pretrain_receiver


class VQPipeline(nn.Module):
    """Pipeline with VQ bottleneck: Emitter → VQ → Channel → Receiver."""

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


def generate_clustered_data(n_samples, dim, n_clusters, cluster_std=0.3, seed=42):
    """Generate mixture-of-Gaussians data with known cluster labels.

    Each cluster center is drawn from N(0, 1), then samples are drawn
    from N(center, cluster_std^2 * I) with equal probability per cluster.

    Returns:
        sounds: (n_samples, dim) tensor
        labels: (n_samples,) integer tensor of true cluster IDs
        centers: (n_clusters, dim) tensor of cluster centers
    """
    gen = torch.Generator().manual_seed(seed)
    centers = torch.randn(n_clusters, dim, generator=gen)

    samples_per_cluster = n_samples // n_clusters
    remainder = n_samples - samples_per_cluster * n_clusters

    all_sounds = []
    all_labels = []
    for i in range(n_clusters):
        n = samples_per_cluster + (1 if i < remainder else 0)
        noise = torch.randn(n, dim, generator=gen) * cluster_std
        all_sounds.append(centers[i].unsqueeze(0) + noise)
        all_labels.append(torch.full((n,), i, dtype=torch.long))

    sounds = torch.cat(all_sounds, dim=0)
    labels = torch.cat(all_labels, dim=0)

    # Shuffle
    perm = torch.randperm(sounds.size(0), generator=gen)
    return sounds[perm], labels[perm], centers


def train_vq_emitter(pipeline, sounds, cfg, epochs):
    """Train Emitter + VQ with reconstruction + VQ loss."""
    pipeline.receiver.requires_grad_(False)
    pipeline.action_to_signal.requires_grad_(False)
    pipeline.environment.requires_grad_(False)

    params = list(pipeline.emitter.parameters()) + list(pipeline.vq.parameters())
    optimizer = torch.optim.Adam(params, lr=cfg.emitter_lr)
    recon_fn = nn.MSELoss()

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
            print(f"  epoch {epoch + 1}/{epochs}, "
                  f"recon={ep_recon/n:.6f}, vq={ep_vq/n:.6f}")

    return losses, all_indices


def train_continuous_emitter(pipeline, sounds, cfg, epochs):
    """Train continuous pipeline (no VQ)."""
    pipeline.receiver.requires_grad_(False)
    pipeline.action_to_signal.requires_grad_(False)
    pipeline.environment.requires_grad_(False)

    optimizer = torch.optim.Adam(pipeline.emitter.parameters(), lr=cfg.emitter_lr)
    loss_fn = nn.MSELoss()
    losses = []

    for epoch in range(epochs):
        perm = torch.randperm(sounds.size(0))
        ep_loss = 0.0
        n = 0
        for i in range(0, sounds.size(0), cfg.emitter_batch_size):
            idx = perm[i : i + cfg.emitter_batch_size]
            batch = sounds[idx]
            decoded = pipeline(batch)
            loss = loss_fn(decoded, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
            n += 1
        losses.append(ep_loss / n)
        if (epoch + 1) % 100 == 0:
            print(f"  epoch {epoch + 1}/{epochs}, loss={ep_loss/n:.6f}")

    return losses


def compute_nmi(true_labels, code_indices):
    """Compute Normalized Mutual Information between cluster labels and VQ codes."""
    return normalized_mutual_info_score(
        true_labels.numpy(), code_indices.numpy(), average_method="arithmetic"
    )


def plot_results(structured_results, random_results, cont_mse_struct, output_path):
    """Main results figure: NMI, MSE comparison, loss curves."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Row 1: Structured inputs
    # Panel 1: NMI over training
    ax = axes[0, 0]
    epochs_x = range(1, len(structured_results["nmi_over_training"]) + 1)
    ax.plot(epochs_x, structured_results["nmi_over_training"],
            color="#4CAF50", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("NMI (cluster labels vs VQ codes)")
    ax.set_title(f"Phoneme Emergence — Final NMI={structured_results['nmi']:.3f}")
    ax.set_ylim(0, 1.05)
    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5, label="Perfect alignment")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Confusion matrix (cluster vs code)
    ax = axes[0, 1]
    n_clusters = structured_results["n_clusters"]
    num_codes = structured_results["num_codes"]
    confusion = np.zeros((n_clusters, num_codes))
    labels_np = structured_results["test_labels"].numpy()
    codes_np = structured_results["test_codes"].numpy()
    for lbl, code in zip(labels_np, codes_np):
        confusion[lbl, code] += 1
    # Normalize rows
    row_sums = confusion.sum(axis=1, keepdims=True)
    confusion_norm = confusion / np.maximum(row_sums, 1)
    im = ax.imshow(confusion_norm, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    ax.set_xlabel("VQ Code Index")
    ax.set_ylabel("True Cluster")
    ax.set_title("Cluster→Code Assignment")
    fig.colorbar(im, ax=ax, shrink=0.8, label="P(code | cluster)")

    # Panel 3: MSE comparison bar chart
    ax = axes[0, 2]
    methods = ["Random+VQ", "Struct+VQ", "Struct+Cont"]
    mses = [random_results["test_mse"], structured_results["test_mse"], cont_mse_struct]
    colors = ["#9E9E9E", "#E91E63", "#2196F3"]
    bars = ax.bar(methods, mses, color=colors, alpha=0.8, edgecolor=colors, linewidth=2)
    ax.set_ylabel("Test MSE")
    ax.set_title("Reconstruction Quality")
    for bar, mse in zip(bars, mses):
        ax.text(bar.get_x() + bar.get_width() / 2, mse + max(mses) * 0.02,
                f"{mse:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # Row 2: Comparisons
    # Panel 4: Training loss curves
    ax = axes[1, 0]
    ax.plot(range(1, len(structured_results["losses"]["recon"]) + 1),
            structured_results["losses"]["recon"],
            label="Struct+VQ", color="#E91E63", linewidth=1.5)
    ax.plot(range(1, len(random_results["losses"]["recon"]) + 1),
            random_results["losses"]["recon"],
            label="Random+VQ", color="#9E9E9E", linewidth=1.5)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Recon Loss (log)")
    ax.set_title("Training Loss Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 5: Codebook utilization comparison
    ax = axes[1, 1]
    x_pos = np.arange(2)
    active = [random_results["active_codes"], structured_results["active_codes"]]
    total = [random_results["num_codes"], structured_results["num_codes"]]
    ax.bar(x_pos, active, color=["#9E9E9E", "#4CAF50"], alpha=0.8)
    ax.bar(x_pos, [t - a for t, a in zip(total, active)], bottom=active,
           color="#ccc", alpha=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(["Random+VQ", "Struct+VQ"])
    ax.set_ylabel("Code Count")
    ax.set_title("Codebook Utilization")
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 6: NMI comparison
    ax = axes[1, 2]
    nmi_vals = [random_results["nmi"], structured_results["nmi"]]
    ax.bar(["Random+VQ", "Struct+VQ"], nmi_vals,
           color=["#9E9E9E", "#4CAF50"], alpha=0.8, edgecolor=["#9E9E9E", "#4CAF50"], linewidth=2)
    ax.set_ylabel("NMI")
    ax.set_title("Cluster-Code Alignment")
    ax.set_ylim(0, 1.05)
    for i, v in enumerate(nmi_vals):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Structured VQ: Emergent Phonemes from Clustered Inputs",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def run_vq_experiment(sounds, labels, n_clusters, num_codes, cfg, epochs, seed,
                      action_to_signal, environment, receiver, test_sounds, test_labels,
                      label=""):
    """Run a single VQ experiment and return results dict."""
    torch.manual_seed(seed)
    emitter = Emitter(cfg.sound_dim, cfg.action_dim, cfg.hidden_dim)
    vq = VectorQuantizer(num_codes, cfg.action_dim, commitment_cost=0.1)

    # Initialize codebook from encoder outputs
    with torch.no_grad():
        warmup_z = emitter(sounds[:min(2000, sounds.size(0))])
        perm_init = torch.randperm(warmup_z.size(0))[:num_codes]
        vq.codebook.copy_(warmup_z[perm_init])
        vq.ema_sum.copy_(warmup_z[perm_init])
        vq.ema_count.fill_(1.0)

    vq_pipeline = VQPipeline(emitter, vq, action_to_signal, environment, receiver)

    print(f"\n--- {label} VQ (codes={num_codes}) ---")
    losses, all_indices = train_vq_emitter(vq_pipeline, sounds, cfg, epochs)

    # Compute NMI over training
    nmi_over_training = []
    for ep_indices in all_indices:
        nmi_over_training.append(compute_nmi(labels, ep_indices))

    # Test evaluation
    vq_pipeline.eval()
    with torch.no_grad():
        decoded, _, test_code_indices = vq_pipeline(test_sounds)
        test_mse = nn.functional.mse_loss(decoded, test_sounds).item()

    final_nmi = compute_nmi(test_labels, test_code_indices)

    # Codebook stats
    final_idx = all_indices[-1].numpy()
    usage = np.bincount(final_idx, minlength=num_codes)
    active = int((usage > 0).sum())
    entropy = -np.sum((usage / usage.sum()) * np.log2(usage / usage.sum() + 1e-10))

    print(f"  MSE={test_mse:.6f}, NMI={final_nmi:.3f}, "
          f"active={active}/{num_codes}, entropy={entropy:.2f}/{np.log2(num_codes):.2f}")

    return {
        "test_mse": test_mse,
        "nmi": final_nmi,
        "nmi_over_training": nmi_over_training,
        "active_codes": active,
        "num_codes": num_codes,
        "n_clusters": n_clusters,
        "entropy": entropy,
        "max_entropy": np.log2(num_codes),
        "losses": losses,
        "test_labels": test_labels,
        "test_codes": test_code_indices,
    }


def main():
    dim = 8
    n_clusters = 8
    num_codes = 16  # More codes than clusters — can it still discover the structure?
    epochs = 500
    n_train = 4000
    n_test = 500
    seed = 42
    cluster_std = 0.3

    cfg = Config(
        sound_dim=dim,
        action_dim=dim,
        signal_dim=dim,
        hidden_dim=64,
        receiver_lr=1e-3,
        receiver_epochs=500,
        receiver_samples=4000,
        emitter_lr=1e-3,
        emitter_epochs=epochs,
        emitter_samples=n_train,
        emitter_batch_size=64,
        receiver_batch_size=64,
    )

    os.makedirs("results", exist_ok=True)

    print("=" * 60)
    print("EXPERIMENT: Structured VQ — Phoneme Emergence")
    print(f"dim={dim}, clusters={n_clusters}, codes={num_codes}, "
          f"cluster_std={cluster_std}, epochs={epochs}")
    print("=" * 60)

    # Generate data
    print("\nGenerating clustered data...")
    train_sounds, train_labels, centers = generate_clustered_data(
        n_train, dim, n_clusters, cluster_std=cluster_std, seed=seed
    )
    test_sounds, test_labels, _ = generate_clustered_data(
        n_test, dim, n_clusters, cluster_std=cluster_std, seed=seed + 100
    )

    # Also generate random data for comparison
    torch.manual_seed(seed)
    random_train = torch.randn(n_train, dim)
    random_labels = torch.zeros(n_train, dtype=torch.long)  # No real clusters
    random_test = torch.randn(n_test, dim)
    random_test_labels = torch.zeros(n_test, dtype=torch.long)

    print(f"  Train: {n_train} samples, Test: {n_test} samples")
    print(f"  Cluster centers shape: {centers.shape}")

    # Pre-train receiver (shared across all experiments)
    print("\n--- Pre-training Receiver ---")
    torch.manual_seed(seed)
    action_to_signal = ActionToSignal(cfg.action_dim, cfg.signal_dim, seed=100)
    environment = Environment(cfg.signal_dim, seed=200)
    receiver = Receiver(cfg.signal_dim, cfg.sound_dim, cfg.hidden_dim)
    pretrain_receiver(action_to_signal, environment, receiver, cfg)
    receiver.requires_grad_(False)
    receiver.eval()

    # Experiment 1: Structured inputs + VQ
    struct_results = run_vq_experiment(
        train_sounds, train_labels, n_clusters, num_codes, cfg, epochs, seed + 1,
        action_to_signal, environment, receiver, test_sounds, test_labels,
        label="Structured"
    )

    # Experiment 2: Random inputs + VQ (baseline)
    random_results = run_vq_experiment(
        random_train, random_labels, 1, num_codes, cfg, epochs, seed + 2,
        action_to_signal, environment, receiver, random_test, random_test_labels,
        label="Random"
    )

    # Experiment 3: Structured inputs + continuous (upper bound)
    print("\n--- Structured Continuous (upper bound) ---")
    torch.manual_seed(seed + 3)
    emitter_cont = Emitter(cfg.sound_dim, cfg.action_dim, cfg.hidden_dim)
    pipeline_cont = Pipeline(emitter_cont, action_to_signal, environment, receiver)
    cont_losses = train_continuous_emitter(pipeline_cont, train_sounds, cfg, epochs)
    pipeline_cont.eval()
    with torch.no_grad():
        cont_mse = nn.functional.mse_loss(pipeline_cont(test_sounds), test_sounds).item()
    print(f"  Continuous test MSE: {cont_mse:.6f}")

    # Plot main results
    plot_results(struct_results, random_results, cont_mse,
                 "results/obj-013-structured-vq.png")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY — Structured VQ Experiment")
    print("=" * 60)
    print(f"  {'Method':<20} | {'MSE':>10} | {'NMI':>6} | {'Active':>8}")
    print(f"  {'--------------------':<20} | {'----------':>10} | {'------':>6} | {'--------':>8}")
    print(f"  {'Random+VQ':<20} | {random_results['test_mse']:>10.6f} | {random_results['nmi']:>6.3f} | {random_results['active_codes']:>3}/{num_codes}")
    print(f"  {'Structured+VQ':<20} | {struct_results['test_mse']:>10.6f} | {struct_results['nmi']:>6.3f} | {struct_results['active_codes']:>3}/{num_codes}")
    print(f"  {'Structured+Cont':<20} | {cont_mse:>10.6f} | {'--':>6} | {'--':>8}")
    print(f"\n  Key result: Structured VQ NMI = {struct_results['nmi']:.3f}")
    print(f"  (NMI=1.0 means VQ codes perfectly match input clusters)")


if __name__ == "__main__":
    main()
