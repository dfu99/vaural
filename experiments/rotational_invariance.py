"""Experiment: Rotational Invariance of the Communication Channel.

Investigates how the spectral structure of the fixed channel transforms
(ActionToSignal @ Environment) affects learning and reconstruction.

Key questions:
1. Orthogonal (rotation) transforms vs. random transforms — how does
   condition number affect convergence and final MSE?
2. Per-singular-direction error — does reconstruction fail preferentially
   on directions with small singular values?
3. Does the Emitter learn to align with the channel's singular structure?
4. Does anisotropic (directional) input data interact with channel geometry?

The combined channel matrix M = Env.weight @ A2S.weight has a singular value
decomposition M = U @ diag(σ) @ V^T. Directions in V with large σ pass through
the channel cleanly; directions with small σ are attenuated and harder to recover.
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


def make_orthogonal_transform(dim, seed):
    """Create a random orthogonal matrix via QR decomposition."""
    gen = torch.Generator().manual_seed(seed)
    M = torch.randn(dim, dim, generator=gen)
    Q, R = torch.linalg.qr(M)
    # Ensure det(Q) = +1 (proper rotation, not reflection)
    diag_sign = torch.sign(torch.diag(R))
    Q = Q * diag_sign.unsqueeze(0)
    return Q


def make_channel_components(dim, mode, seed_a2s=100, seed_env=200):
    """Build ActionToSignal + Environment with specified transform type.

    Args:
        mode: "random" (default Gaussian), "orthogonal" (pure rotation),
              "ill_conditioned" (exaggerated singular value spread)

    Returns:
        action_to_signal, environment, combined_matrix M
    """
    a2s = ActionToSignal(dim, dim, seed=seed_a2s)
    env = Environment(dim, seed=seed_env)

    if mode == "orthogonal":
        a2s.weight.copy_(make_orthogonal_transform(dim, seed_a2s))
        env.weight.copy_(make_orthogonal_transform(dim, seed_env))
    elif mode == "ill_conditioned":
        # Start from orthogonal, then scale singular values geometrically
        Q_a = make_orthogonal_transform(dim, seed_a2s)
        Q_e = make_orthogonal_transform(dim, seed_env)
        # Create a diagonal with condition number ~1000
        sigmas = torch.logspace(0, -3, dim)  # 1.0 down to 0.001
        a2s.weight.copy_(Q_a @ torch.diag(sigmas))
        env.weight.copy_(Q_e)
    # else: "random" — keep the default Gaussian initialization

    M = env.weight @ a2s.weight
    return a2s, env, M


def analyze_channel(M, label=""):
    """Compute SVD and return spectral stats."""
    U, S, Vh = torch.linalg.svd(M)
    cond = (S[0] / S[-1]).item()
    print(f"  [{label}] singular values: {S.numpy().round(3)}")
    print(f"  [{label}] condition number: {cond:.1f}")
    return U, S, Vh, cond


def train_pipeline(a2s, env, cfg, seed=42):
    """Pre-train receiver, then train emitter. Return trained pipeline + losses."""
    torch.manual_seed(seed)
    receiver = Receiver(cfg.signal_dim, cfg.sound_dim, cfg.hidden_dim)

    recv_losses = pretrain_receiver(a2s, env, receiver, cfg)
    receiver.requires_grad_(False)

    emitter = Emitter(cfg.sound_dim, cfg.action_dim, cfg.hidden_dim)
    pipeline = Pipeline(emitter, a2s, env, receiver)

    emit_losses = train_emitter(pipeline, cfg)
    return pipeline, recv_losses, emit_losses


def per_direction_error(pipeline, V, n_test=2000, seed=99):
    """Measure reconstruction MSE along each right-singular direction of the channel.

    Projects test data onto each singular direction, passes through pipeline,
    and measures per-direction reconstruction error.
    """
    torch.manual_seed(seed)
    dim = V.shape[0]
    sounds = torch.randn(n_test, dim)

    pipeline.eval()
    with torch.no_grad():
        reconstructed = pipeline(sounds)
        errors = (reconstructed - sounds)  # (n_test, dim)

    # Project errors onto each right-singular direction
    # V is (dim, dim) — rows are right singular vectors
    direction_errors = []
    for i in range(dim):
        v_i = V[i]  # i-th right singular vector
        # Project each error vector onto v_i
        proj = (errors @ v_i).pow(2).mean().item()
        direction_errors.append(proj)

    # Also measure total MSE
    total_mse = errors.pow(2).mean().item()
    return direction_errors, total_mse


def emitter_alignment(pipeline, V, n_test=2000, seed=99):
    """Measure how the Emitter's output aligns with the channel's singular vectors.

    For each input direction v_i, check what fraction of the Emitter output
    variance falls along each singular direction. If the Emitter is "aware" of
    the channel structure, it should concentrate energy on high-σ directions.
    """
    torch.manual_seed(seed)
    dim = V.shape[0]
    sounds = torch.randn(n_test, dim)

    pipeline.eval()
    with torch.no_grad():
        actions = pipeline.emitter(sounds)  # (n_test, dim)

    # Project actions onto singular basis
    # actions_proj[i] = variance of action projections onto v_i
    action_projections = actions @ V.T  # (n_test, dim)
    variance_per_dir = action_projections.var(dim=0).numpy()  # (dim,)
    return variance_per_dir


def run_experiment(dim=8, epochs_recv=800, epochs_emit=1000, n_samples=4000):
    """Main experiment comparing channel types."""
    cfg = Config(
        sound_dim=dim, action_dim=dim, signal_dim=dim,
        hidden_dim=64,
        receiver_lr=1e-3, receiver_epochs=epochs_recv,
        receiver_samples=n_samples, receiver_batch_size=64,
        emitter_lr=1e-3, emitter_epochs=epochs_emit,
        emitter_samples=n_samples, emitter_batch_size=64,
        plot_every=200,
    )

    modes = ["random", "orthogonal", "ill_conditioned"]
    results = {}

    for mode in modes:
        print(f"\n{'='*60}")
        print(f"Channel type: {mode.upper()}")
        print(f"{'='*60}")

        a2s, env, M = make_channel_components(dim, mode)
        U, S, Vh, cond = analyze_channel(M, mode)

        pipeline, recv_losses, emit_losses = train_pipeline(a2s, env, cfg, seed=42)

        dir_errors, total_mse = per_direction_error(pipeline, Vh)
        action_var = emitter_alignment(pipeline, Vh)

        results[mode] = {
            "singular_values": S.numpy(),
            "condition_number": cond,
            "recv_losses": recv_losses,
            "emit_losses": emit_losses,
            "direction_errors": dir_errors,
            "total_mse": total_mse,
            "action_variance": action_var,
            "U": U, "S": S, "Vh": Vh,
        }

        print(f"  Final receiver loss: {recv_losses[-1]:.6f}")
        print(f"  Final emitter loss:  {emit_losses[-1]:.6f}")
        print(f"  Test MSE: {total_mse:.6f}")

    return results, dim


def plot_results(results, dim, output_path):
    """Generate comprehensive visualization."""
    modes = ["random", "orthogonal", "ill_conditioned"]
    mode_labels = {"random": "Random", "orthogonal": "Orthogonal",
                   "ill_conditioned": "Ill-conditioned"}
    mode_colors = {"random": "#2196F3", "orthogonal": "#4CAF50",
                   "ill_conditioned": "#E91E63"}

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # --- Panel 1: Singular value spectra ---
    ax = axes[0, 0]
    for mode in modes:
        sv = results[mode]["singular_values"]
        ax.plot(range(1, dim + 1), sv, "o-", color=mode_colors[mode],
                label=f"{mode_labels[mode]} (κ={results[mode]['condition_number']:.0f})",
                linewidth=2, markersize=6)
    ax.set_xlabel("Singular value index")
    ax.set_ylabel("σ")
    ax.set_title("Channel Singular Value Spectrum")
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Receiver training curves ---
    ax = axes[0, 1]
    for mode in modes:
        losses = results[mode]["recv_losses"]
        ax.plot(range(1, len(losses) + 1), losses, color=mode_colors[mode],
                label=mode_labels[mode], linewidth=1.5)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Receiver Loss (log)")
    ax.set_title("Receiver Pre-training")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panel 3: Emitter training curves ---
    ax = axes[0, 2]
    for mode in modes:
        losses = results[mode]["emit_losses"]
        ax.plot(range(1, len(losses) + 1), losses, color=mode_colors[mode],
                label=mode_labels[mode], linewidth=1.5)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Emitter Loss (log)")
    ax.set_title("Emitter Training")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panel 4: Per-direction reconstruction error vs. singular value ---
    ax = axes[1, 0]
    for mode in modes:
        sv = results[mode]["singular_values"]
        de = results[mode]["direction_errors"]
        ax.scatter(sv, de, color=mode_colors[mode], s=60, alpha=0.8,
                   label=mode_labels[mode], edgecolors="white", linewidth=0.5)
    ax.set_xlabel("Channel singular value σ")
    ax.set_ylabel("Per-direction MSE")
    ax.set_title("Error vs. Channel Gain per Direction")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panel 5: Emitter output variance per singular direction ---
    ax = axes[1, 1]
    x = np.arange(1, dim + 1)
    width = 0.25
    for i, mode in enumerate(modes):
        av = results[mode]["action_variance"]
        sv = results[mode]["singular_values"]
        # Normalize variance so it sums to 1 for comparison
        av_norm = av / av.sum()
        ax.bar(x + (i - 1) * width, av_norm, width=width,
               color=mode_colors[mode], alpha=0.8, label=mode_labels[mode])
    ax.set_xlabel("Singular direction index (sorted by σ)")
    ax.set_ylabel("Fraction of Emitter output variance")
    ax.set_title("Emitter Energy Allocation across Channel Directions")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # --- Panel 6: Summary bar chart ---
    ax = axes[1, 2]
    mses = [results[m]["total_mse"] for m in modes]
    conds = [results[m]["condition_number"] for m in modes]
    bars = ax.bar([mode_labels[m] for m in modes], mses,
                  color=[mode_colors[m] for m in modes], alpha=0.8,
                  edgecolor=[mode_colors[m] for m in modes], linewidth=2)
    ax.set_ylabel("Test MSE")
    ax.set_title("Reconstruction Quality by Channel Type")
    for bar, mse, cond in zip(bars, mses, conds):
        ax.text(bar.get_x() + bar.get_width() / 2, mse + max(mses) * 0.03,
                f"MSE={mse:.5f}\nκ={cond:.0f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"Rotational Invariance: Channel Geometry and Learning (dim={dim})",
        fontsize=14, fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved: {output_path}")


def main():
    os.makedirs("results", exist_ok=True)

    print("=" * 60)
    print("EXPERIMENT: Rotational Invariance of Communication Channel")
    print("=" * 60)

    results, dim = run_experiment(
        dim=8, epochs_recv=800, epochs_emit=1000, n_samples=4000
    )

    plot_results(results, dim, "results/rotational_invariance.png")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    modes = ["random", "orthogonal", "ill_conditioned"]
    print(f"  {'Channel':<18} | {'κ':>8} | {'Recv Loss':>10} | {'Emit Loss':>10} | {'Test MSE':>10}")
    print(f"  {'-'*18} | {'-'*8} | {'-'*10} | {'-'*10} | {'-'*10}")
    for mode in modes:
        r = results[mode]
        print(f"  {mode:<18} | {r['condition_number']:>8.1f} | "
              f"{r['recv_losses'][-1]:>10.6f} | {r['emit_losses'][-1]:>10.6f} | "
              f"{r['total_mse']:>10.6f}")

    print("\nKey observations:")
    orth_mse = results["orthogonal"]["total_mse"]
    rand_mse = results["random"]["total_mse"]
    ill_mse = results["ill_conditioned"]["total_mse"]

    if orth_mse < rand_mse:
        print(f"  - Orthogonal channel is {rand_mse/max(orth_mse,1e-10):.1f}x easier than random")
    if ill_mse > rand_mse:
        print(f"  - Ill-conditioned channel is {ill_mse/max(rand_mse,1e-10):.1f}x harder than random")

    print(f"  - Condition numbers: ortho={results['orthogonal']['condition_number']:.1f}, "
          f"random={results['random']['condition_number']:.1f}, "
          f"ill={results['ill_conditioned']['condition_number']:.1f}")


if __name__ == "__main__":
    main()
