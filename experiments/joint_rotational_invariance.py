"""Experiment: Does Joint Training Overcome Channel Geometry?

Bridges obj-011 (joint > sequential) and obj-013 (channel κ dominates learning).

Key question: If Emitter and Receiver co-adapt (joint training), does the
channel condition number still dominate reconstruction quality? Or can joint
training discover coding schemes that neutralize ill-conditioning?

Sub-questions:
1. Does joint training reduce the MSE gap between orthogonal and ill-conditioned channels?
2. Under joint training, does the Emitter learn to boost weak singular directions
   (active pre-compensation) instead of staying near-identity?
3. Does per-direction error still follow 1/σ² under joint training, or does
   co-adaptation flatten the error profile?

Design: 2 training modes × 3 channel types = 6 conditions, all at dim=8.
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
from experiments.rotational_invariance import (
    make_orthogonal_transform,
    make_channel_components,
    analyze_channel,
    per_direction_error,
    emitter_alignment,
)
from experiments.joint_training import train_joint


def compute_jacobian(module, dim, n_samples=500, seed=99):
    """Estimate the Jacobian of a module via finite differences.

    Returns the average Jacobian matrix (dim × dim).
    """
    torch.manual_seed(seed)
    eps = 1e-4
    x = torch.randn(n_samples, dim)

    module.eval()
    with torch.no_grad():
        y0 = module(x)  # (n_samples, dim)

    J_sum = torch.zeros(dim, dim)
    for j in range(dim):
        x_plus = x.clone()
        x_plus[:, j] += eps
        with torch.no_grad():
            y_plus = module(x_plus)
        # dY/dx_j averaged over samples
        dydx_j = ((y_plus - y0) / eps).mean(dim=0)  # (dim,)
        J_sum[:, j] = dydx_j

    return J_sum


def run_sequential_with_channel(a2s, env, cfg, seed=42):
    """Sequential training: pre-train receiver, freeze, train emitter."""
    torch.manual_seed(seed)
    receiver = Receiver(cfg.signal_dim, cfg.sound_dim, cfg.hidden_dim)
    recv_losses = pretrain_receiver(a2s, env, receiver, cfg)
    receiver.requires_grad_(False)

    emitter = Emitter(cfg.sound_dim, cfg.action_dim, cfg.hidden_dim)
    pipeline = Pipeline(emitter, a2s, env, receiver)
    emit_losses = train_emitter(pipeline, cfg)

    all_losses = recv_losses + emit_losses
    return pipeline, all_losses


def run_joint_with_channel(a2s, env, cfg, total_epochs, seed=42):
    """Joint training: both emitter and receiver from scratch."""
    torch.manual_seed(seed)
    receiver = Receiver(cfg.signal_dim, cfg.sound_dim, cfg.hidden_dim)
    emitter = Emitter(cfg.sound_dim, cfg.action_dim, cfg.hidden_dim)
    pipeline = Pipeline(emitter, a2s, env, receiver)

    result = train_joint(pipeline, cfg, total_epochs)
    return pipeline, result["losses"]


def evaluate_pipeline(pipeline, Vh, dim, n_test=2000, seed=99):
    """Full evaluation: MSE, per-direction error, emitter alignment, Jacobians."""
    dir_errors, total_mse = per_direction_error(pipeline, Vh, n_test, seed)
    action_var = emitter_alignment(pipeline, Vh, n_test, seed)

    # Compute Jacobians
    M = pipeline.environment.weight @ pipeline.action_to_signal.weight
    emitter_J = compute_jacobian(pipeline.emitter, dim)
    receiver_J = compute_jacobian(pipeline.receiver, dim)

    # Compare to identity and M_inv
    M_inv = torch.linalg.inv(M)
    emit_dist_I = torch.norm(emitter_J - torch.eye(dim), p="fro").item()
    emit_dist_Minv = torch.norm(emitter_J - M_inv, p="fro").item()
    recv_dist_Minv = torch.norm(receiver_J - M_inv, p="fro").item()

    return {
        "total_mse": total_mse,
        "direction_errors": dir_errors,
        "action_variance": action_var,
        "emitter_J": emitter_J,
        "receiver_J": receiver_J,
        "emit_dist_identity": emit_dist_I,
        "emit_dist_Minv": emit_dist_Minv,
        "recv_dist_Minv": recv_dist_Minv,
    }


def run_experiment(dim=8, epochs_recv=400, epochs_emit=500, n_samples=2000):
    """Run the full 2×3 experiment grid."""
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

    modes = ["orthogonal", "random", "ill_conditioned"]
    training_types = ["sequential", "joint"]
    results = {}

    for mode in modes:
        for ttype in training_types:
            key = f"{mode}_{ttype}"
            print(f"\n{'='*60}")
            print(f"Channel: {mode.upper()} | Training: {ttype.upper()}")
            print(f"{'='*60}")

            # Build fresh channel components for each run (same seeds = same channel)
            a2s, env, M = make_channel_components(dim, mode)
            U, S, Vh, cond = analyze_channel(M, mode)

            if ttype == "sequential":
                pipeline, losses = run_sequential_with_channel(a2s, env, cfg)
            else:
                pipeline, losses = run_joint_with_channel(a2s, env, cfg, total_epochs)

            evals = evaluate_pipeline(pipeline, Vh, dim)

            results[key] = {
                "mode": mode,
                "training": ttype,
                "singular_values": S.numpy(),
                "condition_number": cond,
                "losses": losses,
                **evals,
            }

            print(f"  Test MSE: {evals['total_mse']:.6f}")
            print(f"  Emitter dist from I: {evals['emit_dist_identity']:.3f}")
            print(f"  Emitter dist from M⁻¹: {evals['emit_dist_Minv']:.3f}")
            print(f"  Receiver dist from M⁻¹: {evals['recv_dist_Minv']:.3f}")

    return results, dim


def plot_results(results, dim, output_path):
    """Generate 2×3 comparison visualization."""
    modes = ["orthogonal", "random", "ill_conditioned"]
    mode_labels = {"orthogonal": "Orthogonal", "random": "Random",
                   "ill_conditioned": "Ill-conditioned"}
    train_colors = {"sequential": "#2196F3", "joint": "#E91E63"}
    train_labels = {"sequential": "Sequential", "joint": "Joint"}

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # --- Panel 1: MSE comparison bar chart (the key result) ---
    ax = axes[0, 0]
    x = np.arange(len(modes))
    width = 0.35
    seq_mses = [results[f"{m}_sequential"]["total_mse"] for m in modes]
    jnt_mses = [results[f"{m}_joint"]["total_mse"] for m in modes]

    bars_s = ax.bar(x - width / 2, seq_mses, width, label="Sequential",
                    color=train_colors["sequential"], alpha=0.8)
    bars_j = ax.bar(x + width / 2, jnt_mses, width, label="Joint",
                    color=train_colors["joint"], alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([mode_labels[m] for m in modes])
    ax.set_ylabel("Test MSE")
    ax.set_yscale("log")
    ax.set_title("Reconstruction Quality: Sequential vs Joint")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Annotate bars
    for bars, mses in [(bars_s, seq_mses), (bars_j, jnt_mses)]:
        for bar, mse in zip(bars, mses):
            ax.text(bar.get_x() + bar.get_width() / 2, mse * 1.5,
                    f"{mse:.5f}", ha="center", va="bottom", fontsize=7,
                    fontweight="bold")

    # --- Panel 2: Sensitivity ratio (ill/ortho MSE ratio) ---
    ax = axes[0, 1]
    seq_ratio = seq_mses[2] / max(seq_mses[0], 1e-10)
    jnt_ratio = jnt_mses[2] / max(jnt_mses[0], 1e-10)
    bars = ax.bar(["Sequential", "Joint"], [seq_ratio, jnt_ratio],
                  color=[train_colors["sequential"], train_colors["joint"]],
                  alpha=0.8)
    ax.set_ylabel("MSE Ratio (Ill-cond / Orthogonal)")
    ax.set_title("Channel Sensitivity: How Much Does κ Matter?")
    for bar, ratio in zip(bars, [seq_ratio, jnt_ratio]):
        ax.text(bar.get_x() + bar.get_width() / 2, ratio + max(seq_ratio, jnt_ratio) * 0.03,
                f"{ratio:.0f}×", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # --- Panel 3: Training curves for ill-conditioned (hardest case) ---
    ax = axes[0, 2]
    for ttype in ["sequential", "joint"]:
        key = f"ill_conditioned_{ttype}"
        losses = results[key]["losses"]
        ax.plot(range(1, len(losses) + 1), losses, color=train_colors[ttype],
                label=train_labels[ttype], linewidth=1.5)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log)")
    ax.set_title("Training Curves: Ill-conditioned Channel")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panel 4: Per-direction error vs σ (sequential vs joint, ill-conditioned) ---
    ax = axes[1, 0]
    for ttype in ["sequential", "joint"]:
        key = f"ill_conditioned_{ttype}"
        sv = results[key]["singular_values"]
        de = results[key]["direction_errors"]
        ax.scatter(sv, de, color=train_colors[ttype], s=60, alpha=0.8,
                   label=train_labels[ttype], edgecolors="white", linewidth=0.5)

    # Reference line: 1/σ²
    sv_range = np.logspace(np.log10(min(sv) * 0.5), np.log10(max(sv) * 2), 50)
    # Scale reference to match data
    ref_scale = np.median(results["ill_conditioned_sequential"]["direction_errors"])
    median_sv = np.median(results["ill_conditioned_sequential"]["singular_values"])
    ref_line = ref_scale * (median_sv / sv_range) ** 2
    ax.plot(sv_range, ref_line, "k--", alpha=0.4, label="∝ 1/σ²")

    ax.set_xlabel("Channel singular value σ")
    ax.set_ylabel("Per-direction MSE")
    ax.set_title("Error vs. Channel Gain (Ill-conditioned)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panel 5: Emitter Jacobian analysis ---
    ax = axes[1, 1]
    categories = ["Emitter≈I", "Emitter≈M⁻¹", "Receiver≈M⁻¹"]
    x = np.arange(len(categories))
    width = 0.35

    seq_vals = []
    jnt_vals = []
    # Use ill-conditioned channel (most informative)
    for cat, key_fn in [
        ("Emitter≈I", lambda r: r["emit_dist_identity"]),
        ("Emitter≈M⁻¹", lambda r: r["emit_dist_Minv"]),
        ("Receiver≈M⁻¹", lambda r: r["recv_dist_Minv"]),
    ]:
        seq_vals.append(key_fn(results["ill_conditioned_sequential"]))
        jnt_vals.append(key_fn(results["ill_conditioned_joint"]))

    ax.bar(x - width / 2, seq_vals, width, label="Sequential",
           color=train_colors["sequential"], alpha=0.8)
    ax.bar(x + width / 2, jnt_vals, width, label="Joint",
           color=train_colors["joint"], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylabel("Frobenius Distance")
    ax.set_title("Who Inverts the Channel? (Ill-conditioned)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # --- Panel 6: Emitter energy allocation (ill-conditioned, seq vs joint) ---
    ax = axes[1, 2]
    x = np.arange(1, dim + 1)
    width = 0.35
    for i, ttype in enumerate(["sequential", "joint"]):
        key = f"ill_conditioned_{ttype}"
        av = results[key]["action_variance"]
        av_norm = av / av.sum()
        ax.bar(x + (i - 0.5) * width, av_norm, width=width,
               color=train_colors[ttype], alpha=0.8, label=train_labels[ttype])
    ax.set_xlabel("Singular direction index (sorted by σ)")
    ax.set_ylabel("Fraction of Emitter output variance")
    ax.set_title("Emitter Energy Allocation (Ill-conditioned)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"Does Joint Training Overcome Channel Geometry? (dim={dim})",
        fontsize=14, fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved: {output_path}")


def main():
    os.makedirs("results", exist_ok=True)

    print("=" * 60)
    print("EXPERIMENT: Joint Training vs Channel Geometry")
    print("Bridging obj-011 (joint > sequential) and obj-013 (κ dominates)")
    print("=" * 60)

    results, dim = run_experiment(
        dim=8, epochs_recv=400, epochs_emit=500, n_samples=2000
    )

    plot_results(results, dim, "results/obj-014-joint-vs-channel-geometry.png")

    # Print summary table
    modes = ["orthogonal", "random", "ill_conditioned"]
    mode_labels = {"orthogonal": "Orthogonal", "random": "Random",
                   "ill_conditioned": "Ill-cond"}

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  {'Channel':<14} | {'κ':>6} | {'Seq MSE':>10} | {'Joint MSE':>10} | {'Ratio':>8} | {'Winner':>10}")
    print(f"  {'-'*14} | {'-'*6} | {'-'*10} | {'-'*10} | {'-'*8} | {'-'*10}")
    for mode in modes:
        seq = results[f"{mode}_sequential"]
        jnt = results[f"{mode}_joint"]
        ratio = seq["total_mse"] / max(jnt["total_mse"], 1e-10)
        winner = "Joint" if jnt["total_mse"] < seq["total_mse"] else "Sequential"
        print(f"  {mode_labels[mode]:<14} | {seq['condition_number']:>6.0f} | "
              f"{seq['total_mse']:>10.6f} | {jnt['total_mse']:>10.6f} | "
              f"{ratio:>7.1f}× | {winner:>10}")

    # Channel sensitivity comparison
    seq_sensitivity = (results["ill_conditioned_sequential"]["total_mse"] /
                       max(results["orthogonal_sequential"]["total_mse"], 1e-10))
    jnt_sensitivity = (results["ill_conditioned_joint"]["total_mse"] /
                       max(results["orthogonal_joint"]["total_mse"], 1e-10))

    print(f"\n  Channel sensitivity (ill-cond/ortho MSE ratio):")
    print(f"    Sequential: {seq_sensitivity:.0f}×")
    print(f"    Joint:      {jnt_sensitivity:.0f}×")
    if jnt_sensitivity < seq_sensitivity:
        reduction = (1 - jnt_sensitivity / seq_sensitivity) * 100
        print(f"    → Joint training reduces channel sensitivity by {reduction:.0f}%")
    else:
        print(f"    → Joint training does NOT reduce channel sensitivity")

    # Jacobian analysis for ill-conditioned
    print(f"\n  Jacobian analysis (ill-conditioned channel):")
    for ttype in ["sequential", "joint"]:
        key = f"ill_conditioned_{ttype}"
        r = results[key]
        print(f"    {ttype.capitalize():>12}: Emitter≈I dist={r['emit_dist_identity']:.3f}, "
              f"Emitter≈M⁻¹ dist={r['emit_dist_Minv']:.3f}, "
              f"Receiver≈M⁻¹ dist={r['recv_dist_Minv']:.3f}")


if __name__ == "__main__":
    main()
