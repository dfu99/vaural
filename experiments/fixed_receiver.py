"""Experiment: Fixed Receiver — Can the Emitter learn to encode for the channel?

Biological motivation: ears are fixed transducers. They can't be trained.
The auditory cortex learns to decode, but the physical sensor is invariant.

In the current vaural setup, the Receiver pre-trains to invert the channel
(Jacobian ≈ M⁻¹), so the Emitter learns ≈ identity (C_i ≈ 0). This is
biologically unrealistic — the Receiver gets a "free lunch."

This experiment tests three Receiver regimes:
  1. Trained (current default) — Receiver pre-trained with identity mapping
  2. Fixed random — Receiver is a random MLP, never trained
  3. Fixed linear — Receiver is a random linear projection, never trained

For each regime, train the Emitter end-to-end and measure:
  - Final MSE (can communication work at all?)
  - Emitter Jacobian (does it learn M⁻¹, or something else?)
  - C_i = cos(emitter(s), M⁻¹·s) (does the Emitter now align with optimal?)

If the fixed-receiver Emitter learns a non-identity mapping with high C_i,
it confirms the PI's intuition: forcing the Emitter to encode for the
channel produces biologically realistic behavior.
"""

import os
import sys
import json

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from components import Emitter, Receiver, ActionToSignal, Environment, Pipeline
from train import pretrain_receiver, train_emitter
from experiments.pure_rotational_invariance import make_channel_from_svd


class FixedLinearReceiver(nn.Module):
    """Fixed random linear projection (no nonlinearity, no training)."""
    def __init__(self, in_dim, out_dim, seed=300):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        weight = torch.randn(out_dim, in_dim, generator=gen)
        self.register_buffer("weight", weight)

    def forward(self, x):
        return x @ self.weight.T


def compute_jacobian(model, x):
    """Compute Jacobian of model at single input x."""
    x = x.unsqueeze(0).requires_grad_(True)
    y = model(x)
    d_out = y.shape[-1]
    J = []
    for i in range(d_out):
        model.zero_grad()
        if x.grad is not None:
            x.grad.zero_()
        y[0, i].backward(retain_graph=True)
        J.append(x.grad[0].clone())
    return torch.stack(J)


def compute_ci(emitter, M_inv, dim, n_samples=2000, seed=99):
    """C_i = cos(emitter(s), M⁻¹·s)."""
    torch.manual_seed(seed)
    sounds = torch.randn(n_samples, dim)
    emitter.eval()
    with torch.no_grad():
        emit_out = emitter(sounds)
        optimal = (M_inv @ sounds.T).T
        cos = nn.functional.cosine_similarity(emit_out, optimal, dim=-1)
    emitter.train()
    return cos.mean().item(), cos.std().item()


def compute_jacobian_analysis(emitter, M_inv, dim, n_points=20):
    """Compute Emitter Jacobian and compare to M⁻¹ and I."""
    emitter.eval()
    torch.manual_seed(42)
    test_points = torch.randn(n_points, dim)

    dist_to_Minv = []
    dist_to_I = []
    I = torch.eye(dim)

    for i in range(n_points):
        J = compute_jacobian(emitter, test_points[i])
        dist_to_Minv.append((J - M_inv).norm().item())
        dist_to_I.append((J - I).norm().item())

    emitter.train()
    return np.mean(dist_to_Minv), np.mean(dist_to_I)


def evaluate(pipeline, dim, n_test=2000, seed=99):
    torch.manual_seed(seed)
    sounds = torch.randn(n_test, dim)
    pipeline.eval()
    with torch.no_grad():
        mse = (pipeline(sounds) - sounds).pow(2).mean().item()
    pipeline.train()
    return mse


def run_experiment(dim=8, n_rotations=4):
    print("=" * 60)
    print("EXPERIMENT: Fixed Receiver")
    print("Can the Emitter learn to encode for the channel?")
    print("=" * 60)

    sigmas = torch.logspace(0, -1, dim)  # κ=10
    kappa = (sigmas[0] / sigmas[-1]).item()

    cfg = Config(
        sound_dim=dim, action_dim=dim, signal_dim=dim, hidden_dim=64,
        receiver_lr=1e-3, receiver_epochs=200,
        receiver_samples=2000, receiver_batch_size=64,
        emitter_lr=1e-3, emitter_epochs=500,
        emitter_samples=2000, emitter_batch_size=64,
        plot_every=9999,
    )

    # More emitter epochs for fixed receiver (harder problem)
    cfg_hard = Config(
        sound_dim=dim, action_dim=dim, signal_dim=dim, hidden_dim=64,
        receiver_lr=1e-3, receiver_epochs=200,
        receiver_samples=2000, receiver_batch_size=64,
        emitter_lr=1e-3, emitter_epochs=1000,
        emitter_samples=2000, emitter_batch_size=64,
        plot_every=9999,
    )

    results = {"trained": [], "fixed_mlp": [], "fixed_linear": []}

    for rot_idx in range(n_rotations):
        seed_u = 1000 + rot_idx * 7
        seed_v = 2000 + rot_idx * 13
        a2s, env, M = make_channel_from_svd(dim, sigmas, seed_u, seed_v)
        M_inv = torch.linalg.inv(M)

        print(f"\n--- Rotation {rot_idx+1}/{n_rotations} ---")

        # === 1. Trained Receiver (current default) ===
        print("  Trained Receiver:")
        torch.manual_seed(42)
        recv_trained = Receiver(dim, dim, 64)
        pretrain_receiver(a2s, env, recv_trained, cfg)
        recv_trained.requires_grad_(False)

        torch.manual_seed(42)
        emit_trained = Emitter(dim, dim, 64)
        pipe_trained = Pipeline(emit_trained, a2s, env, recv_trained)
        train_emitter(pipe_trained, cfg)

        mse_trained = evaluate(pipe_trained, dim)
        ci_trained, ci_std = compute_ci(emit_trained, M_inv, dim)
        j_minv, j_i = compute_jacobian_analysis(emit_trained, M_inv, dim)
        results["trained"].append({
            "mse": mse_trained, "ci": ci_trained, "ci_std": ci_std,
            "j_dist_minv": j_minv, "j_dist_i": j_i,
        })
        print(f"    MSE={mse_trained:.6f}, C_i={ci_trained:.4f}, "
              f"J→M⁻¹={j_minv:.3f}, J→I={j_i:.3f}")

        # === 2. Fixed Random MLP Receiver ===
        print("  Fixed Random MLP Receiver:")
        torch.manual_seed(42)
        recv_fixed_mlp = Receiver(dim, dim, 64)
        recv_fixed_mlp.requires_grad_(False)  # Never train!

        torch.manual_seed(42)
        emit_fixed_mlp = Emitter(dim, dim, 64)
        pipe_fixed_mlp = Pipeline(emit_fixed_mlp, a2s, env, recv_fixed_mlp)
        train_emitter(pipe_fixed_mlp, cfg_hard)

        mse_fixed_mlp = evaluate(pipe_fixed_mlp, dim)
        ci_fixed_mlp, ci_std_mlp = compute_ci(emit_fixed_mlp, M_inv, dim)
        j_minv_mlp, j_i_mlp = compute_jacobian_analysis(emit_fixed_mlp, M_inv, dim)
        results["fixed_mlp"].append({
            "mse": mse_fixed_mlp, "ci": ci_fixed_mlp, "ci_std": ci_std_mlp,
            "j_dist_minv": j_minv_mlp, "j_dist_i": j_i_mlp,
        })
        print(f"    MSE={mse_fixed_mlp:.6f}, C_i={ci_fixed_mlp:.4f}, "
              f"J→M⁻¹={j_minv_mlp:.3f}, J→I={j_i_mlp:.3f}")

        # === 3. Fixed Linear Receiver ===
        print("  Fixed Linear Receiver:")
        recv_fixed_lin = FixedLinearReceiver(dim, dim, seed=300 + rot_idx)
        recv_fixed_lin.requires_grad_(False)

        torch.manual_seed(42)
        emit_fixed_lin = Emitter(dim, dim, 64)
        pipe_fixed_lin = Pipeline(emit_fixed_lin, a2s, env, recv_fixed_lin)
        train_emitter(pipe_fixed_lin, cfg_hard)

        mse_fixed_lin = evaluate(pipe_fixed_lin, dim)
        ci_fixed_lin, ci_std_lin = compute_ci(emit_fixed_lin, M_inv, dim)
        j_minv_lin, j_i_lin = compute_jacobian_analysis(emit_fixed_lin, M_inv, dim)
        results["fixed_linear"].append({
            "mse": mse_fixed_lin, "ci": ci_fixed_lin, "ci_std": ci_std_lin,
            "j_dist_minv": j_minv_lin, "j_dist_i": j_i_lin,
        })
        print(f"    MSE={mse_fixed_lin:.6f}, C_i={ci_fixed_lin:.4f}, "
              f"J→M⁻¹={j_minv_lin:.3f}, J→I={j_i_lin:.3f}")

    return results, dim, kappa


def plot_results(results, dim, kappa, output_path):
    regimes = ["trained", "fixed_mlp", "fixed_linear"]
    labels = ["Trained\nReceiver", "Fixed Random\nMLP Receiver", "Fixed Linear\nReceiver"]
    colors = ["#2196F3", "#E91E63", "#FF9800"]

    def mean_metric(regime, key):
        return np.mean([r[key] for r in results[regime]])

    def std_metric(regime, key):
        return np.std([r[key] for r in results[regime]])

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # --- Row 1: Key metrics ---
    # Panel 1: MSE
    ax = axes[0, 0]
    for i, (reg, label) in enumerate(zip(regimes, labels)):
        m = mean_metric(reg, "mse")
        s = std_metric(reg, "mse")
        ax.bar(i, m, yerr=s, color=colors[i], alpha=0.85, capsize=6)
        ax.text(i, m + s + m * 0.05, f"{m:.5f}", ha="center", fontsize=10,
                fontweight="bold")
    ax.set_xticks(range(len(regimes)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Test MSE")
    ax.set_yscale("log")
    ax.set_title("Reconstruction Quality\n(can the Emitter compensate?)")
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 2: C_i
    ax = axes[0, 1]
    for i, (reg, label) in enumerate(zip(regimes, labels)):
        m = mean_metric(reg, "ci")
        s = std_metric(reg, "ci")
        ax.bar(i, m, yerr=s, color=colors[i], alpha=0.85, capsize=6)
        ax.text(i, max(m + s, 0) + 0.02, f"{m:.3f}", ha="center", fontsize=11,
                fontweight="bold")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xticks(range(len(regimes)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("C_i (cosine alignment)")
    ax.set_title("Coordination Quality\n(does Emitter learn optimal action?)")
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 3: Jacobian distance to M⁻¹ vs I
    ax = axes[0, 2]
    x = np.arange(len(regimes))
    width = 0.35
    for i, reg in enumerate(regimes):
        d_minv = mean_metric(reg, "j_dist_minv")
        d_i = mean_metric(reg, "j_dist_i")
        ax.bar(i - width/2, d_minv, width, color=colors[i], alpha=0.85,
               label="J → M⁻¹" if i == 0 else None)
        ax.bar(i + width/2, d_i, width, color=colors[i], alpha=0.4,
               label="J → I" if i == 0 else None, hatch="//")
        ax.text(i - width/2, d_minv + 0.1, f"{d_minv:.1f}", ha="center", fontsize=8)
        ax.text(i + width/2, d_i + 0.1, f"{d_i:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Frobenius Distance")
    ax.set_title("Emitter Jacobian Comparison\n(solid=M⁻¹, hatched=Identity)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # --- Row 2: Per-rotation scatter ---
    for col, (key, ylabel, title) in enumerate([
        ("mse", "Test MSE", "Per-Rotation MSE"),
        ("ci", "C_i", "Per-Rotation C_i"),
        ("j_dist_minv", "||J - M⁻¹||", "Per-Rotation Jacobian → M⁻¹"),
    ]):
        ax = axes[1, col]
        for i, reg in enumerate(regimes):
            vals = [r[key] for r in results[reg]]
            jitter = np.random.default_rng(42 + i).uniform(-0.15, 0.15, len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals, color=colors[i],
                       s=60, alpha=0.7, edgecolors="white", linewidth=0.5, zorder=3)
            ax.errorbar(i, np.mean(vals), yerr=np.std(vals), fmt="D",
                        color="black", markersize=6, capsize=5, capthick=1.5, zorder=4)
        ax.set_xticks(range(len(regimes)))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if key == "mse":
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"Fixed Receiver Experiment (dim={dim}, κ={kappa:.0f}, SiLU)\n"
        f"Does removing Receiver pre-training force the Emitter to learn the channel?",
        fontsize=14, fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved: {output_path}")


def print_summary(results, dim, kappa):
    regimes = ["trained", "fixed_mlp", "fixed_linear"]
    labels = ["Trained Recv", "Fixed MLP Recv", "Fixed Linear Recv"]

    print("\n" + "=" * 70)
    print("SUMMARY: Fixed Receiver Experiment")
    print("=" * 70)

    print(f"\n  {'Regime':<20} | {'MSE':>12} | {'C_i':>8} | {'J→M⁻¹':>8} | {'J→I':>8}")
    print(f"  {'-'*20}-+-{'-'*12}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")

    for reg, label in zip(regimes, labels):
        mse = np.mean([r["mse"] for r in results[reg]])
        ci = np.mean([r["ci"] for r in results[reg]])
        j_minv = np.mean([r["j_dist_minv"] for r in results[reg]])
        j_i = np.mean([r["j_dist_i"] for r in results[reg]])
        print(f"  {label:<20} | {mse:>12.6f} | {ci:>8.4f} | {j_minv:>8.3f} | {j_i:>8.3f}")

    # Key comparison
    trained_ci = np.mean([r["ci"] for r in results["trained"]])
    fixed_mlp_ci = np.mean([r["ci"] for r in results["fixed_mlp"]])
    fixed_lin_ci = np.mean([r["ci"] for r in results["fixed_linear"]])

    print(f"\n  KEY QUESTION: Does the Emitter learn the channel when Receiver is fixed?")
    if abs(fixed_mlp_ci) > abs(trained_ci) * 2 or abs(fixed_lin_ci) > abs(trained_ci) * 2:
        print(f"  → YES: Fixed receiver forces higher C_i ({max(abs(fixed_mlp_ci), abs(fixed_lin_ci)):.3f} vs {abs(trained_ci):.3f})")
    else:
        print(f"  → INCONCLUSIVE or NO: C_i remains similar across regimes")

    trained_ji = np.mean([r["j_dist_i"] for r in results["trained"]])
    fixed_mlp_ji = np.mean([r["j_dist_i"] for r in results["fixed_mlp"]])
    if fixed_mlp_ji > trained_ji * 1.5:
        print(f"  → Emitter Jacobian moves AWAY from identity with fixed receiver")
        print(f"    (dist to I: {fixed_mlp_ji:.2f} vs {trained_ji:.2f})")
    else:
        print(f"  → Emitter Jacobian stays near identity regardless of receiver regime")


def main():
    os.makedirs("results", exist_ok=True)
    results, dim, kappa = run_experiment(dim=8, n_rotations=4)
    plot_results(results, dim, kappa, "results/obj-026-fixed-receiver.png")
    print_summary(results, dim, kappa)

    # Save raw results
    serializable = {}
    for regime, runs in results.items():
        serializable[regime] = runs
    with open("results/obj-026-fixed-receiver.json", "w") as f:
        json.dump(serializable, f, indent=2)
    print("Saved: results/obj-026-fixed-receiver.json")


if __name__ == "__main__":
    main()
