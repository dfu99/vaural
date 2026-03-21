"""Experiment: C_i with Full Pipeline Inverse.

obj-026 showed the Emitter learns (Receiver·Env·A2S)⁻¹, not M⁻¹ = (Env·A2S)⁻¹.
So C_i = cos(emitter(s), M⁻¹·s) measures the wrong thing.

Corrected metric:
  C_i_full = cos(emitter(s), P⁻¹·s)
  where P = Receiver ∘ Environment ∘ ActionToSignal (full pipeline minus Emitter)

For the trained Receiver case, P⁻¹ should be close to identity (since the
Receiver already inverts the channel), so C_i_full ≈ cos(emitter(s), s).

For the fixed linear Receiver case, P is a different linear transform, and
the Emitter should learn P⁻¹ to achieve reconstruction.

We test both regimes and compare C_i (channel-only) vs C_i_full (full pipeline).
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
from experiments.fixed_receiver import FixedLinearReceiver


def compute_metrics(emitter, M_inv, P_inv, dim, n_samples=2000, seed=99):
    """Compute both C_i variants and Jacobian distances."""
    torch.manual_seed(seed)
    sounds = torch.randn(n_samples, dim)
    emitter.eval()
    with torch.no_grad():
        emit_out = emitter(sounds)
        optimal_channel = (M_inv @ sounds.T).T      # M⁻¹·s
        optimal_pipeline = (P_inv @ sounds.T).T      # P⁻¹·s

        ci_channel = nn.functional.cosine_similarity(emit_out, optimal_channel, dim=-1)
        ci_pipeline = nn.functional.cosine_similarity(emit_out, optimal_pipeline, dim=-1)

        # Also measure magnitude ratios
        mag_ratio_channel = (emit_out.norm(dim=-1) / optimal_channel.norm(dim=-1).clamp(min=1e-8)).mean().item()
        mag_ratio_pipeline = (emit_out.norm(dim=-1) / optimal_pipeline.norm(dim=-1).clamp(min=1e-8)).mean().item()

        # MSE if we used emitter output directly vs optimal
        mse_emit = (emit_out - optimal_pipeline).pow(2).mean().item()

    emitter.train()
    return {
        "ci_channel": ci_channel.mean().item(),
        "ci_channel_std": ci_channel.std().item(),
        "ci_pipeline": ci_pipeline.mean().item(),
        "ci_pipeline_std": ci_pipeline.std().item(),
        "mag_ratio_channel": mag_ratio_channel,
        "mag_ratio_pipeline": mag_ratio_pipeline,
        "mse_to_optimal": mse_emit,
    }


def evaluate(pipeline, dim, n_test=2000, seed=99):
    torch.manual_seed(seed)
    sounds = torch.randn(n_test, dim)
    pipeline.eval()
    with torch.no_grad():
        mse = (pipeline(sounds) - sounds).pow(2).mean().item()
    pipeline.train()
    return mse


def compute_P_matrix(a2s, env, receiver, dim):
    """Compute the effective linear transform P = Receiver ∘ Env ∘ A2S.

    For a linear receiver (FixedLinearReceiver), P is exactly:
      P = receiver.weight @ env.weight @ a2s.weight

    For a trained MLP receiver, we linearize around zero (Jacobian at origin).
    """
    if isinstance(receiver, FixedLinearReceiver):
        P = receiver.weight @ env.weight @ a2s.weight
    else:
        # Linearize: compute Jacobian of receiver at the expected input
        # Use multiple points and average for a better approximation
        receiver.eval()
        n_pts = 100
        torch.manual_seed(42)
        test_sounds = torch.randn(n_pts, dim)
        with torch.no_grad():
            test_received = env(a2s(test_sounds))

        # Compute Jacobian at each point
        Js = []
        for i in range(min(20, n_pts)):
            x = test_received[i:i+1].requires_grad_(True)
            y = receiver(x)
            J = []
            for j in range(dim):
                receiver.zero_grad()
                if x.grad is not None:
                    x.grad.zero_()
                y[0, j].backward(retain_graph=True)
                J.append(x.grad[0].clone())
            Js.append(torch.stack(J))

        J_recv = torch.stack(Js).mean(dim=0)
        P = J_recv @ env.weight @ a2s.weight
        receiver.train()

    return P


def run_experiment(dim=8, n_rotations=4):
    print("=" * 60)
    print("EXPERIMENT: C_i with Full Pipeline Inverse")
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

    cfg_hard = Config(
        sound_dim=dim, action_dim=dim, signal_dim=dim, hidden_dim=64,
        receiver_lr=1e-3, receiver_epochs=200,
        receiver_samples=2000, receiver_batch_size=64,
        emitter_lr=1e-3, emitter_epochs=1000,
        emitter_samples=2000, emitter_batch_size=64,
        plot_every=9999,
    )

    regimes = {}

    for rot_idx in range(n_rotations):
        seed_u = 1000 + rot_idx * 7
        seed_v = 2000 + rot_idx * 13
        a2s, env, M = make_channel_from_svd(dim, sigmas, seed_u, seed_v)
        M_inv = torch.linalg.inv(M)

        print(f"\n--- Rotation {rot_idx+1}/{n_rotations} ---")

        for regime_name, recv_factory, use_cfg in [
            ("trained", lambda: _train_receiver(a2s, env, dim, cfg), cfg),
            ("fixed_linear", lambda: FixedLinearReceiver(dim, dim, seed=300 + rot_idx), cfg_hard),
        ]:
            print(f"\n  {regime_name}:")
            receiver = recv_factory()
            receiver.requires_grad_(False)

            # Compute P and P⁻¹
            P = compute_P_matrix(a2s, env, receiver, dim)
            try:
                P_inv = torch.linalg.inv(P)
            except Exception:
                print(f"    P is singular, skipping")
                continue

            # Train Emitter
            torch.manual_seed(42)
            emitter = Emitter(dim, dim, 64)
            pipeline = Pipeline(emitter, a2s, env, receiver)
            train_emitter(pipeline, use_cfg)

            # Evaluate
            mse = evaluate(pipeline, dim)
            metrics = compute_metrics(emitter, M_inv, P_inv, dim)
            metrics["mse"] = mse

            print(f"    MSE={mse:.6f}")
            print(f"    C_i(channel) = {metrics['ci_channel']:.4f}")
            print(f"    C_i(pipeline) = {metrics['ci_pipeline']:.4f}")
            print(f"    mag_ratio(channel) = {metrics['mag_ratio_channel']:.3f}")
            print(f"    mag_ratio(pipeline) = {metrics['mag_ratio_pipeline']:.3f}")

            if regime_name not in regimes:
                regimes[regime_name] = []
            regimes[regime_name].append(metrics)

    return regimes, dim, kappa


def _train_receiver(a2s, env, dim, cfg):
    torch.manual_seed(42)
    recv = Receiver(dim, dim, 64)
    pretrain_receiver(a2s, env, recv, cfg)
    return recv


def plot_results(regimes, dim, kappa, output_path):
    regime_names = list(regimes.keys())
    colors = {"trained": "#2196F3", "fixed_linear": "#FF9800"}

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # Metric pairs to compare
    metric_pairs = [
        ("ci_channel", "ci_pipeline", "C_i: Channel vs Pipeline Inverse",
         "cos(emit, M⁻¹·s)", "cos(emit, P⁻¹·s)"),
        ("mag_ratio_channel", "mag_ratio_pipeline", "Magnitude Ratio: Channel vs Pipeline",
         "||emit|| / ||M⁻¹·s||", "||emit|| / ||P⁻¹·s||"),
    ]

    for col, (key_old, key_new, title, label_old, label_new) in enumerate(metric_pairs):
        ax = axes[0, col]
        x = np.arange(len(regime_names))
        width = 0.35

        for i, reg in enumerate(regime_names):
            old_vals = [r[key_old] for r in regimes[reg]]
            new_vals = [r[key_new] for r in regimes[reg]]
            ax.bar(i - width/2, np.mean(old_vals), width, yerr=np.std(old_vals),
                   color=colors[reg], alpha=0.5, capsize=5, hatch="//",
                   label=f"{label_old}" if i == 0 else None)
            ax.bar(i + width/2, np.mean(new_vals), width, yerr=np.std(new_vals),
                   color=colors[reg], alpha=0.85, capsize=5,
                   label=f"{label_new}" if i == 0 else None)
            ax.text(i - width/2, np.mean(old_vals) + np.std(old_vals) + 0.02,
                    f"{np.mean(old_vals):.3f}", ha="center", fontsize=9)
            ax.text(i + width/2, np.mean(new_vals) + np.std(new_vals) + 0.02,
                    f"{np.mean(new_vals):.3f}", ha="center", fontsize=9, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([r.replace("_", "\n") for r in regime_names])
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.axhline(y=0 if "ci" in key_old else 1.0, color="gray", linestyle="--", alpha=0.5)
        ax.grid(True, alpha=0.3, axis="y")

    # Panel 3: MSE comparison
    ax = axes[0, 2]
    for i, reg in enumerate(regime_names):
        mses = [r["mse"] for r in regimes[reg]]
        ax.bar(i, np.mean(mses), yerr=np.std(mses), color=colors[reg],
               alpha=0.85, capsize=6)
        ax.text(i, np.mean(mses) + np.std(mses) + np.mean(mses) * 0.05,
                f"{np.mean(mses):.5f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(range(len(regime_names)))
    ax.set_xticklabels([r.replace("_", "\n") for r in regime_names])
    ax.set_ylabel("Test MSE")
    ax.set_yscale("log")
    ax.set_title("Reconstruction Quality")
    ax.grid(True, alpha=0.3, axis="y")

    # Row 2: Per-rotation scatter
    for col, (key, ylabel, title) in enumerate([
        ("ci_channel", "C_i (channel)", "Per-Rotation: C_i Channel"),
        ("ci_pipeline", "C_i (pipeline)", "Per-Rotation: C_i Pipeline"),
        ("mse", "Test MSE", "Per-Rotation: MSE"),
    ]):
        ax = axes[1, col]
        for i, reg in enumerate(regime_names):
            vals = [r[key] for r in regimes[reg]]
            jitter = np.random.default_rng(42 + i).uniform(-0.15, 0.15, len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals, color=colors[reg],
                       s=60, alpha=0.7, edgecolors="white", linewidth=0.5, zorder=3)
            ax.errorbar(i, np.mean(vals), yerr=np.std(vals), fmt="D",
                        color="black", markersize=6, capsize=5, zorder=4)
        ax.set_xticks(range(len(regime_names)))
        ax.set_xticklabels([r.replace("_", "\n") for r in regime_names])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if key == "mse":
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"C_i Redefined: Full Pipeline Inverse (dim={dim}, κ={kappa:.0f}, SiLU)\n"
        f"C_i_pipeline = cos(emitter(s), P⁻¹·s) where P = Recv∘Env∘A2S",
        fontsize=14, fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved: {output_path}")


def print_summary(regimes):
    print("\n" + "=" * 70)
    print("SUMMARY: C_i Full Pipeline Inverse")
    print("=" * 70)

    print(f"\n  {'Regime':<18} | {'MSE':>10} | {'C_i(chan)':>10} | {'C_i(pipe)':>10} | {'mag(chan)':>10} | {'mag(pipe)':>10}")
    print(f"  {'-'*18}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

    for reg, runs in regimes.items():
        mse = np.mean([r["mse"] for r in runs])
        ci_c = np.mean([r["ci_channel"] for r in runs])
        ci_p = np.mean([r["ci_pipeline"] for r in runs])
        mag_c = np.mean([r["mag_ratio_channel"] for r in runs])
        mag_p = np.mean([r["mag_ratio_pipeline"] for r in runs])
        print(f"  {reg:<18} | {mse:>10.6f} | {ci_c:>10.4f} | {ci_p:>10.4f} | {mag_c:>10.3f} | {mag_p:>10.3f}")

    # Key comparison
    if "trained" in regimes and "fixed_linear" in regimes:
        trained_pipe = np.mean([r["ci_pipeline"] for r in regimes["trained"]])
        fixed_pipe = np.mean([r["ci_pipeline"] for r in regimes["fixed_linear"]])
        print(f"\n  KEY RESULT:")
        print(f"    Trained Receiver C_i(pipeline): {trained_pipe:.4f}")
        print(f"    Fixed Linear C_i(pipeline): {fixed_pipe:.4f}")
        if abs(fixed_pipe) > 0.5:
            print(f"    → Fixed Receiver Emitter strongly aligns with pipeline inverse!")
        elif abs(fixed_pipe) > abs(trained_pipe) * 2:
            print(f"    → Fixed Receiver shows stronger alignment than trained")
        else:
            print(f"    → Both regimes show similar pipeline alignment")


def main():
    os.makedirs("results", exist_ok=True)
    regimes, dim, kappa = run_experiment(dim=8, n_rotations=4)
    plot_results(regimes, dim, kappa, "results/obj-027-ci-full-pipeline.png")
    print_summary(regimes)

    with open("results/obj-027-ci-full-pipeline.json", "w") as f:
        json.dump(regimes, f, indent=2)
    print("Saved: results/obj-027-ci-full-pipeline.json")


if __name__ == "__main__":
    main()
