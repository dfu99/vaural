"""Experiment: Validate Rotation Invariance Findings at dim=16.

All rotation experiments (obj-013→023) were at dim=8. The key finding —
SiLU eliminates rotation sensitivity while ReLU doesn't — needs validation
at higher dimensions where:
  - SO(d) is much larger (more possible rotations)
  - The network has more parameters relative to the problem
  - ReLU has more coordinate axes to create bias along

This is a quick validation: ReLU vs SiLU, 4 rotations, κ=10, dim=16.
Reduced training (200 recv + 300 emit epochs, 2k samples) since dim=16
needs more epochs to converge — we're comparing RELATIVE rotation sensitivity,
not absolute MSE quality.
"""

import os
import sys
import time

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from components import ActionToSignal, Environment, Pipeline
from train import pretrain_receiver, train_emitter
from experiments.pure_rotational_invariance import make_orthogonal, make_channel_from_svd


class FlexMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, activation="silu"):
        super().__init__()
        act_map = {"relu": nn.ReLU(), "silu": nn.SiLU()}
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), act_map[activation],
            nn.Linear(hidden_dim, hidden_dim), act_map[activation],
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def train_and_eval(a2s, env, cfg, activation, hidden_dim, seed=42):
    torch.manual_seed(seed)
    receiver = FlexMLP(cfg.signal_dim, cfg.sound_dim, hidden_dim, activation)
    recv_losses = pretrain_receiver(a2s, env, receiver, cfg)
    receiver.requires_grad_(False)

    emitter = FlexMLP(cfg.sound_dim, cfg.action_dim, hidden_dim, activation)
    pipeline = Pipeline(emitter, a2s, env, receiver)
    emit_losses = train_emitter(pipeline, cfg)

    torch.manual_seed(99)
    sounds = torch.randn(1000, cfg.sound_dim)
    pipeline.eval()
    with torch.no_grad():
        mse = (pipeline(sounds) - sounds).pow(2).mean().item()
    return mse


def run_experiment():
    print("=" * 60)
    print("EXPERIMENT: dim=16 Rotation Invariance Validation")
    print("Does SiLU advantage hold at higher dimensions?")
    print("=" * 60)

    dim = 16
    hidden_dim = 128  # larger hidden for dim=16
    n_rotations = 4
    sigmas = torch.logspace(0, -1, dim)  # κ=10
    kappa = (sigmas[0] / sigmas[-1]).item()

    cfg = Config(
        sound_dim=dim, action_dim=dim, signal_dim=dim, hidden_dim=hidden_dim,
        receiver_lr=1e-3, receiver_epochs=200,
        receiver_samples=2000, receiver_batch_size=64,
        emitter_lr=1e-3, emitter_epochs=300,
        emitter_samples=2000, emitter_batch_size=64,
        plot_every=9999,
    )

    activations = ["relu", "silu"]
    results = {}

    for act in activations:
        print(f"\n{'='*40}")
        print(f"Activation: {act.upper()}")
        print(f"{'='*40}")

        mses = []
        for rot_idx in range(n_rotations):
            seed_u = 5000 + rot_idx * 7
            seed_v = 6000 + rot_idx * 13
            a2s, env, M = make_channel_from_svd(dim, sigmas, seed_u, seed_v)

            t0 = time.time()
            mse = train_and_eval(a2s, env, cfg, act, hidden_dim, seed=42)
            elapsed = time.time() - t0
            mses.append(mse)
            print(f"  Rot {rot_idx+1}/{n_rotations}: MSE={mse:.6f} ({elapsed:.0f}s)")

        mses = np.array(mses)
        cv = mses.std() / mses.mean() if mses.mean() > 0 else 0
        results[act] = {
            "mses": mses,
            "mean": mses.mean(),
            "std": mses.std(),
            "cv": cv,
            "min": mses.min(),
            "max": mses.max(),
        }
        print(f"  Mean={mses.mean():.6f} ± {mses.std():.6f} (CV={cv:.1%})")

    return results, dim, kappa


def plot_results(results, dim, kappa, output_path):
    acts = list(results.keys())
    colors = {"relu": "#E91E63", "silu": "#4CAF50"}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    # Panel 1: MSE comparison
    ax = axes[0]
    for i, act in enumerate(acts):
        r = results[act]
        ax.bar(i, r["mean"], yerr=r["std"], color=colors[act], alpha=0.85,
               capsize=8, label=act.upper())
        ax.text(i, r["mean"] + r["std"] + r["mean"] * 0.02,
                f"{r['mean']:.5f}", ha="center", fontsize=11, fontweight="bold")
    ax.set_xticks(range(len(acts)))
    ax.set_xticklabels([a.upper() for a in acts])
    ax.set_ylabel("Test MSE")
    ax.set_title(f"dim={dim}: Reconstruction Quality")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 2: CV comparison
    ax = axes[1]
    for i, act in enumerate(acts):
        r = results[act]
        ax.bar(i, r["cv"], color=colors[act], alpha=0.85)
        ax.text(i, r["cv"] + 0.005, f"{r['cv']:.1%}", ha="center",
                fontsize=12, fontweight="bold")
    ax.set_xticks(range(len(acts)))
    ax.set_xticklabels([a.upper() for a in acts])
    ax.set_ylabel("Coefficient of Variation")
    ax.set_title(f"dim={dim}: Rotation Sensitivity\n(lower = more invariant)")
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 3: Per-rotation scatter
    ax = axes[2]
    for i, act in enumerate(acts):
        mses = results[act]["mses"]
        jitter = np.random.default_rng(42 + i).uniform(-0.15, 0.15, len(mses))
        ax.scatter(np.full_like(mses, i) + jitter, mses, color=colors[act],
                   s=80, alpha=0.7, edgecolors="white", linewidth=0.5, zorder=3)
        ax.errorbar(i, results[act]["mean"], yerr=results[act]["std"],
                    fmt="D", color="black", markersize=7, capsize=6,
                    capthick=1.5, zorder=4)
    ax.set_xticks(range(len(acts)))
    ax.set_xticklabels([a.upper() for a in acts])
    ax.set_ylabel("Test MSE")
    ax.set_title(f"dim={dim}: Per-Rotation Results")
    ax.grid(True, alpha=0.3, axis="y")

    # Comparison with dim=8 results
    fig.suptitle(
        f"dim=16 Rotation Invariance Validation (κ={kappa:.0f}, h={128})\n"
        f"dim=8 reference: ReLU CV=19%, SiLU CV=9% — does the pattern hold?",
        fontsize=13, fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved: {output_path}")


def print_summary(results, dim):
    print("\n" + "=" * 60)
    print(f"SUMMARY: dim={dim} Rotation Invariance Validation")
    print("=" * 60)

    relu = results["relu"]
    silu = results["silu"]

    print(f"\n  {'Metric':<20} | {'ReLU':>12} | {'SiLU':>12}")
    print(f"  {'-'*20}-+-{'-'*12}-+-{'-'*12}")
    print(f"  {'Mean MSE':<20} | {relu['mean']:>12.6f} | {silu['mean']:>12.6f}")
    print(f"  {'Rotation CV':<20} | {relu['cv']:>11.1%} | {silu['cv']:>11.1%}")
    print(f"  {'Max/Min ratio':<20} | {relu['max']/max(relu['min'],1e-10):>11.2f}× | {silu['max']/max(silu['min'],1e-10):>11.2f}×")
    print(f"  {'MSE improvement':<20} | {'':>12} | {relu['mean']/max(silu['mean'],1e-10):>11.1f}×")

    print(f"\n  dim=8 reference (obj-017): ReLU CV=19%, SiLU CV=9%")
    if silu["cv"] < relu["cv"]:
        print(f"  dim={dim} result: ReLU CV={relu['cv']:.1%}, SiLU CV={silu['cv']:.1%}")
        print(f"  → VALIDATED: SiLU advantage holds at dim={dim}")
    else:
        print(f"  dim={dim} result: ReLU CV={relu['cv']:.1%}, SiLU CV={silu['cv']:.1%}")
        print(f"  → NOT VALIDATED at this training budget")


def main():
    os.makedirs("results", exist_ok=True)
    results, dim, kappa = run_experiment()
    plot_results(results, dim, kappa, "results/obj-024-dim16-rotation.png")
    print_summary(results, dim)


if __name__ == "__main__":
    main()
