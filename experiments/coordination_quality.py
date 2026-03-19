"""Experiment: Coordination Quality C_i in Vaural.

Cross-project metric from WorldNN. In vaural terms:
  C_i = E[cos(emitter(s), M⁻¹ · s)]

Where M = env.weight @ a2s.weight is the combined channel transform and
M⁻¹ · s is the optimal action that would produce perfect reconstruction.

Three tests:
  Test 1: C_i vs MSE correlation across channel conditions
  Test 2: C_i trajectory during adaptation (does direction precede magnitude?)
  Test 3: C_i decomposition by singular direction

Uses SiLU activation (project default).
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


def compute_ci(emitter, M_inv, sound_dim, n_samples=2000, seed=99):
    """Compute C_i = E[cos(emitter(s), M⁻¹·s)]."""
    torch.manual_seed(seed)
    sounds = torch.randn(n_samples, sound_dim)
    emitter.eval()
    with torch.no_grad():
        emitter_output = emitter(sounds)
        optimal_action = (M_inv @ sounds.T).T  # M⁻¹ · s
        cos_sim = nn.functional.cosine_similarity(emitter_output, optimal_action, dim=-1)
    emitter.train()
    return {
        "ci_mean": cos_sim.mean().item(),
        "ci_std": cos_sim.std().item(),
        "ci_positive_frac": (cos_sim > 0).float().mean().item(),
        "emitter_norm": emitter_output.norm(dim=-1).mean().item(),
        "optimal_norm": optimal_action.norm(dim=-1).mean().item(),
        "magnitude_ratio": (emitter_output.norm(dim=-1) / optimal_action.norm(dim=-1).clamp(min=1e-8)).mean().item(),
    }


def compute_mse(pipeline, sound_dim, n_samples=2000, seed=99):
    torch.manual_seed(seed)
    sounds = torch.randn(n_samples, sound_dim)
    pipeline.eval()
    with torch.no_grad():
        mse = (pipeline(sounds) - sounds).pow(2).mean().item()
    pipeline.train()
    return mse


def test1_ci_vs_mse(dim=8, n_rotations=6):
    """Test 1: Does C_i predict MSE across channel conditions?"""
    print("\n" + "=" * 60)
    print("TEST 1: C_i vs MSE Correlation")
    print("=" * 60)

    spectra = {
        "κ=1": torch.ones(dim),
        "κ=3": torch.logspace(0, -0.5, dim),
        "κ=10": torch.logspace(0, -1, dim),
        "κ=30": torch.logspace(0, -1.5, dim),
        "κ=100": torch.logspace(0, -2, dim),
    }

    cfg = Config(
        sound_dim=dim, action_dim=dim, signal_dim=dim, hidden_dim=64,
        receiver_lr=1e-3, receiver_epochs=200,
        receiver_samples=2000, receiver_batch_size=64,
        emitter_lr=1e-3, emitter_epochs=300,
        emitter_samples=2000, emitter_batch_size=64,
        plot_every=9999,
    )

    all_ci = []
    all_mse = []
    all_labels = []
    all_kappas = []
    all_mag_ratios = []

    for spec_name, sigmas in spectra.items():
        kappa = (sigmas[0] / sigmas[-1]).item()
        for rot_idx in range(n_rotations):
            seed_u = 1000 + rot_idx * 7
            seed_v = 2000 + rot_idx * 13
            a2s, env, M = make_channel_from_svd(dim, sigmas, seed_u, seed_v)
            M_inv = torch.linalg.inv(M)

            torch.manual_seed(42)
            receiver = Receiver(cfg.signal_dim, cfg.sound_dim, cfg.hidden_dim)
            pretrain_receiver(a2s, env, receiver, cfg)
            receiver.requires_grad_(False)

            emitter = Emitter(cfg.sound_dim, cfg.action_dim, cfg.hidden_dim)
            pipeline = Pipeline(emitter, a2s, env, receiver)
            train_emitter(pipeline, cfg)

            ci = compute_ci(emitter, M_inv, cfg.sound_dim)
            mse = compute_mse(pipeline, cfg.sound_dim)

            all_ci.append(ci["ci_mean"])
            all_mse.append(mse)
            all_labels.append(spec_name)
            all_kappas.append(kappa)
            all_mag_ratios.append(ci["magnitude_ratio"])

            print(f"  {spec_name} rot {rot_idx+1}: C_i={ci['ci_mean']:.4f}, "
                  f"MSE={mse:.6f}, mag_ratio={ci['magnitude_ratio']:.3f}")

    return {
        "ci": np.array(all_ci),
        "mse": np.array(all_mse),
        "labels": all_labels,
        "kappas": np.array(all_kappas),
        "mag_ratios": np.array(all_mag_ratios),
    }


def test2_ci_trajectory(dim=8, n_rotations=4):
    """Test 2: C_i trajectory during adaptation."""
    print("\n" + "=" * 60)
    print("TEST 2: C_i Trajectory During Adaptation")
    print("=" * 60)

    sigmas = torch.logspace(0, -1, dim)  # κ=10
    cfg = Config(
        sound_dim=dim, action_dim=dim, signal_dim=dim, hidden_dim=64,
        receiver_lr=1e-3, receiver_epochs=100,
        receiver_samples=1000, receiver_batch_size=64,
        emitter_lr=1e-3, emitter_epochs=200,
        emitter_samples=1000, emitter_batch_size=64,
        plot_every=9999,
    )

    checkpoints = [0, 5, 10, 20, 50, 100, 150, 200]

    # Pre-train Receiver on M₁
    seed_u1, seed_v1 = 1000, 2000
    a2s1, env1, M1 = make_channel_from_svd(dim, sigmas, seed_u1, seed_v1)
    torch.manual_seed(42)
    receiver_m1 = Receiver(cfg.signal_dim, cfg.sound_dim, cfg.hidden_dim)
    pretrain_receiver(a2s1, env1, receiver_m1, cfg)
    receiver_m1.requires_grad_(False)

    adaptation_trajectories = []

    for rot_idx in range(n_rotations):
        seed_u2 = 3000 + rot_idx * 11
        seed_v2 = 4000 + rot_idx * 17
        a2s2, env2, M2 = make_channel_from_svd(dim, sigmas, seed_u2, seed_v2)
        M2_inv = torch.linalg.inv(M2)

        # Fresh Emitter on M₂ with frozen M₁-Receiver
        torch.manual_seed(42)
        emitter = Emitter(cfg.sound_dim, cfg.action_dim, cfg.hidden_dim)
        pipeline = Pipeline(emitter, a2s2, env2, receiver_m1)

        # Freeze everything except emitter
        pipeline.receiver.requires_grad_(False)
        pipeline.action_to_signal.requires_grad_(False)
        pipeline.environment.requires_grad_(False)

        sounds = torch.randn(cfg.emitter_samples, cfg.sound_dim)
        optimizer = torch.optim.Adam(pipeline.emitter.parameters(), lr=cfg.emitter_lr)
        loss_fn = nn.MSELoss()

        trajectory = {"ci": [], "mse": [], "mag_ratio": [], "epochs": []}
        checkpoint_set = set(checkpoints)

        # Epoch 0
        if 0 in checkpoint_set:
            ci = compute_ci(emitter, M2_inv, cfg.sound_dim)
            mse = compute_mse(pipeline, cfg.sound_dim)
            trajectory["ci"].append(ci["ci_mean"])
            trajectory["mse"].append(mse)
            trajectory["mag_ratio"].append(ci["magnitude_ratio"])
            trajectory["epochs"].append(0)

        for epoch in range(cfg.emitter_epochs):
            perm = torch.randperm(sounds.size(0))
            for i in range(0, sounds.size(0), cfg.emitter_batch_size):
                idx = perm[i:i + cfg.emitter_batch_size]
                decoded = pipeline(sounds[idx])
                loss = loss_fn(decoded, sounds[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            ep = epoch + 1
            if ep in checkpoint_set:
                ci = compute_ci(emitter, M2_inv, cfg.sound_dim)
                mse = compute_mse(pipeline, cfg.sound_dim)
                trajectory["ci"].append(ci["ci_mean"])
                trajectory["mse"].append(mse)
                trajectory["mag_ratio"].append(ci["magnitude_ratio"])
                trajectory["epochs"].append(ep)
                print(f"  Rot {rot_idx+1} epoch {ep}: C_i={ci['ci_mean']:.4f}, "
                      f"MSE={mse:.6f}, mag={ci['magnitude_ratio']:.3f}")

        adaptation_trajectories.append(trajectory)

    return {"trajectories": adaptation_trajectories, "checkpoints": checkpoints}


def test3_ci_by_direction(dim=8, n_rotations=4):
    """Test 3: C_i decomposed by singular direction."""
    print("\n" + "=" * 60)
    print("TEST 3: C_i by Singular Direction")
    print("=" * 60)

    sigmas = torch.logspace(0, -1, dim)  # κ=10
    cfg = Config(
        sound_dim=dim, action_dim=dim, signal_dim=dim, hidden_dim=64,
        receiver_lr=1e-3, receiver_epochs=200,
        receiver_samples=2000, receiver_batch_size=64,
        emitter_lr=1e-3, emitter_epochs=300,
        emitter_samples=2000, emitter_batch_size=64,
        plot_every=9999,
    )

    all_per_dir_ci = []

    for rot_idx in range(n_rotations):
        seed_u = 1000 + rot_idx * 7
        seed_v = 2000 + rot_idx * 13
        a2s, env, M = make_channel_from_svd(dim, sigmas, seed_u, seed_v)
        M_inv = torch.linalg.inv(M)

        # Get right singular vectors of M
        U, S, Vh = torch.linalg.svd(M)
        V = Vh.T  # columns are right singular vectors

        torch.manual_seed(42)
        receiver = Receiver(cfg.signal_dim, cfg.sound_dim, cfg.hidden_dim)
        pretrain_receiver(a2s, env, receiver, cfg)
        receiver.requires_grad_(False)

        emitter = Emitter(cfg.sound_dim, cfg.action_dim, cfg.hidden_dim)
        pipeline = Pipeline(emitter, a2s, env, receiver)
        train_emitter(pipeline, cfg)

        # C_i per singular direction
        emitter.eval()
        per_dir_ci = []
        with torch.no_grad():
            for k in range(dim):
                vk = V[:, k]
                # Generate inputs along this singular direction with varying magnitudes
                scales = torch.randn(500, 1)
                inputs = scales * vk.unsqueeze(0)  # (500, dim)
                emitter_out = emitter(inputs)
                optimal_out = (M_inv @ inputs.T).T
                cos = nn.functional.cosine_similarity(emitter_out, optimal_out, dim=-1)
                per_dir_ci.append(cos.mean().item())

        all_per_dir_ci.append(per_dir_ci)
        print(f"  Rot {rot_idx+1}: C_i by σ: " +
              ", ".join(f"σ={s:.2f}→{c:.3f}" for s, c in zip(sigmas, per_dir_ci)))

    return {
        "per_dir_ci": np.array(all_per_dir_ci),  # (n_rotations, dim)
        "sigmas": sigmas.numpy(),
    }


def plot_all(test1, test2, test3, dim, output_path):
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

    # --- Test 1: C_i vs MSE scatter ---
    ax = fig.add_subplot(gs[0, 0])
    ci = test1["ci"]
    mse = test1["mse"]
    kappas = test1["kappas"]

    # Color by kappa
    unique_kappas = sorted(set(kappas))
    cmap = plt.cm.viridis
    for i, k in enumerate(unique_kappas):
        mask = kappas == k
        color = cmap(i / max(len(unique_kappas) - 1, 1))
        ax.scatter(ci[mask], mse[mask], c=[color], s=60, alpha=0.8,
                   edgecolors="white", linewidth=0.5, label=f"κ={k:.0f}")

    # Correlation
    r = np.corrcoef(ci, np.log10(mse + 1e-10))[0, 1]
    ax.set_xlabel("C_i (cosine alignment)")
    ax.set_ylabel("Test MSE")
    ax.set_yscale("log")
    ax.set_title(f"Test 1: C_i vs MSE\nR(C_i, log MSE) = {r:.3f}")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    # --- Test 1b: Magnitude ratio vs MSE ---
    ax = fig.add_subplot(gs[0, 1])
    mag = test1["mag_ratios"]
    for i, k in enumerate(unique_kappas):
        mask = kappas == k
        color = cmap(i / max(len(unique_kappas) - 1, 1))
        ax.scatter(mag[mask], mse[mask], c=[color], s=60, alpha=0.8,
                   edgecolors="white", linewidth=0.5, label=f"κ={k:.0f}")
    ax.axvline(x=1.0, color="gray", linestyle="--", alpha=0.5)
    r_mag = np.corrcoef(mag, np.log10(mse + 1e-10))[0, 1]
    ax.set_xlabel("Magnitude Ratio (emitter/optimal)")
    ax.set_ylabel("Test MSE")
    ax.set_yscale("log")
    ax.set_title(f"Magnitude Scaling vs MSE\nR(mag, log MSE) = {r_mag:.3f}")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    # --- Test 2: C_i trajectory ---
    ax = fig.add_subplot(gs[0, 2])
    trajs = test2["trajectories"]
    ax2 = ax.twinx()

    for i, t in enumerate(trajs):
        alpha = 0.7 if i == 0 else 0.3
        ax.plot(t["epochs"], t["ci"], "o-", color="#4CAF50", alpha=alpha,
                linewidth=1.5, markersize=4,
                label="C_i" if i == 0 else None)
        ax2.plot(t["epochs"], t["mse"], "s--", color="#E91E63", alpha=alpha,
                 linewidth=1.5, markersize=4,
                 label="MSE" if i == 0 else None)

    ax.set_xlabel("Emitter Training Epochs")
    ax.set_ylabel("C_i (cosine alignment)", color="#4CAF50")
    ax2.set_ylabel("Test MSE", color="#E91E63")
    ax2.set_yscale("log")
    ax.set_title("Test 2: C_i Trajectory During Adaptation\n(does alignment precede reconstruction?)")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="center right")
    ax.grid(True, alpha=0.3)

    # --- Test 2b: Magnitude ratio trajectory ---
    ax = fig.add_subplot(gs[1, 0])
    for i, t in enumerate(trajs):
        alpha = 0.7 if i == 0 else 0.3
        ax.plot(t["epochs"], t["mag_ratio"], "D-", color="#FF9800", alpha=alpha,
                linewidth=1.5, markersize=4)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Optimal magnitude")
    ax.set_xlabel("Emitter Training Epochs")
    ax.set_ylabel("Magnitude Ratio (emitter/optimal)")
    ax.set_title("Magnitude Scaling During Adaptation\n(1.0 = optimal)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Test 3: C_i by singular direction ---
    ax = fig.add_subplot(gs[1, 1])
    per_dir = test3["per_dir_ci"]  # (n_rot, dim)
    sigmas_arr = test3["sigmas"]
    mean_ci = per_dir.mean(axis=0)
    std_ci = per_dir.std(axis=0)

    ax.errorbar(range(dim), mean_ci, yerr=std_ci, fmt="o-", color="#2196F3",
                markersize=6, capsize=4, linewidth=2, label="C_i per direction")
    ax2 = ax.twinx()
    ax2.plot(range(dim), sigmas_arr, "s--", color="#FF9800", markersize=5,
             alpha=0.7, label="σ_k")
    ax.set_xlabel("Singular Direction k")
    ax.set_ylabel("C_i (mean ± std)", color="#2196F3")
    ax2.set_ylabel("Singular Value σ_k", color="#FF9800")
    ax.set_title("Test 3: C_i by Singular Direction\n(is alignment uniform across directions?)")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Summary panel ---
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")

    # Compute summary stats
    r_ci_mse = np.corrcoef(test1["ci"], np.log10(test1["mse"] + 1e-10))[0, 1]
    r_mag_mse = np.corrcoef(test1["mag_ratios"], np.log10(test1["mse"] + 1e-10))[0, 1]

    # Does C_i rise before MSE drops?
    traj0 = trajs[0]
    ci_half = next((i for i, c in enumerate(traj0["ci"]) if c > 0.5 * traj0["ci"][-1]),
                   len(traj0["ci"]) - 1)
    mse_half = next((i for i, m in enumerate(traj0["mse"][1:], 1)
                     if m < 0.5 * (traj0["mse"][0] + traj0["mse"][-1])),
                    len(traj0["mse"]) - 1)

    # Direction uniformity
    ci_cv_across_dirs = std_ci.mean() / max(mean_ci.mean(), 1e-10)

    summary = (
        f"COORDINATION QUALITY SUMMARY\n"
        f"{'─' * 35}\n\n"
        f"*Test 1: C_i vs MSE*\n"
        f"  R(C_i, log MSE) = {r_ci_mse:.3f}\n"
        f"  R(mag, log MSE) = {r_mag_mse:.3f}\n"
        f"  C_i range: [{test1['ci'].min():.3f}, {test1['ci'].max():.3f}]\n\n"
        f"*Test 2: Adaptation trajectory*\n"
        f"  C_i half-max at checkpoint {ci_half}\n"
        f"  MSE half-min at checkpoint {mse_half}\n"
        f"  {'C_i leads MSE ✓' if ci_half <= mse_half else 'MSE leads C_i'}\n\n"
        f"*Test 3: Per-direction*\n"
        f"  Mean C_i: {mean_ci.mean():.3f}\n"
        f"  CV across directions: {ci_cv_across_dirs:.1%}\n"
        f"  {'Uniform ✓' if ci_cv_across_dirs < 0.1 else 'Non-uniform'}"
    )
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5", alpha=0.9))

    fig.suptitle(
        f"Coordination Quality C_i = cos(emitter(s), M⁻¹·s)  —  dim={dim}, SiLU\n"
        f"Cross-project metric from WorldNN: does alignment predict performance?",
        fontsize=14, fontweight="bold"
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved: {output_path}")


def main():
    os.makedirs("results", exist_ok=True)
    dim = 8

    print("=" * 60)
    print("EXPERIMENT: Coordination Quality C_i")
    print("Cross-project metric validation for vaural")
    print("=" * 60)

    test1 = test1_ci_vs_mse(dim=dim, n_rotations=6)
    test2 = test2_ci_trajectory(dim=dim, n_rotations=4)
    test3 = test3_ci_by_direction(dim=dim, n_rotations=4)

    plot_all(test1, test2, test3, dim, "results/obj-025-coordination-quality.png")

    # Save raw results for cross-project comparison
    results = {
        "test1": {k: v.tolist() if isinstance(v, np.ndarray) else v
                  for k, v in test1.items()},
        "test2": test2,
        "test3": {k: v.tolist() if isinstance(v, np.ndarray) else v
                  for k, v in test3.items()},
    }
    with open("results/obj-025-coordination-quality.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("Saved: results/obj-025-coordination-quality.json")


if __name__ == "__main__":
    main()
