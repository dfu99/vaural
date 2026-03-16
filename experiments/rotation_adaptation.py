"""Experiment: Channel Rotation Adaptation (obj-019).

Previous experiments (obj-013→018) always trained fresh models per channel.
But biologically, the Receiver (auditory cortex) is pre-trained on one acoustic
environment. When the channel changes (new room, different speaker), only the
Emitter (motor system) adapts.

This experiment tests:
  1. Pre-train Receiver on channel M₁ = U₁ Σ V₁^T
  2. Train Emitter on M₁ to convergence (baseline MSE)
  3. Swap channel to M₂ = U₂ Σ V₂^T (same spectrum, different rotations)
  4. Re-train ONLY the Emitter (Receiver frozen from M₁)
  5. Measure: adaptation speed, final MSE, and what the Emitter learns

Key question: Does the Emitter learn a non-trivial pre-rotation to compensate
for the mismatch between the Receiver's training channel and the new channel?

This connects rotational invariance to channel adaptation — the biologically
relevant scenario. We also compare ReLU vs SiLU to test whether smooth
activations enable faster/better adaptation.
"""

import os
import sys

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from components import ActionToSignal, Environment, Receiver, Pipeline
from train import pretrain_receiver, train_emitter


class FlexibleEmitter(nn.Module):
    """Emitter with configurable activation function."""

    def __init__(self, sound_dim, action_dim, hidden_dim, activation="relu"):
        super().__init__()
        act_fn = {"relu": nn.ReLU, "silu": nn.SiLU, "gelu": nn.GELU, "tanh": nn.Tanh}
        act = act_fn[activation]
        self.net = nn.Sequential(
            nn.Linear(sound_dim, hidden_dim),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
            act(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class FlexibleReceiver(nn.Module):
    """Receiver with configurable activation function."""

    def __init__(self, signal_dim, sound_dim, hidden_dim, activation="relu"):
        super().__init__()
        act_fn = {"relu": nn.ReLU, "silu": nn.SiLU, "gelu": nn.GELU, "tanh": nn.Tanh}
        act = act_fn[activation]
        self.net = nn.Sequential(
            nn.Linear(signal_dim, hidden_dim),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
            act(),
            nn.Linear(hidden_dim, sound_dim),
        )

    def forward(self, x):
        return self.net(x)


def make_orthogonal(dim, seed):
    """Random orthogonal matrix via QR decomposition."""
    gen = torch.Generator().manual_seed(seed)
    M = torch.randn(dim, dim, generator=gen)
    Q, R = torch.linalg.qr(M)
    Q = Q * torch.sign(torch.diag(R)).unsqueeze(0)
    return Q


def make_channel_from_svd(dim, sigmas, seed_u, seed_v):
    """Build channel M = U diag(σ) V^T with controlled rotations."""
    U = make_orthogonal(dim, seed_u)
    V = make_orthogonal(dim, seed_v)
    a2s = ActionToSignal(dim, dim, seed=0)
    env = Environment(dim, seed=0)
    a2s.weight.copy_(torch.diag(sigmas) @ V.T)
    env.weight.copy_(U)
    return a2s, env


def compute_jacobian(module, dim, x0=None, eps=1e-4):
    """Finite-difference Jacobian of module at x0."""
    if x0 is None:
        x0 = torch.zeros(1, dim)
    module.eval()
    with torch.no_grad():
        y0 = module(x0).squeeze()
        J = torch.zeros(y0.shape[0], dim)
        for i in range(dim):
            dx = torch.zeros_like(x0)
            dx[0, i] = eps
            y_plus = module(x0 + dx).squeeze()
            J[:, i] = (y_plus - y0) / eps
    return J


def rotation_angle_between(A, B):
    """Angle in degrees between two matrices treated as rotations.

    Uses the Frobenius norm of their difference relative to identity.
    Also computes the actual rotation angle from A^T @ B.
    """
    C = A.T @ B  # relative rotation
    # For orthogonal C, trace = 1 + 2*cos(theta) in 2D, sum of cos(angles) in nD
    cos_angles = (torch.trace(C) - 1) / (A.shape[0] - 1)
    cos_angles = cos_angles.clamp(-1, 1)
    return torch.acos(cos_angles).item() * 180 / np.pi


def pretrain_receiver_flex(a2s, env, receiver, cfg):
    """Pre-train receiver (works with FlexibleReceiver)."""
    sounds = torch.randn(cfg.receiver_samples, cfg.sound_dim)
    with torch.no_grad():
        signals = a2s(sounds)
        received = env(signals)

    optimizer = torch.optim.Adam(receiver.parameters(), lr=cfg.receiver_lr)
    loss_fn = nn.MSELoss()
    losses = []

    for epoch in range(cfg.receiver_epochs):
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


def train_emitter_flex(pipeline, cfg, track_every=10):
    """Train emitter, returning loss curve with finer granularity."""
    pipeline.receiver.requires_grad_(False)
    pipeline.action_to_signal.requires_grad_(False)
    pipeline.environment.requires_grad_(False)

    sounds = torch.randn(cfg.emitter_samples, cfg.sound_dim)
    optimizer = torch.optim.Adam(pipeline.emitter.parameters(), lr=cfg.emitter_lr)
    loss_fn = nn.MSELoss()
    losses = []

    for epoch in range(cfg.emitter_epochs):
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


def evaluate(pipeline, n_test=2000, seed=99):
    """Evaluate pipeline on fresh test data."""
    torch.manual_seed(seed)
    sounds = torch.randn(n_test, pipeline.emitter.net[0].in_features)
    pipeline.eval()
    with torch.no_grad():
        recon = pipeline(sounds)
        mse = (recon - sounds).pow(2).mean().item()
    return mse


def run_adaptation_experiment(dim=8, activation="relu", kappa=10,
                               epochs_recv=400, epochs_emit=500,
                               epochs_adapt=500, n_samples=2000,
                               n_rotations=6):
    """Run the channel rotation adaptation experiment.

    Steps:
      1. Train on channel M₁ (reference rotation)
      2. For each new rotation M₂:
         a. Fresh Emitter trained on M₂ with M₁-Receiver (adaptation)
         b. Fresh Emitter trained on M₂ with M₂-Receiver (oracle baseline)
    """
    cfg = Config(
        sound_dim=dim, action_dim=dim, signal_dim=dim, hidden_dim=64,
        receiver_lr=1e-3, receiver_epochs=epochs_recv,
        receiver_samples=n_samples, receiver_batch_size=64,
        emitter_lr=1e-3, emitter_epochs=epochs_emit,
        emitter_samples=n_samples, emitter_batch_size=64,
        plot_every=9999,
    )

    # Build spectrum
    sigmas = torch.logspace(0, -np.log10(kappa), dim)
    print(f"\nSpectrum: σ = [{sigmas[0]:.3f}, ..., {sigmas[-1]:.3f}], κ={kappa}")

    # Reference channel M₁
    seed_u1, seed_v1 = 1000, 2000
    a2s1, env1 = make_channel_from_svd(dim, sigmas, seed_u1, seed_v1)

    # Pre-train Receiver on M₁
    print(f"\n[Step 1] Pre-training Receiver on reference channel M₁ ({activation})...")
    torch.manual_seed(42)
    receiver1 = FlexibleReceiver(dim, dim, 64, activation)
    recv_losses = pretrain_receiver_flex(a2s1, env1, receiver1, cfg)
    receiver1.requires_grad_(False)
    print(f"  Receiver final loss: {recv_losses[-1]:.6f}")

    # Train Emitter on M₁ (baseline)
    print(f"\n[Step 2] Training Emitter on M₁ (baseline)...")
    torch.manual_seed(42)
    emitter1 = FlexibleEmitter(dim, dim, 64, activation)
    pipeline1 = Pipeline(emitter1, a2s1, env1, receiver1)
    emit1_losses = train_emitter_flex(pipeline1, cfg)
    baseline_mse = evaluate(pipeline1)
    print(f"  Baseline MSE on M₁: {baseline_mse:.6f}")

    # Compute Emitter Jacobian on M₁ (should be near-identity)
    J_emit_baseline = compute_jacobian(emitter1, dim)

    results = {
        "baseline_mse": baseline_mse,
        "baseline_emit_losses": emit1_losses,
        "recv_losses": recv_losses,
        "J_emit_baseline": J_emit_baseline,
        "adaptations": [],
    }

    # Now test adaptation to rotated channels
    print(f"\n[Step 3] Testing adaptation to {n_rotations} rotated channels...")

    for rot_idx in range(n_rotations):
        seed_u2 = 3000 + rot_idx * 17
        seed_v2 = 4000 + rot_idx * 23
        a2s2, env2 = make_channel_from_svd(dim, sigmas, seed_u2, seed_v2)

        # Compute rotation angles between M₁ and M₂
        U1 = make_orthogonal(dim, seed_u1)
        V1 = make_orthogonal(dim, seed_v1)
        U2 = make_orthogonal(dim, seed_u2)
        V2 = make_orthogonal(dim, seed_v2)

        print(f"\n  Rotation {rot_idx+1}/{n_rotations}")

        # === Adaptation: new Emitter, same Receiver (from M₁), new channel M₂ ===
        adapt_cfg = Config(
            sound_dim=dim, action_dim=dim, signal_dim=dim, hidden_dim=64,
            receiver_lr=1e-3, receiver_epochs=0,  # no receiver training
            receiver_samples=n_samples, receiver_batch_size=64,
            emitter_lr=1e-3, emitter_epochs=epochs_adapt,
            emitter_samples=n_samples, emitter_batch_size=64,
            plot_every=9999,
        )

        torch.manual_seed(42)
        emitter_adapt = FlexibleEmitter(dim, dim, 64, activation)
        pipeline_adapt = Pipeline(emitter_adapt, a2s2, env2, receiver1)
        adapt_losses = train_emitter_flex(pipeline_adapt, adapt_cfg)
        adapt_mse = evaluate(pipeline_adapt)

        # Compute Emitter Jacobian after adaptation (should show pre-rotation)
        J_emit_adapt = compute_jacobian(emitter_adapt, dim)

        # === Oracle: fresh Receiver on M₂, fresh Emitter on M₂ ===
        torch.manual_seed(42)
        receiver_oracle = FlexibleReceiver(dim, dim, 64, activation)
        recv_oracle_losses = pretrain_receiver_flex(a2s2, env2, receiver_oracle, cfg)
        receiver_oracle.requires_grad_(False)

        torch.manual_seed(42)
        emitter_oracle = FlexibleEmitter(dim, dim, 64, activation)
        pipeline_oracle = Pipeline(emitter_oracle, a2s2, env2, receiver_oracle)
        oracle_losses = train_emitter_flex(pipeline_oracle, cfg)
        oracle_mse = evaluate(pipeline_oracle)

        J_emit_oracle = compute_jacobian(emitter_oracle, dim)

        # Compute the theoretical pre-rotation the Emitter should learn
        # If Receiver learned M₁⁻¹, and channel is now M₂, then:
        #   Receiver(M₂ @ Emitter(x)) ≈ M₁⁻¹ @ M₂ @ Emitter(x) ≈ x
        #   So Emitter should learn M₂⁻¹ @ M₁ = (V₂ Σ⁻¹ U₂^T)(U₁ Σ V₁^T)
        # But wait — Receiver learned to invert the COMBINED channel E@A2S of M₁.
        # With new channel M₂, the Emitter needs: M₁⁻¹ @ M₂ @ Emitter(x) = x
        # i.e., Emitter(x) = M₂⁻¹ @ M₁ @ x
        M1 = env1.weight @ a2s1.weight
        M2 = env2.weight @ a2s2.weight
        target_transform = torch.linalg.solve(M2, M1)

        # Distance of adapted Emitter from the target transform
        J_dist_to_target = torch.norm(J_emit_adapt - target_transform).item()
        J_dist_to_identity = torch.norm(J_emit_adapt - torch.eye(dim)).item()

        result = {
            "rot_idx": rot_idx,
            "adapt_mse": adapt_mse,
            "oracle_mse": oracle_mse,
            "adapt_losses": adapt_losses,
            "oracle_losses": oracle_losses,
            "J_emit_adapt": J_emit_adapt,
            "J_emit_oracle": J_emit_oracle,
            "target_transform": target_transform,
            "J_dist_to_target": J_dist_to_target,
            "J_dist_to_identity": J_dist_to_identity,
            "mse_ratio": adapt_mse / max(oracle_mse, 1e-10),
        }
        results["adaptations"].append(result)

        print(f"    Adapted MSE:  {adapt_mse:.6f}  (Emitter re-trained, Receiver from M₁)")
        print(f"    Oracle MSE:   {oracle_mse:.6f}  (both fresh on M₂)")
        print(f"    Ratio:        {result['mse_ratio']:.2f}×")
        print(f"    J(Emitter) distance to target transform: {J_dist_to_target:.3f}")
        print(f"    J(Emitter) distance to identity:         {J_dist_to_identity:.3f}")

    return results


def plot_results(results_relu, results_silu, dim, kappa, output_path):
    """Visualize adaptation results for ReLU vs SiLU."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    for col, (results, act_name, color) in enumerate([
        (results_relu, "ReLU", "#E91E63"),
        (results_silu, "SiLU", "#2196F3"),
    ]):
        adaptations = results["adaptations"]
        n_rot = len(adaptations)

        # --- Panel 1: Adaptation vs Oracle MSE ---
        ax = axes[0, col]
        adapt_mses = [a["adapt_mse"] for a in adaptations]
        oracle_mses = [a["oracle_mse"] for a in adaptations]
        x = np.arange(n_rot)
        w = 0.35
        ax.bar(x - w/2, adapt_mses, w, label="Adapted (M₁-Receiver)", color=color, alpha=0.8)
        ax.bar(x + w/2, oracle_mses, w, label="Oracle (M₂-Receiver)", color="#4CAF50", alpha=0.8)
        ax.axhline(results["baseline_mse"], color="black", linestyle="--", alpha=0.5,
                    label=f"Baseline (M₁)")
        ax.set_xlabel("Rotation index")
        ax.set_ylabel("Test MSE")
        ax.set_title(f"{act_name}: Adapted vs Oracle MSE")
        ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

        # --- Panel 2: Adaptation loss curves ---
        ax = axes[1, col]
        for i, a in enumerate(adaptations):
            ax.plot(a["adapt_losses"], alpha=0.5, label=f"Rot {i+1}" if i < 4 else None)
        # Plot baseline emitter loss for reference
        ax.plot(results["baseline_emit_losses"], color="black", linewidth=2,
                linestyle="--", alpha=0.7, label="Baseline (M₁)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Training Loss")
        ax.set_title(f"{act_name}: Emitter Adaptation Curves")
        ax.set_yscale("log")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    # --- Panel 3 (top right): Comparative summary ---
    ax = axes[0, 2]
    relu_ratios = [a["mse_ratio"] for a in results_relu["adaptations"]]
    silu_ratios = [a["mse_ratio"] for a in results_silu["adaptations"]]

    positions = [0, 1]
    bp = ax.boxplot([relu_ratios, silu_ratios], positions=positions, widths=0.5,
                     patch_artist=True, showmeans=True,
                     meanprops=dict(marker="D", markerfacecolor="black", markersize=6))
    bp["boxes"][0].set_facecolor("#E91E63")
    bp["boxes"][0].set_alpha(0.5)
    bp["boxes"][1].set_facecolor("#2196F3")
    bp["boxes"][1].set_alpha(0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels(["ReLU", "SiLU"])
    ax.set_ylabel("MSE Ratio (Adapted / Oracle)")
    ax.axhline(1.0, color="green", linestyle="--", alpha=0.5, label="Perfect adaptation")
    ax.set_title("Adaptation Penalty\n(1.0 = no penalty from mismatched Receiver)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # --- Panel 4 (bottom right): Emitter Jacobian analysis ---
    ax = axes[1, 2]
    relu_j_target = [a["J_dist_to_target"] for a in results_relu["adaptations"]]
    relu_j_identity = [a["J_dist_to_identity"] for a in results_relu["adaptations"]]
    silu_j_target = [a["J_dist_to_target"] for a in results_silu["adaptations"]]
    silu_j_identity = [a["J_dist_to_identity"] for a in results_silu["adaptations"]]

    x = np.arange(len(results_relu["adaptations"]))
    w = 0.2
    ax.bar(x - 1.5*w, relu_j_target, w, label="ReLU → target", color="#E91E63", alpha=0.8)
    ax.bar(x - 0.5*w, relu_j_identity, w, label="ReLU → identity", color="#E91E63", alpha=0.4)
    ax.bar(x + 0.5*w, silu_j_target, w, label="SiLU → target", color="#2196F3", alpha=0.8)
    ax.bar(x + 1.5*w, silu_j_identity, w, label="SiLU → identity", color="#2196F3", alpha=0.4)
    ax.set_xlabel("Rotation index")
    ax.set_ylabel("Frobenius distance")
    ax.set_title("Emitter Jacobian: Distance to Target vs Identity\n"
                  "(target = M₂⁻¹M₁, the theoretically optimal pre-rotation)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"Channel Rotation Adaptation (dim={dim}, κ={kappa})\n"
        f"Receiver pre-trained on M₁, Emitter adapts to rotated M₂ (same spectrum)",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved: {output_path}")


def main():
    os.makedirs("results", exist_ok=True)

    dim = 8
    kappa = 10
    n_rotations = 4
    epochs_recv = 300
    epochs_emit = 400
    epochs_adapt = 400
    n_samples = 1500

    print("=" * 70)
    print("EXPERIMENT: Channel Rotation Adaptation (obj-019)")
    print("Can the Emitter adapt when the channel rotates but the Receiver is frozen?")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("Part A: ReLU activation")
    print("=" * 70)
    results_relu = run_adaptation_experiment(
        dim=dim, activation="relu", kappa=kappa,
        epochs_recv=epochs_recv, epochs_emit=epochs_emit,
        epochs_adapt=epochs_adapt, n_samples=n_samples,
        n_rotations=n_rotations,
    )

    print("\n" + "=" * 70)
    print("Part B: SiLU activation")
    print("=" * 70)
    results_silu = run_adaptation_experiment(
        dim=dim, activation="silu", kappa=kappa,
        epochs_recv=epochs_recv, epochs_emit=epochs_emit,
        epochs_adapt=epochs_adapt, n_samples=n_samples,
        n_rotations=n_rotations,
    )

    plot_results(results_relu, results_silu, dim, kappa,
                 "results/obj-019-rotation-adaptation.png")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for act_name, results in [("ReLU", results_relu), ("SiLU", results_silu)]:
        adapt_mses = [a["adapt_mse"] for a in results["adaptations"]]
        oracle_mses = [a["oracle_mse"] for a in results["adaptations"]]
        ratios = [a["mse_ratio"] for a in results["adaptations"]]
        j_targets = [a["J_dist_to_target"] for a in results["adaptations"]]
        j_identities = [a["J_dist_to_identity"] for a in results["adaptations"]]

        print(f"\n{act_name}:")
        print(f"  Baseline MSE (M₁):       {results['baseline_mse']:.6f}")
        print(f"  Adapted MSE (mean):       {np.mean(adapt_mses):.6f} ± {np.std(adapt_mses):.6f}")
        print(f"  Oracle MSE (mean):        {np.mean(oracle_mses):.6f} ± {np.std(oracle_mses):.6f}")
        print(f"  Adaptation ratio (mean):  {np.mean(ratios):.2f}× ± {np.std(ratios):.2f}")
        print(f"  J→target distance (mean): {np.mean(j_targets):.3f}")
        print(f"  J→identity dist (mean):   {np.mean(j_identities):.3f}")

        if np.mean(j_targets) < np.mean(j_identities):
            print(f"  → Emitter LEARNED the pre-rotation (closer to M₂⁻¹M₁ than to I)")
        else:
            print(f"  → Emitter stayed near identity (didn't fully learn pre-rotation)")


if __name__ == "__main__":
    main()
