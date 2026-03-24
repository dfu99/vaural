"""Experiment: Channel Noise Alignment Boundary.

Extended training with varying channel noise levels to characterize where
the Emitter's alignment with the full pipeline inverse breaks down.

At zero noise, C_i(pipeline) = 1.0 (obj-027). As noise increases, the
channel becomes non-invertible and C_i should degrade. The question:
is there a sharp alignment boundary, or a gradual decline?

This also cross-validates C_i as a predictor of reconstruction quality
in a new setting (noise) beyond the rotation experiments.

Setup:
  - Add Gaussian noise after the Environment transform: received = Env(A2S(action)) + σ·ε
  - Sweep noise σ = [0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
  - Extended training: 500 recv + 1000 emit epochs (GPU budget)
  - dim=16, hidden=128 (leveraging GPU)
  - Both trained Receiver and fixed linear Receiver regimes
  - Measure: MSE, C_i(channel), C_i(pipeline), magnitude ratio

Uses GPU if available, CPU otherwise.
"""

import os
import sys
import json
import time

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from components import Emitter, Receiver, ActionToSignal, Environment, Pipeline
from experiments.pure_rotational_invariance import make_channel_from_svd
from experiments.fixed_receiver import FixedLinearReceiver


# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class NoisyEnvironment(nn.Module):
    """Environment transform with additive Gaussian noise."""
    def __init__(self, signal_dim, seed=200, noise_std=0.0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        weight = torch.randn(signal_dim, signal_dim, generator=gen)
        self.register_buffer("weight", weight)
        self.noise_std = noise_std

    def forward(self, x):
        out = x @ self.weight.T
        if self.training and self.noise_std > 0:
            out = out + self.noise_std * torch.randn_like(out)
        return out


class NoisyPipeline(nn.Module):
    """Pipeline with noise injection after environment."""
    def __init__(self, emitter, a2s, env, receiver, noise_std=0.0):
        super().__init__()
        self.emitter = emitter
        self.action_to_signal = a2s
        self.environment = env
        self.receiver = receiver
        self.noise_std = noise_std

    def forward(self, x):
        action = self.emitter(x)
        signal = self.action_to_signal(action)
        received = self.environment(signal)
        if self.training and self.noise_std > 0:
            received = received + self.noise_std * torch.randn_like(received)
        decoded = self.receiver(received)
        return decoded


def pretrain_receiver_noisy(a2s, env, receiver, cfg, noise_std=0.0):
    """Pre-train receiver with optional noise injection."""
    sounds = torch.randn(cfg.receiver_samples, cfg.sound_dim, device=device)
    with torch.no_grad():
        signals = a2s(sounds)
        received = env(signals)

    optimizer = torch.optim.Adam(receiver.parameters(), lr=cfg.receiver_lr)
    loss_fn = nn.MSELoss()
    losses = []

    for epoch in range(cfg.receiver_epochs):
        perm = torch.randperm(sounds.size(0), device=device)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, sounds.size(0), cfg.receiver_batch_size):
            idx = perm[i:i + cfg.receiver_batch_size]
            batch_recv = received[idx]
            if noise_std > 0:
                batch_recv = batch_recv + noise_std * torch.randn_like(batch_recv)
            decoded = receiver(batch_recv)
            loss = loss_fn(decoded, sounds[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        losses.append(epoch_loss / n_batches)
    return losses


def train_emitter_noisy(pipeline, cfg):
    """Train emitter through noisy pipeline."""
    pipeline.receiver.requires_grad_(False)
    pipeline.action_to_signal.requires_grad_(False)
    pipeline.environment.requires_grad_(False)

    sounds = torch.randn(cfg.emitter_samples, cfg.sound_dim, device=device)
    optimizer = torch.optim.Adam(pipeline.emitter.parameters(), lr=cfg.emitter_lr)
    loss_fn = nn.MSELoss()
    losses = []

    for epoch in range(cfg.emitter_epochs):
        perm = torch.randperm(sounds.size(0), device=device)
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


def compute_P_matrix(a2s, env, receiver, dim):
    """Compute linearized P = Receiver ∘ Env ∘ A2S."""
    if isinstance(receiver, FixedLinearReceiver):
        P = receiver.weight @ env.weight @ a2s.weight
    else:
        receiver.eval()
        torch.manual_seed(42)
        test = torch.randn(100, dim, device=device)
        with torch.no_grad():
            test_recv = env(a2s(test))
        Js = []
        for i in range(min(20, 100)):
            x = test_recv[i:i+1].requires_grad_(True)
            y = receiver(x)
            J = []
            for j in range(dim):
                receiver.zero_grad()
                if x.grad is not None:
                    x.grad.zero_()
                y[0, j].backward(retain_graph=True)
                J.append(x.grad[0].clone())
            Js.append(torch.stack(J))
        P = torch.stack(Js).mean(dim=0) @ env.weight @ a2s.weight
        receiver.train()
    return P


def compute_metrics(emitter, M_inv, P_inv, dim, n_samples=2000):
    """Compute C_i variants."""
    torch.manual_seed(99)
    sounds = torch.randn(n_samples, dim, device=device)
    emitter.eval()
    with torch.no_grad():
        emit_out = emitter(sounds)
        opt_chan = (M_inv @ sounds.T).T
        opt_pipe = (P_inv @ sounds.T).T
        ci_chan = nn.functional.cosine_similarity(emit_out, opt_chan, dim=-1)
        ci_pipe = nn.functional.cosine_similarity(emit_out, opt_pipe, dim=-1)
        mag_pipe = (emit_out.norm(dim=-1) / opt_pipe.norm(dim=-1).clamp(min=1e-8)).mean().item()
    emitter.train()
    return {
        "ci_channel": ci_chan.mean().item(),
        "ci_pipeline": ci_pipe.mean().item(),
        "mag_ratio_pipeline": mag_pipe,
    }


def evaluate_noiseless(pipeline, dim, n_samples=2000):
    """Evaluate MSE without noise (test time)."""
    torch.manual_seed(99)
    sounds = torch.randn(n_samples, dim, device=device)
    pipeline.eval()
    with torch.no_grad():
        decoded = pipeline(sounds)
        mse = (decoded - sounds).pow(2).mean().item()
    pipeline.train()
    return mse


def run_experiment(dim=16, n_rotations=3):
    print("=" * 60)
    print("EXPERIMENT: Channel Noise Alignment Boundary")
    print(f"dim={dim}, device={device}")
    print("=" * 60)

    noise_levels = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
    hidden_dim = 128

    sigmas = torch.logspace(0, -1, dim, device=device)  # κ=10
    kappa = (sigmas[0] / sigmas[-1]).item()

    cfg = Config(
        sound_dim=dim, action_dim=dim, signal_dim=dim, hidden_dim=hidden_dim,
        receiver_lr=1e-3, receiver_epochs=500,
        receiver_samples=5000, receiver_batch_size=128,
        emitter_lr=1e-3, emitter_epochs=1000,
        emitter_samples=5000, emitter_batch_size=128,
        plot_every=9999,
    )

    results = {"trained": {}, "fixed_linear": {}}

    for noise_std in noise_levels:
        print(f"\n{'='*50}")
        print(f"Noise σ = {noise_std}")
        print(f"{'='*50}")

        noise_results = {"trained": [], "fixed_linear": []}

        for rot_idx in range(n_rotations):
            seed_u = 1000 + rot_idx * 7
            seed_v = 2000 + rot_idx * 13
            a2s, env, M = make_channel_from_svd(dim, sigmas.cpu(), seed_u, seed_v)
            a2s = a2s.to(device)
            env = env.to(device)
            M = M.to(device)
            M_inv = torch.linalg.inv(M)

            # --- Trained Receiver ---
            print(f"  Rot {rot_idx+1} - Trained Receiver:")
            t0 = time.time()
            torch.manual_seed(42)
            recv_t = Receiver(dim, dim, hidden_dim).to(device)
            pretrain_receiver_noisy(a2s, env, recv_t, cfg, noise_std=noise_std)
            recv_t.requires_grad_(False)

            P_t = compute_P_matrix(a2s, env, recv_t, dim)
            P_inv_t = torch.linalg.inv(P_t)

            torch.manual_seed(42)
            emit_t = Emitter(dim, dim, hidden_dim).to(device)
            pipe_t = NoisyPipeline(emit_t, a2s, env, recv_t, noise_std=noise_std)
            train_emitter_noisy(pipe_t, cfg)

            pipe_t.noise_std = 0.0  # noiseless eval
            mse_t = evaluate_noiseless(pipe_t, dim)
            metrics_t = compute_metrics(emit_t, M_inv, P_inv_t, dim)
            metrics_t["mse"] = mse_t
            noise_results["trained"].append(metrics_t)
            elapsed = time.time() - t0
            print(f"    MSE={mse_t:.6f}, C_i(pipe)={metrics_t['ci_pipeline']:.4f}, "
                  f"C_i(chan)={metrics_t['ci_channel']:.4f} ({elapsed:.0f}s)")

            # --- Fixed Linear Receiver ---
            print(f"  Rot {rot_idx+1} - Fixed Linear Receiver:")
            t0 = time.time()
            recv_f = FixedLinearReceiver(dim, dim, seed=300 + rot_idx).to(device)
            recv_f.requires_grad_(False)

            P_f = recv_f.weight @ env.weight @ a2s.weight
            P_inv_f = torch.linalg.inv(P_f)

            torch.manual_seed(42)
            emit_f = Emitter(dim, dim, hidden_dim).to(device)
            pipe_f = NoisyPipeline(emit_f, a2s, env, recv_f, noise_std=noise_std)
            train_emitter_noisy(pipe_f, cfg)

            pipe_f.noise_std = 0.0
            mse_f = evaluate_noiseless(pipe_f, dim)
            metrics_f = compute_metrics(emit_f, M_inv, P_inv_f, dim)
            metrics_f["mse"] = mse_f
            noise_results["fixed_linear"].append(metrics_f)
            elapsed = time.time() - t0
            print(f"    MSE={mse_f:.6f}, C_i(pipe)={metrics_f['ci_pipeline']:.4f}, "
                  f"C_i(chan)={metrics_f['ci_channel']:.4f} ({elapsed:.0f}s)")

        # Aggregate per noise level
        for regime in ["trained", "fixed_linear"]:
            runs = noise_results[regime]
            results[regime][noise_std] = {
                "mse_mean": np.mean([r["mse"] for r in runs]),
                "mse_std": np.std([r["mse"] for r in runs]),
                "ci_pipe_mean": np.mean([r["ci_pipeline"] for r in runs]),
                "ci_pipe_std": np.std([r["ci_pipeline"] for r in runs]),
                "ci_chan_mean": np.mean([r["ci_channel"] for r in runs]),
                "ci_chan_std": np.std([r["ci_channel"] for r in runs]),
                "mag_mean": np.mean([r["mag_ratio_pipeline"] for r in runs]),
                "runs": runs,
            }

    return results, dim, kappa, noise_levels


def plot_results(results, dim, kappa, noise_levels, output_path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {"trained": "#2196F3", "fixed_linear": "#FF9800"}
    labels = {"trained": "Trained Receiver", "fixed_linear": "Fixed Linear Receiver"}

    for regime in ["trained", "fixed_linear"]:
        data = results[regime]
        sigmas_list = sorted(data.keys())
        mse_means = [data[s]["mse_mean"] for s in sigmas_list]
        mse_stds = [data[s]["mse_std"] for s in sigmas_list]
        ci_pipe_means = [data[s]["ci_pipe_mean"] for s in sigmas_list]
        ci_pipe_stds = [data[s]["ci_pipe_std"] for s in sigmas_list]
        ci_chan_means = [data[s]["ci_chan_mean"] for s in sigmas_list]
        ci_chan_stds = [data[s]["ci_chan_std"] for s in sigmas_list]
        mag_means = [data[s]["mag_mean"] for s in sigmas_list]

        c = colors[regime]
        lab = labels[regime]

        # Panel 1: MSE vs noise
        ax = axes[0, 0]
        ax.errorbar(sigmas_list, mse_means, yerr=mse_stds, fmt="o-",
                     color=c, linewidth=2, markersize=6, capsize=4, label=lab)

        # Panel 2: C_i(pipeline) vs noise
        ax = axes[0, 1]
        ax.errorbar(sigmas_list, ci_pipe_means, yerr=ci_pipe_stds, fmt="o-",
                     color=c, linewidth=2, markersize=6, capsize=4, label=lab)

        # Panel 3: C_i(channel) vs noise
        ax = axes[0, 2]
        ax.errorbar(sigmas_list, ci_chan_means, yerr=ci_chan_stds, fmt="o-",
                     color=c, linewidth=2, markersize=6, capsize=4, label=lab)

        # Panel 4: Magnitude ratio vs noise
        ax = axes[1, 0]
        ax.plot(sigmas_list, mag_means, "o-", color=c, linewidth=2,
                markersize=6, label=lab)

        # Panel 5: C_i(pipeline) vs MSE scatter
        ax = axes[1, 1]
        ax.scatter(ci_pipe_means, mse_means, c=c, s=80, alpha=0.8,
                   edgecolors="white", linewidth=0.5, label=lab, zorder=3)
        for i, s in enumerate(sigmas_list):
            ax.annotate(f"σ={s}", (ci_pipe_means[i], mse_means[i]),
                        textcoords="offset points", xytext=(5, 5), fontsize=7)

    # Format panels
    for ax, title, ylabel in [
        (axes[0, 0], "MSE vs Channel Noise", "Test MSE"),
        (axes[0, 1], "C_i(pipeline) vs Noise\n(1.0 = perfect alignment)", "C_i(pipeline)"),
        (axes[0, 2], "C_i(channel) vs Noise", "C_i(channel)"),
        (axes[1, 0], "Magnitude Ratio vs Noise\n(1.0 = optimal)", "Mag Ratio"),
    ]:
        ax.set_xlabel("Channel Noise σ")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        if "MSE" in ylabel:
            ax.set_yscale("log")

    axes[1, 1].set_xlabel("C_i(pipeline)")
    axes[1, 1].set_ylabel("Test MSE")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("C_i(pipeline) vs MSE\n(does alignment predict quality?)")
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)

    # Panel 6: Summary table
    ax = axes[1, 2]
    ax.axis("off")
    table_data = []
    for s in noise_levels:
        t = results["trained"][s]
        f = results["fixed_linear"][s]
        table_data.append([
            f"σ={s}",
            f"{t['mse_mean']:.5f}", f"{t['ci_pipe_mean']:.3f}",
            f"{f['mse_mean']:.5f}", f"{f['ci_pipe_mean']:.3f}",
        ])
    table = ax.table(
        cellText=table_data,
        colLabels=["Noise", "Tr MSE", "Tr C_i", "Fix MSE", "Fix C_i"],
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.5)
    ax.set_title("Summary", fontsize=11)

    fig.suptitle(
        f"Channel Noise Alignment Boundary (dim={dim}, κ={kappa:.0f}, SiLU, GPU)\n"
        f"At what noise level does Emitter alignment with P⁻¹ break down?",
        fontsize=14, fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved: {output_path}")


def print_summary(results, noise_levels):
    print("\n" + "=" * 70)
    print("SUMMARY: Channel Noise Alignment Boundary")
    print("=" * 70)

    for regime, label in [("trained", "TRAINED RECEIVER"), ("fixed_linear", "FIXED LINEAR")]:
        print(f"\n  {label}:")
        print(f"  {'Noise σ':>8} | {'MSE':>12} | {'C_i(pipe)':>10} | {'C_i(chan)':>10} | {'Mag ratio':>10}")
        print(f"  {'-'*8}-+-{'-'*12}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
        for s in noise_levels:
            d = results[regime][s]
            print(f"  {s:>8.2f} | {d['mse_mean']:>12.6f} | {d['ci_pipe_mean']:>10.4f} | "
                  f"{d['ci_chan_mean']:>10.4f} | {d['mag_mean']:>10.3f}")

    # Correlation: C_i(pipeline) vs log(MSE)
    for regime in ["trained", "fixed_linear"]:
        ci_vals = [results[regime][s]["ci_pipe_mean"] for s in noise_levels]
        mse_vals = [results[regime][s]["mse_mean"] for s in noise_levels]
        r = np.corrcoef(ci_vals, np.log10(np.array(mse_vals) + 1e-10))[0, 1]
        print(f"\n  {regime}: R(C_i_pipeline, log MSE) = {r:.3f}")


def main():
    os.makedirs("results", exist_ok=True)
    results, dim, kappa, noise_levels = run_experiment(dim=16, n_rotations=3)
    plot_results(results, dim, kappa, noise_levels,
                 "results/obj-028-noise-alignment.png")
    print_summary(results, noise_levels)

    # Serialize
    serializable = {}
    for regime, data in results.items():
        serializable[regime] = {}
        for noise, vals in data.items():
            serializable[regime][str(noise)] = {
                k: v for k, v in vals.items() if k != "runs"
            }
    with open("results/obj-028-noise-alignment.json", "w") as f:
        json.dump(serializable, f, indent=2)
    print("Saved: results/obj-028-noise-alignment.json")


if __name__ == "__main__":
    main()
