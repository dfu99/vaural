"""
Transformer Architecture Ablation: Does two-phase training generalize beyond MLPs?

Tests the same param-count ablation (Exp 1 from publishable.py) using
small Transformer encoder/decoder instead of MLPs. If sequential still
dominates, the result is architecture-agnostic.
"""

import torch
import torch.nn as nn
import numpy as np
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from components import ActionToSignal, Environment

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ─── Transformer Components ─────────────────────────────────────────────

class TransformerEmitter(nn.Module):
    """Transformer-based emitter. Treats each dim as a token."""
    def __init__(self, dim, hidden=64, n_heads=4, n_layers=2):
        super().__init__()
        self.dim = dim
        # Project scalar tokens to hidden dim
        self.input_proj = nn.Linear(1, hidden)
        self.pos_embed = nn.Parameter(torch.randn(1, dim, hidden) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=hidden*2,
            dropout=0.0, batch_first=True, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(hidden, 1)

    def forward(self, x):
        # x: (batch, dim) -> treat as (batch, dim, 1) sequence
        tokens = self.input_proj(x.unsqueeze(-1)) + self.pos_embed
        encoded = self.encoder(tokens)
        return self.output_proj(encoded).squeeze(-1)


class TransformerReceiver(nn.Module):
    """Transformer-based receiver."""
    def __init__(self, dim, hidden=64, n_heads=4, n_layers=2):
        super().__init__()
        self.dim = dim
        self.input_proj = nn.Linear(1, hidden)
        self.pos_embed = nn.Parameter(torch.randn(1, dim, hidden) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=hidden*2,
            dropout=0.0, batch_first=True, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(hidden, 1)

    def forward(self, x):
        tokens = self.input_proj(x.unsqueeze(-1)) + self.pos_embed
        encoded = self.encoder(tokens)
        return self.output_proj(encoded).squeeze(-1)


class MLPEmitter(nn.Module):
    """MLP emitter for comparison (same as components.py but standalone)."""
    def __init__(self, dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim),
        )
    def forward(self, x):
        return self.net(x)


class MLPReceiver(nn.Module):
    """MLP receiver for comparison."""
    def __init__(self, dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim),
        )
    def forward(self, x):
        return self.net(x)


class NoisyPipeline(nn.Module):
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
        if self.noise_std > 0 and self.training:
            received = received + torch.randn_like(received) * self.noise_std
        return self.receiver(received)


# ─── Helpers ─────────────────────────────────────────────────────────────

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def pretrain_receiver(recv, dim, a2s, env, epochs, n_samples, seed=42):
    """Pre-train receiver to invert channel."""
    torch.manual_seed(seed)
    sounds = torch.randn(n_samples, dim)
    with torch.no_grad():
        received = env(a2s(sounds))
    opt = torch.optim.Adam(recv.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    for ep in range(epochs):
        perm = torch.randperm(n_samples)
        for i in range(0, n_samples, 64):
            idx = perm[i:i+64]
            loss = loss_fn(recv(received[idx]), sounds[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    return recv


def run_training(model, params_to_train, sounds, test_sounds, epochs,
                 lr=1e-3, batch_size=64, eval_every=10):
    opt = torch.optim.Adam(params_to_train, lr=lr)
    loss_fn = nn.MSELoss()
    curve = []
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(sounds.size(0))
        for i in range(0, sounds.size(0), batch_size):
            idx = perm[i:i+batch_size]
            loss = loss_fn(model(sounds[idx]), sounds[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        if (epoch + 1) % eval_every == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                mse = loss_fn(model(test_sounds), test_sounds).item()
            curve.append((epoch + 1, mse))
    return curve


def find_threshold_epoch(curve, threshold):
    for ep, mse in curve:
        if mse < threshold:
            return ep
    return None


# ─── Main Experiment ─────────────────────────────────────────────────────

def run_architecture_comparison(dim=8, epochs=500, n_samples=3000, seeds=range(42, 45)):
    """
    Compare MLP vs Transformer on the same 3-condition ablation:
    - Sequential: pre-trained receiver, frozen, train emitter
    - Joint: both from scratch
    - Matched: frozen random receiver, train emitter (same params as sequential)
    """
    threshold = 0.01  # Relaxed for 300-epoch budget
    arch_configs = {
        "mlp": {
            "emitter_cls": lambda d: MLPEmitter(d, 64),
            "receiver_cls": lambda d: MLPReceiver(d, 64),
        },
        "transformer": {
            "emitter_cls": lambda d: TransformerEmitter(d, hidden=32, n_heads=4, n_layers=1),
            "receiver_cls": lambda d: TransformerReceiver(d, hidden=32, n_heads=4, n_layers=1),
        },
    }

    all_results = {}
    for arch_name, arch in arch_configs.items():
        print(f"\n{'='*60}")
        print(f"ARCHITECTURE: {arch_name.upper()}")
        print(f"{'='*60}")

        methods = {"sequential": [], "joint": [], "matched": []}

        for seed in seeds:
            print(f"  seed={seed}...", end=" ", flush=True)
            torch.manual_seed(seed)
            test_sounds = torch.randn(2000, dim)
            sounds = torch.randn(n_samples, dim)

            # Sequential
            a2s = ActionToSignal(dim, dim, seed=300)
            env = Environment(dim, seed=400)
            recv = arch["receiver_cls"](dim)
            recv = pretrain_receiver(recv, dim, a2s, env, epochs, n_samples, seed)
            torch.manual_seed(seed + 1000)
            emit = arch["emitter_cls"](dim)
            pipe = NoisyPipeline(emit, a2s, env, recv)
            pipe.receiver.requires_grad_(False)
            pipe.action_to_signal.requires_grad_(False)
            pipe.environment.requires_grad_(False)
            curve = run_training(pipe, list(emit.parameters()), sounds, test_sounds, epochs)
            methods["sequential"].append(curve)
            seq_params = count_params(emit)

            # Joint
            a2s2 = ActionToSignal(dim, dim, seed=300)
            env2 = Environment(dim, seed=400)
            recv2 = arch["receiver_cls"](dim)
            torch.manual_seed(seed + 1000)
            emit2 = arch["emitter_cls"](dim)
            pipe2 = NoisyPipeline(emit2, a2s2, env2, recv2)
            params = list(emit2.parameters()) + list(recv2.parameters())
            curve2 = run_training(pipe2, params, sounds, test_sounds, epochs)
            methods["joint"].append(curve2)
            joint_params = count_params(emit2) + count_params(recv2)

            # Matched (random frozen receiver)
            a2s3 = ActionToSignal(dim, dim, seed=300)
            env3 = Environment(dim, seed=400)
            recv3 = arch["receiver_cls"](dim)  # random, NOT pre-trained
            torch.manual_seed(seed + 1000)
            emit3 = arch["emitter_cls"](dim)
            pipe3 = NoisyPipeline(emit3, a2s3, env3, recv3)
            pipe3.receiver.requires_grad_(False)
            pipe3.action_to_signal.requires_grad_(False)
            pipe3.environment.requires_grad_(False)
            curve3 = run_training(pipe3, list(emit3.parameters()), sounds, test_sounds, epochs)
            methods["matched"].append(curve3)

            print(f"seq={curve[-1][1]:.6f}, jnt={curve2[-1][1]:.6f}, matched={curve3[-1][1]:.6f}")

        # Aggregate
        arch_results = {"emitter_params": seq_params, "joint_params": joint_params}
        for method, curves in methods.items():
            common_epochs = [ep for ep, _ in curves[0]]
            mse_matrix = np.array([[m for _, m in c] for c in curves])
            thresh_epochs = []
            for c in curves:
                te = find_threshold_epoch(c, threshold)
                thresh_epochs.append(te if te else epochs + 1)
            arch_results[method] = {
                "mean_curve": mse_matrix.mean(axis=0).tolist(),
                "std_curve": mse_matrix.std(axis=0).tolist(),
                "epochs": common_epochs,
                "thresh_mean": float(np.mean(thresh_epochs)),
                "thresh_std": float(np.std(thresh_epochs)),
                "final_mse_mean": float(mse_matrix[:, -1].mean()),
                "final_mse_std": float(mse_matrix[:, -1].std()),
            }
            print(f"  {method}: MSE={arch_results[method]['final_mse_mean']:.6f}"
                  f"±{arch_results[method]['final_mse_std']:.6f}, "
                  f"thresh={arch_results[method]['thresh_mean']:.0f}"
                  f"±{arch_results[method]['thresh_std']:.0f}")

        all_results[arch_name] = arch_results

    all_results["threshold"] = threshold
    all_results["dim"] = dim
    return all_results


def plot_comparison(results, save_dir="figures"):
    os.makedirs(save_dir, exist_ok=True)

    COLORS = {'sequential': '#2196F3', 'joint': '#FF9800', 'matched': '#9C27B0'}
    LABELS = {
        'sequential': 'Sequential (pre-trained recv)',
        'joint': 'Joint (both trainable)',
        'matched': 'Matched params (random recv)',
    }

    archs = [k for k in results if k not in ("threshold", "dim")]
    fig, axes = plt.subplots(1, len(archs), figsize=(7*len(archs), 5))
    if len(archs) == 1:
        axes = [axes]

    for i, arch in enumerate(archs):
        ax = axes[i]
        data = results[arch]
        for method in ["sequential", "joint", "matched"]:
            r = data[method]
            epochs = r["epochs"]
            mean = np.array(r["mean_curve"])
            std = np.array(r["std_curve"])
            ax.semilogy(epochs, mean, '-', color=COLORS[method],
                       label=LABELS[method], linewidth=2)
            ax.fill_between(epochs, np.maximum(mean - std, 1e-8),
                          mean + std, alpha=0.15, color=COLORS[method])
        ax.axhline(y=results["threshold"], color='gray', ls='--', alpha=0.5)
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Test MSE (log)')
        ep = data["emitter_params"]
        jp = data["joint_params"]
        ax.set_title(f'{arch.upper()} (emit={ep}, joint={jp} params)',
                    fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle(f'Architecture-Agnostic Two-Phase Advantage (dim={results["dim"]}, 3 seeds)',
                fontweight='bold', y=1.02)
    plt.tight_layout()
    p = os.path.join(save_dir, "pub_architecture_comparison.png")
    fig.savefig(p, bbox_inches='tight', dpi=150)
    print(f"Saved {p}")

    # Bar chart of threshold epochs
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(3)
    width = 0.35
    for j, arch in enumerate(archs):
        data = results[arch]
        means = [data[m]["thresh_mean"] for m in ["sequential", "joint", "matched"]]
        stds = [data[m]["thresh_std"] for m in ["sequential", "joint", "matched"]]
        offset = (j - 0.5) * width
        bars = ax.bar(x + offset, means, width, yerr=stds, label=arch.upper(),
                     capsize=4, edgecolor='white')
        for bar, tm in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
                    f'{tm:.0f}', ha='center', fontsize=8, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(['Sequential\n(pre-trained)', 'Joint\n(both trainable)',
                       'Matched\n(random recv)'])
    ax.set_ylabel(f'Epochs to MSE < {results["threshold"]}')
    ax.set_title('Convergence Speed: MLP vs Transformer', fontweight='bold')
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    p2 = os.path.join(save_dir, "pub_arch_threshold_comparison.png")
    fig.savefig(p2, bbox_inches='tight', dpi=150)
    print(f"Saved {p2}")

    return [p, p2]


def serialize(obj):
    if isinstance(obj, dict):
        return {k: serialize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize(v) for v in obj]
    elif isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    return obj


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    t_start = time.time()

    results = run_architecture_comparison()

    with open("results/transformer_ablation.json", "w") as f:
        json.dump(serialize(results), f, indent=2)

    print("\n=== GENERATING FIGURES ===")
    plot_comparison(results)

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed/60:.1f} minutes")
