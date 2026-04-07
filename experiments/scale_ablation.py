"""
Scale Ablation: Does two-phase advantage hold at dim=16, 32, 64?

Runs the 3-condition param-count ablation at multiple dimensions.
"""

import torch
import torch.nn as nn
import numpy as np
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from components import Emitter, ActionToSignal, Environment, Receiver

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class NoisyPipeline(nn.Module):
    def __init__(self, emitter, a2s, env, receiver):
        super().__init__()
        self.emitter = emitter
        self.action_to_signal = a2s
        self.environment = env
        self.receiver = receiver

    def forward(self, x):
        action = self.emitter(x)
        signal = self.action_to_signal(action)
        received = self.environment(signal)
        return self.receiver(received)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def pretrain_receiver(dim, hidden, a2s, env, epochs, n_samples, seed=42):
    torch.manual_seed(seed)
    recv = Receiver(dim, dim, hidden)
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


def serialize(obj):
    if isinstance(obj, dict):
        return {k: serialize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize(v) for v in obj]
    elif isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    return obj


def run_scale_ablation():
    """Run 3-condition ablation at dim=8, 16, 32, 64."""
    configs = [
        {"dim": 8,  "hidden": 64,  "epochs": 1000, "n_samples": 5000},
        {"dim": 16, "hidden": 128, "epochs": 2000, "n_samples": 5000},
        {"dim": 32, "hidden": 256, "epochs": 3000, "n_samples": 8000},
        {"dim": 64, "hidden": 512, "epochs": 3000, "n_samples": 10000},
    ]
    seeds = range(42, 45)
    all_results = {}

    for cfg in configs:
        dim, hidden, epochs, n_samples = cfg["dim"], cfg["hidden"], cfg["epochs"], cfg["n_samples"]
        print(f"\n{'='*60}")
        print(f"DIM={dim}, hidden={hidden}, epochs={epochs}")
        print(f"{'='*60}")

        methods = {"sequential": [], "joint": [], "matched": []}

        for seed in seeds:
            print(f"  seed={seed}...", end=" ", flush=True)
            t0 = time.time()
            torch.manual_seed(seed)
            test_sounds = torch.randn(2000, dim)
            sounds = torch.randn(n_samples, dim)

            # Sequential
            a2s = ActionToSignal(dim, dim, seed=300)
            env = Environment(dim, seed=400)
            recv = pretrain_receiver(dim, hidden, a2s, env, epochs, n_samples, seed)
            torch.manual_seed(seed + 1000)
            emit = Emitter(dim, dim, hidden)
            pipe = NoisyPipeline(emit, a2s, env, recv)
            pipe.receiver.requires_grad_(False)
            pipe.action_to_signal.requires_grad_(False)
            pipe.environment.requires_grad_(False)
            curve_seq = run_training(pipe, list(emit.parameters()), sounds, test_sounds, epochs)
            methods["sequential"].append(curve_seq)

            # Joint
            a2s2 = ActionToSignal(dim, dim, seed=300)
            env2 = Environment(dim, seed=400)
            recv2 = Receiver(dim, dim, hidden)
            torch.manual_seed(seed + 1000)
            emit2 = Emitter(dim, dim, hidden)
            pipe2 = NoisyPipeline(emit2, a2s2, env2, recv2)
            params = list(emit2.parameters()) + list(recv2.parameters())
            curve_jnt = run_training(pipe2, params, sounds, test_sounds, epochs)
            methods["joint"].append(curve_jnt)

            # Matched (random frozen receiver)
            a2s3 = ActionToSignal(dim, dim, seed=300)
            env3 = Environment(dim, seed=400)
            recv3 = Receiver(dim, dim, hidden)
            torch.manual_seed(seed + 1000)
            emit3 = Emitter(dim, dim, hidden)
            pipe3 = NoisyPipeline(emit3, a2s3, env3, recv3)
            pipe3.receiver.requires_grad_(False)
            pipe3.action_to_signal.requires_grad_(False)
            pipe3.environment.requires_grad_(False)
            curve_matched = run_training(pipe3, list(emit3.parameters()), sounds, test_sounds, epochs)
            methods["matched"].append(curve_matched)

            elapsed = time.time() - t0
            print(f"seq={curve_seq[-1][1]:.6f}, jnt={curve_jnt[-1][1]:.6f}, "
                  f"matched={curve_matched[-1][1]:.6f} ({elapsed/60:.1f}min)")

        # Aggregate
        dim_results = {"dim": dim, "hidden": hidden, "epochs": epochs,
                       "emitter_params": count_params(emit),
                       "joint_params": count_params(emit) + count_params(recv)}
        for method, curves in methods.items():
            common_epochs = [ep for ep, _ in curves[0]]
            mse_matrix = np.array([[m for _, m in c] for c in curves])
            dim_results[method] = {
                "mean_curve": mse_matrix.mean(axis=0).tolist(),
                "std_curve": mse_matrix.std(axis=0).tolist(),
                "epochs": common_epochs,
                "final_mse_mean": float(mse_matrix[:, -1].mean()),
                "final_mse_std": float(mse_matrix[:, -1].std()),
            }
            print(f"  {method}: MSE={dim_results[method]['final_mse_mean']:.6f}"
                  f"±{dim_results[method]['final_mse_std']:.6f}")

        all_results[f"dim{dim}"] = dim_results

        # Checkpoint after each dim
        with open(f"results/scale_ablation_dim{dim}.json", "w") as f:
            json.dump(serialize(dim_results), f, indent=2)
        print(f"  [checkpoint: results/scale_ablation_dim{dim}.json]")

    return all_results


def plot_scale(results, save_dir="figures"):
    os.makedirs(save_dir, exist_ok=True)
    COLORS = {'sequential': '#2196F3', 'joint': '#FF9800', 'matched': '#9C27B0'}

    dims = sorted([k for k in results], key=lambda x: int(x.replace("dim", "")))
    n = len(dims)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4.5))
    if n == 1:
        axes = [axes]

    for i, dk in enumerate(dims):
        ax = axes[i]
        data = results[dk]
        for method, color, label in [
            ("sequential", COLORS["sequential"], "Sequential"),
            ("joint", COLORS["joint"], "Joint"),
            ("matched", COLORS["matched"], "Matched (random recv)"),
        ]:
            r = data[method]
            epochs = r["epochs"]
            mean = np.array(r["mean_curve"])
            std = np.array(r["std_curve"])
            ax.semilogy(epochs, mean, '-', color=color, label=label, linewidth=2)
            ax.fill_between(epochs, np.maximum(mean - std, 1e-8),
                          mean + std, alpha=0.15, color=color)
        ax.set_xlabel('Epochs')
        if i == 0:
            ax.set_ylabel('Test MSE (log)')
        d = data["dim"]
        h = data["hidden"]
        ep = data["emitter_params"]
        ax.set_title(f'dim={d}, h={h}\n({ep} emit params)', fontweight='bold')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle('Two-Phase Advantage Across Dimensions (3 seeds, mean±std)',
                fontweight='bold', y=1.02)
    plt.tight_layout()
    p = os.path.join(save_dir, "pub_scale_ablation.png")
    fig.savefig(p, bbox_inches='tight', dpi=150)
    print(f"Saved {p}")

    # Summary bar chart - final MSE by dim
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(n)
    width = 0.25
    for j, (method, color) in enumerate([
        ("sequential", COLORS["sequential"]),
        ("joint", COLORS["joint"]),
        ("matched", COLORS["matched"]),
    ]):
        means = [results[dk][method]["final_mse_mean"] for dk in dims]
        stds = [results[dk][method]["final_mse_std"] for dk in dims]
        offset = (j - 1) * width
        ax.bar(x + offset, means, width, yerr=stds, label=method.capitalize(),
               color=color, edgecolor='white', capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels([f'dim={results[dk]["dim"]}' for dk in dims])
    ax.set_ylabel('Final Test MSE')
    ax.set_yscale('log')
    ax.set_title('Final MSE by Dimension and Training Method', fontweight='bold')
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    p2 = os.path.join(save_dir, "pub_scale_summary.png")
    fig.savefig(p2, bbox_inches='tight', dpi=150)
    print(f"Saved {p2}")
    return [p, p2]


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    t_start = time.time()

    results = run_scale_ablation()

    with open("results/scale_ablation.json", "w") as f:
        json.dump(serialize(results), f, indent=2)

    print("\n=== GENERATING FIGURES ===")
    plot_scale(results)

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed/60:.1f} minutes")
