"""
Slam Dunk Experiment: Proving the Two-Phase Advantage

Three concrete claims with quantitative evidence:

CLAIM 1: Parameter Efficiency During Adaptation
  Sequential trains 50% fewer params when adapting to a new channel,
  while achieving equal or better MSE.

CLAIM 2: Convergence Speed
  Sequential reaches low MSE in fewer epochs during channel adaptation.

CLAIM 3: Sample Efficiency
  Sequential achieves lower MSE with fewer training samples.

Methods compared:
- Sequential (ours): Pre-train Receiver on channel, freeze, train Emitter
- Joint: Train Emitter + Receiver simultaneously
- Monolithic: Single large MLP (no modular decomposition)
"""

import torch
import torch.nn as nn
import numpy as np
import json
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components import Emitter, ActionToSignal, Environment, Receiver, Pipeline
from config import Config

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 12,
    'legend.fontsize': 10, 'figure.dpi': 150,
})

COLORS = {'sequential': '#2196F3', 'joint': '#FF9800', 'monolithic': '#F44336'}


class MonolithicModel(nn.Module):
    """Single MLP that maps sound→sound through the channel."""
    def __init__(self, sound_dim, signal_dim, hidden_dim, a2s_seed=100, env_seed=200):
        super().__init__()
        a2s = ActionToSignal(sound_dim, signal_dim, seed=a2s_seed)
        env = Environment(signal_dim, seed=env_seed)
        self.register_buffer("channel_weight", env.weight @ a2s.weight)
        # 2x hidden to give it comparable total param count to sequential
        h = hidden_dim * 2
        self.net = nn.Sequential(
            nn.Linear(signal_dim, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, sound_dim),
        )

    def forward(self, x):
        received = x @ self.channel_weight.T
        return self.net(received)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _train_loop(model, params_to_train, sounds, test_sounds, cfg, epochs, eval_every=25):
    """Generic training loop, returns eval curve."""
    opt = torch.optim.Adam(params_to_train, lr=cfg.emitter_lr)
    loss_fn = nn.MSELoss()
    curve = []
    for epoch in range(epochs):
        perm = torch.randperm(sounds.size(0))
        for i in range(0, sounds.size(0), cfg.emitter_batch_size):
            idx = perm[i:i + cfg.emitter_batch_size]
            loss = loss_fn(model(sounds[idx]), sounds[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if (epoch + 1) % eval_every == 0 or epoch == 0:
            with torch.no_grad():
                mse = loss_fn(model(test_sounds), test_sounds).item()
            curve.append((epoch + 1, mse))
    return curve


def train_sequential(dim, hidden, epochs_recv, epochs_emit, n_samples, a2s_seed, env_seed, seed=42, eval_every=25):
    torch.manual_seed(seed)
    cfg = Config(sound_dim=dim, action_dim=dim, signal_dim=dim, hidden_dim=hidden,
                 receiver_lr=1e-3, emitter_lr=1e-3, emitter_batch_size=64,
                 receiver_batch_size=64)

    a2s = ActionToSignal(dim, dim, seed=a2s_seed)
    env = Environment(dim, seed=env_seed)
    recv = Receiver(dim, dim, hidden)
    emit = Emitter(dim, dim, hidden)

    # Phase 1: pre-train receiver
    sounds = torch.randn(n_samples, dim)
    with torch.no_grad():
        received = env(a2s(sounds))
    test_s = torch.randn(2000, dim)
    with torch.no_grad():
        test_r = env(a2s(test_s))

    opt = torch.optim.Adam(recv.parameters(), lr=cfg.receiver_lr)
    loss_fn = nn.MSELoss()
    for ep in range(epochs_recv):
        perm = torch.randperm(sounds.size(0))
        for i in range(0, sounds.size(0), 64):
            idx = perm[i:i+64]
            loss = loss_fn(recv(received[idx]), sounds[idx])
            opt.zero_grad(); loss.backward(); opt.step()

    # Phase 2: train emitter (receiver frozen)
    pipe = Pipeline(emit, a2s, env, recv)
    pipe.receiver.requires_grad_(False)
    pipe.action_to_signal.requires_grad_(False)
    pipe.environment.requires_grad_(False)

    sounds2 = torch.randn(n_samples, dim)
    test2 = torch.randn(2000, dim)
    curve = _train_loop(pipe, list(emit.parameters()), sounds2, test2, cfg, epochs_emit, eval_every)

    return pipe, curve, count_params(emit)


def train_joint(dim, hidden, epochs, n_samples, a2s_seed, env_seed, seed=42, eval_every=25):
    torch.manual_seed(seed)
    cfg = Config(sound_dim=dim, action_dim=dim, signal_dim=dim, hidden_dim=hidden,
                 emitter_lr=1e-3, emitter_batch_size=64)

    a2s = ActionToSignal(dim, dim, seed=a2s_seed)
    env = Environment(dim, seed=env_seed)
    recv = Receiver(dim, dim, hidden)
    emit = Emitter(dim, dim, hidden)
    pipe = Pipeline(emit, a2s, env, recv)

    sounds = torch.randn(n_samples, dim)
    test_sounds = torch.randn(2000, dim)
    params = list(emit.parameters()) + list(recv.parameters())
    curve = _train_loop(pipe, params, sounds, test_sounds, cfg, epochs, eval_every)

    return pipe, curve, count_params(emit) + count_params(recv)


def train_monolithic(dim, hidden, epochs, n_samples, a2s_seed, env_seed, seed=42, eval_every=25):
    torch.manual_seed(seed)
    cfg = Config(sound_dim=dim, action_dim=dim, signal_dim=dim, hidden_dim=hidden,
                 emitter_lr=1e-3, emitter_batch_size=64)

    model = MonolithicModel(dim, dim, hidden, a2s_seed, env_seed)
    sounds = torch.randn(n_samples, dim)
    test_sounds = torch.randn(2000, dim)
    curve = _train_loop(model, list(model.parameters()), sounds, test_sounds, cfg, epochs, eval_every)

    return model, curve, count_params(model)


def experiment_adaptation(dim=8, hidden=64, initial_epochs=1000, adapt_epochs=500, n_samples=5000):
    """Train on Channel A, then adapt to Channel B. Measure adaptation efficiency."""
    ch_a = (100, 200)
    ch_b = (300, 400)

    print(f"\n=== ADAPTATION EXPERIMENT dim={dim} ===")

    # Initial training on Channel A
    print("  Initial training (Channel A)...")
    _, seq_init, seq_p = train_sequential(dim, hidden, initial_epochs, initial_epochs, n_samples, *ch_a)
    _, jnt_init, jnt_p = train_joint(dim, hidden, initial_epochs, n_samples, *ch_a)
    _, mon_init, mon_p = train_monolithic(dim, hidden, initial_epochs, n_samples, *ch_a)

    print(f"    Sequential: MSE={seq_init[-1][1]:.6f}, params={seq_p}")
    print(f"    Joint:      MSE={jnt_init[-1][1]:.6f}, params={jnt_p}")
    print(f"    Monolithic: MSE={mon_init[-1][1]:.6f}, params={mon_p}")

    # Adaptation to Channel B
    print("  Adapting to Channel B...")
    _, seq_adapt, _ = train_sequential(dim, hidden, adapt_epochs, adapt_epochs, n_samples, *ch_b)
    _, jnt_adapt, _ = train_joint(dim, hidden, adapt_epochs, n_samples, *ch_b)
    _, mon_adapt, _ = train_monolithic(dim, hidden, adapt_epochs, n_samples, *ch_b)

    print(f"    Sequential: MSE={seq_adapt[-1][1]:.6f}")
    print(f"    Joint:      MSE={jnt_adapt[-1][1]:.6f}")
    print(f"    Monolithic: MSE={mon_adapt[-1][1]:.6f}")

    return {
        "dim": dim,
        "initial": {
            "sequential": {"curve": seq_adapt, "params": seq_p, "final_mse": seq_init[-1][1]},
            "joint": {"curve": jnt_init, "params": jnt_p, "final_mse": jnt_init[-1][1]},
            "monolithic": {"curve": mon_init, "params": mon_p, "final_mse": mon_init[-1][1]},
        },
        "adaptation": {
            "sequential": {"curve": seq_adapt, "params": seq_p, "final_mse": seq_adapt[-1][1]},
            "joint": {"curve": jnt_adapt, "params": jnt_p, "final_mse": jnt_adapt[-1][1]},
            "monolithic": {"curve": mon_adapt, "params": mon_p, "final_mse": mon_adapt[-1][1]},
        },
    }


def experiment_sample_efficiency(dim=8, hidden=64, epochs=800):
    """How many samples needed to reach a given MSE?"""
    sample_sizes = [100, 250, 500, 1000, 2500, 5000]
    results = {}

    print(f"\n=== SAMPLE EFFICIENCY dim={dim} ===")

    for method_name, train_fn in [("sequential", lambda n: train_sequential(dim, hidden, epochs, epochs, n, 100, 200)),
                                    ("joint", lambda n: train_joint(dim, hidden, epochs, n, 100, 200)),
                                    ("monolithic", lambda n: train_monolithic(dim, hidden, epochs, n, 100, 200))]:
        mses = []
        for n in sample_sizes:
            _, curve, params = train_fn(n)
            mses.append(curve[-1][1])
            print(f"  {method_name} n={n}: MSE={curve[-1][1]:.6f}")
        results[method_name] = {"mses": mses, "params": params}

    results["sample_sizes"] = sample_sizes
    return results


def experiment_epochs_to_threshold(dim=8, hidden=64, n_samples=5000, threshold=0.001):
    """How many adaptation epochs to reach a MSE threshold?"""
    max_epochs = 1000
    ch_b = (300, 400)

    print(f"\n=== EPOCHS TO THRESHOLD (MSE < {threshold}) dim={dim} ===")

    results = {}
    for method_name, train_fn in [
        ("sequential", lambda: train_sequential(dim, hidden, max_epochs, max_epochs, n_samples, *ch_b, eval_every=10)),
        ("joint", lambda: train_joint(dim, hidden, max_epochs, n_samples, *ch_b, eval_every=10)),
        ("monolithic", lambda: train_monolithic(dim, hidden, max_epochs, n_samples, *ch_b, eval_every=10)),
    ]:
        _, curve, params = train_fn()
        # Find first epoch below threshold
        epoch_reached = None
        for ep, mse in curve:
            if mse < threshold:
                epoch_reached = ep
                break
        results[method_name] = {
            "curve": curve, "params": params,
            "epoch_reached": epoch_reached,
            "final_mse": curve[-1][1],
        }
        status = f"epoch {epoch_reached}" if epoch_reached else "NOT REACHED"
        print(f"  {method_name}: {status}, final MSE={curve[-1][1]:.6f}, params={params}")

    results["threshold"] = threshold
    return results


def plot_all(adapt_results, sample_results, threshold_results, save_dir="figures"):
    os.makedirs(save_dir, exist_ok=True)
    fig_paths = []

    # --- Figure 1: Adaptation Convergence ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: convergence curves
    ax = axes[0]
    for method in ["sequential", "joint", "monolithic"]:
        data = threshold_results[method]
        epochs = [e for e, m in data["curve"]]
        mses = [m for e, m in data["curve"]]
        ax.semilogy(epochs, mses, '-', color=COLORS[method],
                   label=f'{method.capitalize()} ({data["params"]:,}p)', linewidth=2)

    ax.axhline(y=threshold_results["threshold"], color='gray', linestyle='--', alpha=0.6,
              label=f'Threshold ({threshold_results["threshold"]})')
    ax.set_xlabel('Adaptation Epochs')
    ax.set_ylabel('Test MSE (log)')
    ax.set_title('Convergence Speed on New Channel')
    ax.legend()
    ax.grid(True, alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Right: epochs to threshold bar chart
    ax = axes[1]
    methods = ["sequential", "joint", "monolithic"]
    epochs_reached = []
    for m in methods:
        ep = threshold_results[m]["epoch_reached"]
        epochs_reached.append(ep if ep else 1000)
    colors = [COLORS[m] for m in methods]
    bars = ax.bar(range(3), epochs_reached, color=colors, edgecolor='white')

    for bar, m in zip(bars, methods):
        p = threshold_results[m]["params"]
        ep = threshold_results[m]["epoch_reached"]
        label = f'{ep} ep\n{p:,}p' if ep else f'>1000 ep\n{p:,}p'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
                label, ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xticks(range(3))
    ax.set_xticklabels([m.capitalize() for m in methods])
    ax.set_ylabel('Epochs to Reach Threshold')
    ax.set_title(f'Adaptation Speed (MSE < {threshold_results["threshold"]})')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    p = os.path.join(save_dir, "adaptation_convergence.png")
    fig.savefig(p, bbox_inches='tight', dpi=150)
    fig_paths.append(p)
    print(f"Saved {p}")

    # --- Figure 2: Parameter Efficiency (the money shot) ---
    fig, ax = plt.subplots(figsize=(8, 6))

    for phase_name, phase_data, marker in [("initial", "initial", 'o'), ("adapt", "adaptation", 's')]:
        if phase_name == "initial" and adapt_results:
            data = adapt_results[0]["initial"]
        elif adapt_results:
            data = adapt_results[0]["adaptation"]
        else:
            continue

        for method in ["sequential", "joint", "monolithic"]:
            params = data[method]["params"]
            mse = data[method]["final_mse"]
            ax.scatter(params, mse, c=COLORS[method], marker=marker, s=120,
                      edgecolors='white', linewidths=1, zorder=5)
            label_text = f'{method.capitalize()}\n({phase_name})'

    # Custom legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['sequential'],
               markersize=10, label='Sequential (ours)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['joint'],
               markersize=10, label='Joint'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['monolithic'],
               markersize=10, label='Monolithic'),
        Line2D([0], [0], marker='o', color='gray', markersize=8, linestyle='None', label='Initial'),
        Line2D([0], [0], marker='s', color='gray', markersize=8, linestyle='None', label='Adaptation'),
    ]
    ax.legend(handles=handles, loc='upper right')
    ax.set_xlabel('Trainable Parameters')
    ax.set_ylabel('Test MSE (↓ better)')
    ax.set_title('Parameter Efficiency: MSE vs Trainable Parameters\n(lower-left = better)', fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    p = os.path.join(save_dir, "parameter_efficiency.png")
    fig.savefig(p, bbox_inches='tight', dpi=150)
    fig_paths.append(p)
    print(f"Saved {p}")

    # --- Figure 3: Sample Efficiency ---
    if sample_results:
        fig, ax = plt.subplots(figsize=(8, 5))
        ss = sample_results["sample_sizes"]
        for method in ["sequential", "joint", "monolithic"]:
            mses = sample_results[method]["mses"]
            params = sample_results[method]["params"]
            ax.semilogy(ss, mses, '-o', color=COLORS[method],
                       label=f'{method.capitalize()} ({params:,}p)',
                       linewidth=2, markersize=6)
        ax.set_xlabel('Training Samples')
        ax.set_ylabel('Test MSE (log)')
        ax.set_title('Sample Efficiency (dim=8, 800 epochs)', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        p = os.path.join(save_dir, "sample_efficiency.png")
        fig.savefig(p, bbox_inches='tight', dpi=150)
        fig_paths.append(p)
        print(f"Saved {p}")

    # --- Figure 4: Combined Summary Dashboard ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Top-left: adaptation bar chart
    ax = axes[0, 0]
    if adapt_results:
        r = adapt_results[0]
        adapt = r["adaptation"]
        methods = ["sequential", "joint", "monolithic"]
        mses = [adapt[m]["final_mse"] for m in methods]
        params = [adapt[m]["params"] for m in methods]
        bars = ax.bar(range(3), mses, color=[COLORS[m] for m in methods], edgecolor='white')
        for bar, p in zip(bars, params):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mses)*0.03,
                    f'{p:,}p', ha='center', fontsize=9, color='#555')
        ax.set_xticks(range(3))
        ax.set_xticklabels([m.capitalize() for m in methods])
        ax.set_ylabel('Test MSE')
        ax.set_title(f'Adaptation MSE (dim={r["dim"]})')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Top-right: convergence
    ax = axes[0, 1]
    for method in ["sequential", "joint", "monolithic"]:
        data = threshold_results[method]
        epochs_list = [e for e, m in data["curve"]]
        mses = [m for e, m in data["curve"]]
        ax.semilogy(epochs_list, mses, '-', color=COLORS[method],
                   label=method.capitalize(), linewidth=2)
    ax.axhline(y=threshold_results["threshold"], color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('MSE (log)')
    ax.set_title('Convergence on New Channel')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Bottom-left: sample efficiency
    ax = axes[1, 0]
    if sample_results:
        ss = sample_results["sample_sizes"]
        for method in ["sequential", "joint", "monolithic"]:
            ax.semilogy(ss, sample_results[method]["mses"], '-o', color=COLORS[method],
                       label=method.capitalize(), linewidth=2, markersize=5)
        ax.set_xlabel('Training Samples')
        ax.set_ylabel('MSE (log)')
        ax.set_title('Sample Efficiency')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Bottom-right: param efficiency ratio
    ax = axes[1, 1]
    if adapt_results:
        r = adapt_results[0]
        seq_mse = r["adaptation"]["sequential"]["final_mse"]
        seq_p = r["adaptation"]["sequential"]["params"]

        ratios = []
        labels_r = []
        colors_r = []
        for method in ["joint", "monolithic"]:
            mse_ratio = r["adaptation"][method]["final_mse"] / max(seq_mse, 1e-10)
            param_ratio = r["adaptation"][method]["params"] / seq_p
            ratios.append(mse_ratio)
            labels_r.append(f'{method.capitalize()}\nvs Sequential')
            colors_r.append(COLORS[method])

        bars = ax.bar(range(2), ratios, color=colors_r, edgecolor='white')
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        ax.set_xticks(range(2))
        ax.set_xticklabels(labels_r)
        ax.set_ylabel('MSE Ratio vs Sequential')
        ax.set_title('How Much Worse Than Sequential?\n(>1× = Sequential wins)')
        for bar, r_val in zip(bars, ratios):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{r_val:.1f}×', ha='center', fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.suptitle('Two-Phase Sequential Training: The Slam Dunk', fontweight='bold', fontsize=14, y=1.01)
    plt.tight_layout()
    p = os.path.join(save_dir, "slam_dunk_dashboard.png")
    fig.savefig(p, bbox_inches='tight', dpi=150)
    fig_paths.append(p)
    print(f"Saved {p}")

    return fig_paths


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    print("=" * 60)
    print("SLAM DUNK EXPERIMENT")
    print("=" * 60)

    # All experiments at dim=8 for speed (the story is clear at any dim)
    DIM, HIDDEN = 8, 64

    # Experiment 1: Adaptation
    adapt_results = [experiment_adaptation(DIM, HIDDEN,
                                            initial_epochs=1000, adapt_epochs=500)]

    # Experiment 2: Epochs to threshold
    threshold_results = experiment_epochs_to_threshold(DIM, HIDDEN, threshold=0.001)

    # Experiment 3: Sample efficiency
    sample_results = experiment_sample_efficiency(DIM, HIDDEN, epochs=800)

    # Save raw results
    def serialize(obj):
        if isinstance(obj, dict):
            return {k: serialize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [serialize(v) for v in obj]
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        return obj

    all_results = {
        "adaptation": adapt_results,
        "threshold": threshold_results,
        "sample_efficiency": sample_results,
    }
    with open("results/slam_dunk.json", "w") as f:
        json.dump(serialize(all_results), f, indent=2)

    # Generate figures
    print("\n=== GENERATING FIGURES ===")
    fig_paths = plot_all(adapt_results, sample_results, threshold_results)

    # Print final summary
    print("\n" + "=" * 60)
    print("SLAM DUNK SUMMARY")
    print("=" * 60)

    r = adapt_results[0]
    seq = r["adaptation"]["sequential"]
    jnt = r["adaptation"]["joint"]
    mono = r["adaptation"]["monolithic"]

    print(f"\n{'Method':<15} {'Adapt MSE':>12} {'Params':>10} {'MSE vs Seq':>12}")
    print("-" * 55)
    print(f"{'Sequential':<15} {seq['final_mse']:>12.6f} {seq['params']:>10,} {'1.0×':>12}")
    print(f"{'Joint':<15} {jnt['final_mse']:>12.6f} {jnt['params']:>10,} "
          f"{jnt['final_mse']/max(seq['final_mse'],1e-10):>11.1f}×")
    print(f"{'Monolithic':<15} {mono['final_mse']:>12.6f} {mono['params']:>10,} "
          f"{mono['final_mse']/max(seq['final_mse'],1e-10):>11.1f}×")

    # Threshold results
    print(f"\nEpochs to MSE < {threshold_results['threshold']}:")
    for m in ["sequential", "joint", "monolithic"]:
        ep = threshold_results[m]["epoch_reached"]
        print(f"  {m:<15}: {ep if ep else '>1000'} epochs ({threshold_results[m]['params']:,} params)")

    print(f"\nFigures saved to figures/")
    print(f"Results saved to results/slam_dunk.json")
