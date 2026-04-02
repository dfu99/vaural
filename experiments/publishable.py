"""
Publishable Experiments: All four reviewer-demanded experiments.

1. Param-count ablation: Does the 14× speedup survive when joint trains
   the same number of params? (freeze receiver at random init)
2. Multi-seed with error bars at dim=8 and dim=16
3. Structured channels: noisy, ill-conditioned, bandlimited
4. Gradient conflict: cosine similarity of emitter vs receiver gradients
"""

import torch
import torch.nn as nn
import numpy as np
import json
import os
import sys
import time
from collections import defaultdict

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

COLORS = {
    'sequential': '#2196F3', 'joint': '#FF9800', 'joint_matched': '#9C27B0',
    'monolithic': '#F44336',
}


# ─── Helpers ──────────────────────────────────────────────────────────────

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_channel(dim, a2s_seed=100, env_seed=200, channel_type="random",
                 noise_std=0.0, condition_number=None, rank=None):
    """Create channel components with optional modifications."""
    a2s = ActionToSignal(dim, dim, seed=a2s_seed)
    env = Environment(dim, seed=env_seed)

    if condition_number is not None:
        # Reshape singular values to target condition number
        M = env.weight @ a2s.weight
        U, S, Vh = torch.linalg.svd(M)
        S_new = torch.linspace(condition_number, 1.0, dim)
        M_new = U @ torch.diag(S_new) @ Vh
        # Distribute back (approximately) — put it all in env
        a2s.weight.copy_(torch.eye(dim))
        env.weight.copy_(M_new)

    if rank is not None and rank < dim:
        # Bandlimited: zero out smallest singular values
        M = env.weight @ a2s.weight
        U, S, Vh = torch.linalg.svd(M)
        S[rank:] = 0.0
        M_new = U @ torch.diag(S) @ Vh
        a2s.weight.copy_(torch.eye(dim))
        env.weight.copy_(M_new)

    return a2s, env, noise_std


class NoisyPipeline(nn.Module):
    """Pipeline with optional additive Gaussian noise in the channel."""
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


def pretrain_receiver(dim, hidden, a2s, env, noise_std, epochs, n_samples, seed=42):
    """Pre-train receiver to invert channel. Returns trained receiver."""
    torch.manual_seed(seed)
    recv = Receiver(dim, dim, hidden)
    sounds = torch.randn(n_samples, dim)
    with torch.no_grad():
        received = env(a2s(sounds))
        if noise_std > 0:
            received = received + torch.randn_like(received) * noise_std

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
    """Generic training loop returning (epoch, mse) curve."""
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


# ─── Experiment 1: Param-Count Ablation ──────────────────────────────────

def exp1_param_ablation(dim=8, hidden=64, epochs=1000, n_samples=5000, seeds=range(42, 45)):
    """
    Key ablation: "joint_matched" freezes receiver at random init and trains
    only emitter — same param count as sequential but no pre-trained receiver.
    If sequential still wins, it's the landscape quality, not param count.
    """
    print("\n" + "="*60)
    print("EXP 1: PARAM-COUNT ABLATION")
    print("="*60)

    threshold = 0.001
    all_results = {m: [] for m in ["sequential", "joint", "joint_matched"]}

    for seed in seeds:
        print(f"  seed={seed}...")
        torch.manual_seed(seed)
        test_sounds = torch.randn(2000, dim)
        sounds = torch.randn(n_samples, dim)
        a2s_seed, env_seed = 300, 400  # Channel B

        # Sequential: pre-train receiver, freeze, train emitter
        a2s, env, _ = make_channel(dim, a2s_seed, env_seed)
        recv = pretrain_receiver(dim, hidden, a2s, env, 0.0, epochs, n_samples, seed)
        emit = Emitter(dim, dim, hidden)
        torch.manual_seed(seed + 1000)  # different init for emitter
        emit = Emitter(dim, dim, hidden)
        pipe = NoisyPipeline(emit, a2s, env, recv)
        pipe.receiver.requires_grad_(False)
        pipe.action_to_signal.requires_grad_(False)
        pipe.environment.requires_grad_(False)
        curve_seq = run_training(pipe, list(emit.parameters()), sounds, test_sounds, epochs)
        all_results["sequential"].append(curve_seq)

        # Joint: train both emitter + receiver from scratch
        a2s2, env2, _ = make_channel(dim, a2s_seed, env_seed)
        recv2 = Receiver(dim, dim, hidden)
        torch.manual_seed(seed + 1000)
        emit2 = Emitter(dim, dim, hidden)
        pipe2 = NoisyPipeline(emit2, a2s2, env2, recv2)
        params_joint = list(emit2.parameters()) + list(recv2.parameters())
        curve_jnt = run_training(pipe2, params_joint, sounds, test_sounds, epochs)
        all_results["joint"].append(curve_jnt)

        # Joint-matched: freeze receiver at RANDOM init, train only emitter
        # Same param count as sequential, but no pre-trained receiver
        a2s3, env3, _ = make_channel(dim, a2s_seed, env_seed)
        recv3 = Receiver(dim, dim, hidden)  # random init, NOT pre-trained
        torch.manual_seed(seed + 1000)
        emit3 = Emitter(dim, dim, hidden)
        pipe3 = NoisyPipeline(emit3, a2s3, env3, recv3)
        pipe3.receiver.requires_grad_(False)
        pipe3.action_to_signal.requires_grad_(False)
        pipe3.environment.requires_grad_(False)
        curve_matched = run_training(pipe3, list(emit3.parameters()), sounds, test_sounds, epochs)
        all_results["joint_matched"].append(curve_matched)

    # Compute stats
    results = {}
    for method, curves in all_results.items():
        # Align on common epochs
        common_epochs = [ep for ep, _ in curves[0]]
        mse_matrix = []
        for curve in curves:
            mses = [m for _, m in curve]
            mse_matrix.append(mses)
        mse_matrix = np.array(mse_matrix)

        thresh_epochs = []
        for curve in curves:
            te = find_threshold_epoch(curve, threshold)
            thresh_epochs.append(te if te else epochs + 1)

        results[method] = {
            "mean_curve": mse_matrix.mean(axis=0).tolist(),
            "std_curve": mse_matrix.std(axis=0).tolist(),
            "epochs": common_epochs,
            "thresh_mean": float(np.mean(thresh_epochs)),
            "thresh_std": float(np.std(thresh_epochs)),
            "final_mse_mean": float(mse_matrix[:, -1].mean()),
            "final_mse_std": float(mse_matrix[:, -1].std()),
            "params": count_params(emit),  # same for seq and matched
        }

    results["joint"]["params"] = count_params(emit) + count_params(recv)
    results["threshold"] = threshold
    results["n_seeds"] = len(list(seeds))

    for m in ["sequential", "joint", "joint_matched"]:
        r = results[m]
        print(f"  {m:<20}: MSE={r['final_mse_mean']:.6f}±{r['final_mse_std']:.6f}, "
              f"thresh={r['thresh_mean']:.0f}±{r['thresh_std']:.0f} ep, params={r['params']}")

    return results


# ─── Experiment 2: Multi-Seed at dim=8 and dim=16 ────────────────────────

def exp2_multiseed(dims_and_hidden=[(8, 64), (16, 128)], epochs_map={8: 1000, 16: 1500},
                   n_samples=5000, seeds=range(42, 45)):
    """Run threshold experiment across seeds and dimensions."""
    print("\n" + "="*60)
    print("EXP 2: MULTI-SEED SCALING")
    print("="*60)

    threshold = 0.001
    all_results = {}

    for dim, hidden in dims_and_hidden:
        epochs = epochs_map.get(dim, 1500)
        print(f"\n  dim={dim}, hidden={hidden}, epochs={epochs}")
        dim_results = {m: [] for m in ["sequential", "joint"]}

        for seed in seeds:
            print(f"    seed={seed}...", end=" ", flush=True)
            torch.manual_seed(seed)
            test_sounds = torch.randn(2000, dim)
            sounds = torch.randn(n_samples, dim)
            a2s_seed, env_seed = 300, 400

            # Sequential
            a2s, env, _ = make_channel(dim, a2s_seed, env_seed)
            recv = pretrain_receiver(dim, hidden, a2s, env, 0.0, epochs, n_samples, seed)
            torch.manual_seed(seed + 1000)
            emit = Emitter(dim, dim, hidden)
            pipe = NoisyPipeline(emit, a2s, env, recv)
            pipe.receiver.requires_grad_(False)
            pipe.action_to_signal.requires_grad_(False)
            pipe.environment.requires_grad_(False)
            curve_seq = run_training(pipe, list(emit.parameters()), sounds, test_sounds, epochs)
            dim_results["sequential"].append(curve_seq)

            # Joint
            a2s2, env2, _ = make_channel(dim, a2s_seed, env_seed)
            recv2 = Receiver(dim, dim, hidden)
            torch.manual_seed(seed + 1000)
            emit2 = Emitter(dim, dim, hidden)
            pipe2 = NoisyPipeline(emit2, a2s2, env2, recv2)
            params = list(emit2.parameters()) + list(recv2.parameters())
            curve_jnt = run_training(pipe2, params, sounds, test_sounds, epochs)
            dim_results["joint"].append(curve_jnt)
            print(f"seq={curve_seq[-1][1]:.6f}, jnt={curve_jnt[-1][1]:.6f}")

        # Aggregate
        results_dim = {}
        for method, curves in dim_results.items():
            common_epochs = [ep for ep, _ in curves[0]]
            mse_matrix = np.array([[m for _, m in c] for c in curves])
            thresh_epochs = []
            for c in curves:
                te = find_threshold_epoch(c, threshold)
                thresh_epochs.append(te if te else epochs + 1)

            results_dim[method] = {
                "mean_curve": mse_matrix.mean(axis=0).tolist(),
                "std_curve": mse_matrix.std(axis=0).tolist(),
                "epochs": common_epochs,
                "thresh_mean": float(np.mean(thresh_epochs)),
                "thresh_std": float(np.std(thresh_epochs)),
                "final_mse_mean": float(mse_matrix[:, -1].mean()),
                "final_mse_std": float(mse_matrix[:, -1].std()),
            }
            print(f"    {method} dim={dim}: MSE={results_dim[method]['final_mse_mean']:.6f}"
                  f"±{results_dim[method]['final_mse_std']:.6f}, "
                  f"thresh={results_dim[method]['thresh_mean']:.0f}"
                  f"±{results_dim[method]['thresh_std']:.0f}")

        all_results[f"dim{dim}"] = results_dim

    all_results["threshold"] = threshold
    return all_results


# ─── Experiment 3: Structured Channels ───────────────────────────────────

def exp3_structured_channels(dim=8, hidden=64, epochs=1000, n_samples=5000, seeds=range(42, 45)):
    """Test adaptation advantage across channel types."""
    print("\n" + "="*60)
    print("EXP 3: STRUCTURED CHANNELS")
    print("="*60)

    threshold = 0.001
    channel_configs = {
        "random": {"channel_type": "random"},
        "noisy_0.1": {"noise_std": 0.1},
        "noisy_0.5": {"noise_std": 0.5},
        "ill_cond_100": {"condition_number": 100.0},
        "bandlimited_4": {"rank": 4},  # 4 of 8 dims preserved
    }

    all_results = {}
    for ch_name, ch_kwargs in channel_configs.items():
        print(f"\n  Channel: {ch_name}")
        ch_results = {m: [] for m in ["sequential", "joint"]}

        for seed in seeds:
            torch.manual_seed(seed)
            test_sounds = torch.randn(2000, dim)
            sounds = torch.randn(n_samples, dim)

            noise_std = ch_kwargs.get("noise_std", 0.0)
            cond = ch_kwargs.get("condition_number", None)
            rank = ch_kwargs.get("rank", None)

            # Sequential
            a2s, env, _ = make_channel(dim, 300, 400, condition_number=cond, rank=rank)
            recv = pretrain_receiver(dim, hidden, a2s, env, noise_std, epochs, n_samples, seed)
            torch.manual_seed(seed + 1000)
            emit = Emitter(dim, dim, hidden)
            pipe = NoisyPipeline(emit, a2s, env, recv, noise_std)
            pipe.receiver.requires_grad_(False)
            pipe.action_to_signal.requires_grad_(False)
            pipe.environment.requires_grad_(False)
            curve_seq = run_training(pipe, list(emit.parameters()), sounds, test_sounds, epochs)
            ch_results["sequential"].append(curve_seq)

            # Joint
            a2s2, env2, _ = make_channel(dim, 300, 400, condition_number=cond, rank=rank)
            recv2 = Receiver(dim, dim, hidden)
            torch.manual_seed(seed + 1000)
            emit2 = Emitter(dim, dim, hidden)
            pipe2 = NoisyPipeline(emit2, a2s2, env2, recv2, noise_std)
            params = list(emit2.parameters()) + list(recv2.parameters())
            curve_jnt = run_training(pipe2, params, sounds, test_sounds, epochs)
            ch_results["joint"].append(curve_jnt)

        # Aggregate
        results_ch = {}
        for method, curves in ch_results.items():
            mse_matrix = np.array([[m for _, m in c] for c in curves])
            thresh_epochs = []
            for c in curves:
                te = find_threshold_epoch(c, threshold)
                thresh_epochs.append(te if te else epochs + 1)
            results_ch[method] = {
                "mean_curve": mse_matrix.mean(axis=0).tolist(),
                "std_curve": mse_matrix.std(axis=0).tolist(),
                "epochs": [ep for ep, _ in curves[0]],
                "thresh_mean": float(np.mean(thresh_epochs)),
                "thresh_std": float(np.std(thresh_epochs)),
                "final_mse_mean": float(mse_matrix[:, -1].mean()),
                "final_mse_std": float(mse_matrix[:, -1].std()),
            }

        all_results[ch_name] = results_ch
        seq_t = results_ch["sequential"]["thresh_mean"]
        jnt_t = results_ch["joint"]["thresh_mean"]
        speedup = jnt_t / max(seq_t, 1)
        print(f"    Sequential: thresh={seq_t:.0f}±{results_ch['sequential']['thresh_std']:.0f}, "
              f"MSE={results_ch['sequential']['final_mse_mean']:.6f}")
        print(f"    Joint:      thresh={jnt_t:.0f}±{results_ch['joint']['thresh_std']:.0f}, "
              f"MSE={results_ch['joint']['final_mse_mean']:.6f}")
        print(f"    Speedup: {speedup:.1f}×")

    return all_results


# ─── Experiment 4: Gradient Conflict ─────────────────────────────────────

def exp4_gradient_conflict(dim=8, hidden=64, epochs=500, n_samples=5000):
    """Measure gradient conflict between emitter and receiver during joint training."""
    print("\n" + "="*60)
    print("EXP 4: GRADIENT CONFLICT")
    print("="*60)

    torch.manual_seed(42)
    a2s, env, _ = make_channel(dim, 300, 400)
    recv = Receiver(dim, dim, hidden)
    emit = Emitter(dim, dim, hidden)
    pipe = NoisyPipeline(emit, a2s, env, recv)

    sounds = torch.randn(n_samples, dim)
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(list(emit.parameters()) + list(recv.parameters()), lr=1e-3)

    cosine_sims = []
    emit_grad_norms = []
    recv_grad_norms = []
    losses = []
    eval_epochs = []

    for epoch in range(epochs):
        pipe.train()
        perm = torch.randperm(n_samples)
        epoch_cos = []
        epoch_loss = 0
        n_batch = 0

        for i in range(0, n_samples, 64):
            idx = perm[i:i+64]
            loss = loss_fn(pipe(sounds[idx]), sounds[idx])
            opt.zero_grad()
            loss.backward()

            # Collect gradients
            emit_grads = []
            recv_grads = []
            for p in emit.parameters():
                if p.grad is not None:
                    emit_grads.append(p.grad.flatten())
            for p in recv.parameters():
                if p.grad is not None:
                    recv_grads.append(p.grad.flatten())

            if emit_grads and recv_grads:
                eg = torch.cat(emit_grads)
                rg = torch.cat(recv_grads)
                # Pad to same length for cosine sim
                max_len = max(eg.numel(), rg.numel())
                eg_padded = torch.zeros(max_len)
                rg_padded = torch.zeros(max_len)
                eg_padded[:eg.numel()] = eg
                rg_padded[:rg.numel()] = rg
                cos = torch.nn.functional.cosine_similarity(
                    eg_padded.unsqueeze(0), rg_padded.unsqueeze(0)
                ).item()
                epoch_cos.append(cos)

            opt.step()
            epoch_loss += loss.item()
            n_batch += 1

        if (epoch + 1) % 5 == 0:
            cosine_sims.append(float(np.mean(epoch_cos)))
            eg_norm = sum(p.grad.norm().item()**2 for p in emit.parameters() if p.grad is not None)**0.5
            rg_norm = sum(p.grad.norm().item()**2 for p in recv.parameters() if p.grad is not None)**0.5
            emit_grad_norms.append(eg_norm)
            recv_grad_norms.append(rg_norm)
            losses.append(epoch_loss / n_batch)
            eval_epochs.append(epoch + 1)

    results = {
        "epochs": eval_epochs,
        "cosine_similarity": cosine_sims,
        "emit_grad_norm": emit_grad_norms,
        "recv_grad_norm": recv_grad_norms,
        "loss": losses,
    }

    print(f"  Early cos_sim (ep 5-25): {np.mean(cosine_sims[:5]):.4f}")
    print(f"  Mid cos_sim (ep 100-200): {np.mean(cosine_sims[20:40]):.4f}")
    print(f"  Late cos_sim (ep 400-500): {np.mean(cosine_sims[-20:]):.4f}")

    return results


# ─── Plotting ────────────────────────────────────────────────────────────

def plot_all(exp1, exp2, exp3, exp4, save_dir="figures"):
    os.makedirs(save_dir, exist_ok=True)
    paths = []

    # --- Fig 1: Param Ablation (THE key figure) ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    for method, color, label in [
        ("sequential", COLORS["sequential"], "Sequential (pre-trained recv)"),
        ("joint", COLORS["joint"], "Joint (both trainable)"),
        ("joint_matched", COLORS["joint_matched"], "Matched-params (random recv, frozen)"),
    ]:
        r = exp1[method]
        epochs = r["epochs"]
        mean = np.array(r["mean_curve"])
        std = np.array(r["std_curve"])
        ax.semilogy(epochs, mean, '-', color=color, label=label, linewidth=2)
        ax.fill_between(epochs, mean - std, mean + std, alpha=0.15, color=color)
    ax.axhline(y=exp1["threshold"], color='gray', ls='--', alpha=0.5, label='Threshold')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Test MSE (log)')
    ax.set_title('Param-Count Ablation: Is It Just Fewer Params?')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax = axes[1]
    methods = ["sequential", "joint", "joint_matched"]
    labels = ["Sequential\n(pre-trained)", "Joint\n(2× params)", "Matched\n(random recv)"]
    colors = [COLORS[m] for m in methods]
    thresh_means = [exp1[m]["thresh_mean"] for m in methods]
    thresh_stds = [exp1[m]["thresh_std"] for m in methods]
    bars = ax.bar(range(3), thresh_means, yerr=thresh_stds, color=colors,
                  edgecolor='white', capsize=5)
    for bar, tm, ts in zip(bars, thresh_means, thresh_stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + ts + 10,
                f'{tm:.0f}±{ts:.0f}', ha='center', fontsize=9, fontweight='bold')
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels)
    ax.set_ylabel(f'Epochs to MSE < {exp1["threshold"]}')
    ax.set_title('Convergence Speed (5 seeds)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    p = os.path.join(save_dir, "pub_param_ablation.png")
    fig.savefig(p, bbox_inches='tight', dpi=150)
    paths.append(p)
    print(f"Saved {p}")

    # --- Fig 2: Multi-seed scaling ---
    dims_available = [k for k in exp2.keys() if k.startswith("dim")]
    fig, axes = plt.subplots(1, len(dims_available), figsize=(6*len(dims_available), 5))
    if len(dims_available) == 1:
        axes = [axes]

    for i, dk in enumerate(sorted(dims_available)):
        ax = axes[i]
        dim_data = exp2[dk]
        for method, color, label in [
            ("sequential", COLORS["sequential"], "Sequential"),
            ("joint", COLORS["joint"], "Joint"),
        ]:
            r = dim_data[method]
            epochs = r["epochs"]
            mean = np.array(r["mean_curve"])
            std = np.array(r["std_curve"])
            ax.semilogy(epochs, mean, '-', color=color, label=label, linewidth=2)
            ax.fill_between(epochs, np.maximum(mean - std, 1e-8), mean + std,
                           alpha=0.15, color=color)
        ax.axhline(y=exp2["threshold"], color='gray', ls='--', alpha=0.5)
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Test MSE (log)' if i == 0 else '')
        ax.set_title(f'{dk} (mean±std)', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    p = os.path.join(save_dir, "pub_multiseed_scaling.png")
    fig.savefig(p, bbox_inches='tight', dpi=150)
    paths.append(p)
    print(f"Saved {p}")

    # --- Fig 3: Structured channels ---
    channel_names = [k for k in exp3.keys() if k != "threshold"]
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(channel_names))
    width = 0.35
    seq_means = [exp3[c]["sequential"]["final_mse_mean"] for c in channel_names]
    seq_stds = [exp3[c]["sequential"]["final_mse_std"] for c in channel_names]
    jnt_means = [exp3[c]["joint"]["final_mse_mean"] for c in channel_names]
    jnt_stds = [exp3[c]["joint"]["final_mse_std"] for c in channel_names]

    ax.bar(x - width/2, seq_means, width, yerr=seq_stds, label='Sequential',
           color=COLORS["sequential"], edgecolor='white', capsize=3)
    ax.bar(x + width/2, jnt_means, width, yerr=jnt_stds, label='Joint',
           color=COLORS["joint"], edgecolor='white', capsize=3)

    # Add speedup annotations
    for i, ch in enumerate(channel_names):
        seq_t = exp3[ch]["sequential"]["thresh_mean"]
        jnt_t = exp3[ch]["joint"]["thresh_mean"]
        speedup = jnt_t / max(seq_t, 1)
        y_max = max(seq_means[i] + seq_stds[i], jnt_means[i] + jnt_stds[i])
        ax.text(i, y_max + 0.001, f'{speedup:.0f}×', ha='center', fontsize=8,
                fontweight='bold', color='#333')

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in channel_names], fontsize=8)
    ax.set_ylabel('Final Test MSE (mean±std, 5 seeds)')
    ax.set_title('Sequential Advantage Across Channel Types\n(number = convergence speedup)',
                fontweight='bold')
    ax.legend()
    ax.set_yscale('log')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    p = os.path.join(save_dir, "pub_structured_channels.png")
    fig.savefig(p, bbox_inches='tight', dpi=150)
    paths.append(p)
    print(f"Saved {p}")

    # --- Fig 4: Gradient conflict ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    ax = axes[0]
    ax.plot(exp4["epochs"], exp4["cosine_similarity"], '-', color='#333', linewidth=1.5)
    ax.axhline(y=0, color='gray', ls='--', alpha=0.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('Emitter-Receiver Gradient Alignment')
    ax.grid(True, alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax = axes[1]
    ax.semilogy(exp4["epochs"], exp4["emit_grad_norm"], '-', color=COLORS["sequential"],
               label='Emitter', linewidth=1.5)
    ax.semilogy(exp4["epochs"], exp4["recv_grad_norm"], '-', color=COLORS["joint"],
               label='Receiver', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Gradient Norm (log)')
    ax.set_title('Gradient Magnitudes')
    ax.legend()
    ax.grid(True, alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax = axes[2]
    ax.semilogy(exp4["epochs"], exp4["loss"], '-', color='#333', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss (log)')
    ax.set_title('Joint Training Loss')
    ax.grid(True, alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.suptitle('Gradient Conflict During Joint Training', fontweight='bold', y=1.02)
    plt.tight_layout()
    p = os.path.join(save_dir, "pub_gradient_conflict.png")
    fig.savefig(p, bbox_inches='tight', dpi=150)
    paths.append(p)
    print(f"Saved {p}")

    return paths


# ─── Main ────────────────────────────────────────────────────────────────

def serialize(obj):
    if isinstance(obj, dict):
        return {k: serialize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize(v) for v in obj]
    elif isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    return obj


def save_checkpoint(results, label="checkpoint"):
    """Save intermediate results so we don't lose progress."""
    with open(f"results/pub_{label}.json", "w") as f:
        json.dump(serialize(results), f, indent=2)
    print(f"  [checkpoint saved: results/pub_{label}.json]")


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    t_start = time.time()

    # Run all experiments with checkpointing
    exp1 = exp1_param_ablation()
    save_checkpoint({"exp1_ablation": exp1}, "exp1")

    exp2 = exp2_multiseed()
    save_checkpoint({"exp1_ablation": exp1, "exp2_scaling": exp2}, "exp2")

    exp3 = exp3_structured_channels()
    save_checkpoint({"exp1_ablation": exp1, "exp2_scaling": exp2,
                     "exp3_channels": exp3}, "exp3")

    exp4 = exp4_gradient_conflict()

    # Save final results
    all_results = {"exp1_ablation": exp1, "exp2_scaling": exp2,
                   "exp3_channels": exp3, "exp4_gradients": exp4}
    with open("results/publishable.json", "w") as f:
        json.dump(serialize(all_results), f, indent=2)

    # Plot
    print("\n=== GENERATING FIGURES ===")
    fig_paths = plot_all(exp1, exp2, exp3, exp4)

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed/60:.1f} minutes")
    print(f"Results: results/publishable.json")
    print(f"Figures: {fig_paths}")
