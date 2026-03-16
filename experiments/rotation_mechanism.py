"""Experiment: Mechanistic Analysis of Rotation (In)variance.

WHY does SiLU achieve rotation invariance while ReLU doesn't?

First attempt (comparing Jacobians of independently-trained networks) showed
null results — because different networks trained on different channels have
uncorrelated weights regardless of activation function.

Correct approach: The mechanism operates through SMOOTHNESS. For a single
trained network, smoothly rotate the input and measure how smoothly the output
changes. ReLU creates discontinuities (neurons flipping on/off) while SiLU
produces smooth output trajectories.

Specifically:
1. Train a single Receiver on a fixed channel
2. Take a test input x, construct a 1-parameter family x(θ) = R(θ)·x
   where R(θ) is a rotation in a randomly chosen 2D plane
3. Compute f(x(θ)) for θ ∈ [0, 2π] and measure:
   - Output trajectory smoothness (total variation)
   - Output norm variation
   - Number of "kinks" (high-curvature points where ReLU neurons flip)
4. Compare ReLU vs SiLU

For ReLU: f(x(θ)) should show kinks at angles where neurons switch on/off
For SiLU: f(x(θ)) should be smooth everywhere
"""

import os
import sys

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from experiments.pure_rotational_invariance import make_channel_from_svd


class AnalysisMLP(nn.Module):
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


def make_rotation_matrix(dim, plane_i, plane_j, theta):
    """Rotation by angle theta in the (i,j) plane, identity elsewhere."""
    R = torch.eye(dim)
    R[plane_i, plane_i] = torch.cos(theta)
    R[plane_i, plane_j] = -torch.sin(theta)
    R[plane_j, plane_i] = torch.sin(theta)
    R[plane_j, plane_j] = torch.cos(theta)
    return R


def train_receiver(a2s, env, dim, hidden_dim, activation, cfg_epochs, cfg_samples, lr=1e-3):
    """Train and return a Receiver."""
    torch.manual_seed(42)
    receiver = AnalysisMLP(dim, dim, hidden_dim, activation=activation)
    sounds = torch.randn(cfg_samples, dim)
    with torch.no_grad():
        received = env(a2s(sounds))

    optimizer = torch.optim.Adam(receiver.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for _ in range(cfg_epochs):
        perm = torch.randperm(sounds.size(0))
        for i in range(0, sounds.size(0), 64):
            idx = perm[i:i+64]
            loss = loss_fn(receiver(received[idx]), sounds[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return receiver


def trace_rotation(receiver, base_input, dim, plane_i, plane_j, n_angles=360):
    """Trace the output as input is continuously rotated in one plane."""
    thetas = torch.linspace(0, 2 * np.pi, n_angles)
    outputs = []
    receiver.eval()
    with torch.no_grad():
        for theta in thetas:
            R = make_rotation_matrix(dim, plane_i, plane_j, theta)
            rotated = (R @ base_input.unsqueeze(-1)).squeeze(-1).unsqueeze(0)
            out = receiver(rotated).squeeze(0)
            outputs.append(out)
    return thetas.numpy(), torch.stack(outputs)


def compute_smoothness_metrics(thetas, outputs):
    """Compute smoothness metrics from an output trajectory."""
    # Output as numpy
    out_np = outputs.numpy()

    # 1. Total variation of output norm
    norms = np.linalg.norm(out_np, axis=-1)
    norm_tv = np.sum(np.abs(np.diff(norms)))

    # 2. Total variation of each output dimension (mean)
    dim_tvs = np.mean([np.sum(np.abs(np.diff(out_np[:, d]))) for d in range(out_np.shape[-1])])

    # 3. Curvature: second derivative magnitude (mean across trajectory)
    dt = thetas[1] - thetas[0]
    d2_out = np.diff(out_np, n=2, axis=0) / (dt ** 2)
    curvature = np.linalg.norm(d2_out, axis=-1)
    mean_curvature = curvature.mean()
    max_curvature = curvature.max()

    # 4. Number of "kinks" — points where curvature exceeds 3× the median
    median_curv = np.median(curvature)
    n_kinks = np.sum(curvature > 3 * median_curv) if median_curv > 0 else 0

    return {
        "norm_tv": norm_tv,
        "dim_tv": dim_tvs,
        "mean_curvature": mean_curvature,
        "max_curvature": max_curvature,
        "n_kinks": int(n_kinks),
        "curvature": curvature,
        "norms": norms,
    }


def run_experiment(dim=8, n_test_inputs=10, n_planes=3):
    print("=" * 60)
    print("EXPERIMENT: Rotation Smoothness Mechanism")
    print("Does SiLU produce smoother output under input rotation?")
    print("=" * 60)

    sigmas = torch.logspace(0, -1, dim)  # κ=10
    kappa = (sigmas[0] / sigmas[-1]).item()

    # One fixed channel
    a2s, env, M = make_channel_from_svd(dim, sigmas, seed_u=1000, seed_v=2000)

    activations = ["relu", "silu"]
    results = {}

    for act in activations:
        print(f"\n  Training {act.upper()} Receiver...")
        receiver = train_receiver(a2s, env, dim, 64, act, 100, 1000)

        # Evaluate MSE
        torch.manual_seed(99)
        test = torch.randn(500, dim)
        with torch.no_grad():
            received = env(a2s(test))
            mse = (receiver(received) - test).pow(2).mean().item()
        print(f"  {act.upper()} MSE: {mse:.6f}")

        # Generate test inputs (pass through channel first)
        torch.manual_seed(42)
        base_inputs = []
        for i in range(n_test_inputs):
            s = torch.randn(dim)
            with torch.no_grad():
                r = env(a2s(s.unsqueeze(0))).squeeze(0)
            base_inputs.append(r)

        # Choose rotation planes
        planes = [(0, 1), (2, 3), (0, dim-1)][:n_planes]

        all_metrics = []
        all_trajectories = []

        for inp_idx, base_input in enumerate(base_inputs):
            for pi, pj in planes:
                thetas, outputs = trace_rotation(receiver, base_input, dim, pi, pj)
                metrics = compute_smoothness_metrics(thetas, outputs)
                metrics["input_idx"] = inp_idx
                metrics["plane"] = (pi, pj)
                all_metrics.append(metrics)
                all_trajectories.append((thetas, outputs, metrics))

        # Aggregate
        mean_norm_tv = np.mean([m["norm_tv"] for m in all_metrics])
        mean_dim_tv = np.mean([m["dim_tv"] for m in all_metrics])
        mean_curvature = np.mean([m["mean_curvature"] for m in all_metrics])
        mean_max_curvature = np.mean([m["max_curvature"] for m in all_metrics])
        mean_kinks = np.mean([m["n_kinks"] for m in all_metrics])

        results[act] = {
            "mse": mse,
            "metrics": all_metrics,
            "trajectories": all_trajectories,
            "mean_norm_tv": mean_norm_tv,
            "mean_dim_tv": mean_dim_tv,
            "mean_curvature": mean_curvature,
            "mean_max_curvature": mean_max_curvature,
            "mean_kinks": mean_kinks,
        }

        print(f"  Norm TV: {mean_norm_tv:.2f}")
        print(f"  Mean curvature: {mean_curvature:.2f}")
        print(f"  Max curvature: {mean_max_curvature:.2f}")
        print(f"  Mean kinks: {mean_kinks:.1f}")

    return results, dim, kappa


def plot_results(results, dim, kappa, output_path):
    acts = ["relu", "silu"]
    colors = {"relu": "#E91E63", "silu": "#4CAF50"}

    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.35)

    # --- Row 1: Example trajectories (output norm vs rotation angle) ---
    for col, act in enumerate(acts):
        ax = fig.add_subplot(gs[0, col * 2:col * 2 + 2])
        trajs = results[act]["trajectories"]
        for i, (thetas, outputs, metrics) in enumerate(trajs[:6]):
            norms = metrics["norms"]
            alpha = 0.7 if i < 3 else 0.3
            ax.plot(np.degrees(thetas), norms, color=colors[act], alpha=alpha,
                    linewidth=1.0)
        ax.set_xlabel("Rotation Angle (degrees)")
        ax.set_ylabel("||f(R(θ)·x)||")
        ax.set_title(f"{act.upper()}: Output Norm Under Rotation\n"
                     f"(TV={results[act]['mean_norm_tv']:.1f})")
        ax.grid(True, alpha=0.3)

    # --- Row 2: Curvature profiles ---
    for col, act in enumerate(acts):
        ax = fig.add_subplot(gs[1, col * 2:col * 2 + 2])
        trajs = results[act]["trajectories"]
        for i, (thetas, outputs, metrics) in enumerate(trajs[:6]):
            curv = metrics["curvature"]
            dt = thetas[1] - thetas[0]
            angles = np.degrees(thetas[1:-1])
            alpha = 0.7 if i < 3 else 0.3
            ax.plot(angles, curv, color=colors[act], alpha=alpha, linewidth=0.8)
        ax.set_xlabel("Rotation Angle (degrees)")
        ax.set_ylabel("Curvature (||d²f/dθ²||)")
        ax.set_title(f"{act.upper()}: Curvature Profile\n"
                     f"(mean={results[act]['mean_curvature']:.1f}, "
                     f"kinks={results[act]['mean_kinks']:.0f})")
        ax.grid(True, alpha=0.3)

    # --- Row 3: Comparison bar charts + mechanism explanation ---

    # Smoothness metrics comparison
    ax = fig.add_subplot(gs[2, 0])
    metrics_names = ["mean_norm_tv", "mean_dim_tv"]
    labels = ["Norm TV", "Dim-wise TV"]
    x = np.arange(len(labels))
    width = 0.35
    for i, act in enumerate(acts):
        vals = [results[act][m] for m in metrics_names]
        ax.bar(x + i * width, vals, width, color=colors[act], alpha=0.85,
               label=act.upper())
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Total Variation")
    ax.set_title("Output Smoothness\n(lower = smoother)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Curvature comparison
    ax = fig.add_subplot(gs[2, 1])
    metrics_names = ["mean_curvature", "mean_max_curvature"]
    labels = ["Mean Curv.", "Max Curv."]
    x = np.arange(len(labels))
    for i, act in enumerate(acts):
        vals = [results[act][m] for m in metrics_names]
        ax.bar(x + i * width, vals, width, color=colors[act], alpha=0.85,
               label=act.upper())
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Curvature")
    ax.set_title("Trajectory Curvature\n(lower = fewer kinks)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Kink count
    ax = fig.add_subplot(gs[2, 2])
    for i, act in enumerate(acts):
        ax.bar(i, results[act]["mean_kinks"], color=colors[act], alpha=0.85)
        ax.text(i, results[act]["mean_kinks"] + 0.5,
                f"{results[act]['mean_kinks']:.0f}", ha="center",
                fontsize=12, fontweight="bold")
    ax.set_xticks(range(len(acts)))
    ax.set_xticklabels([a.upper() for a in acts])
    ax.set_ylabel("Mean Kink Count")
    ax.set_title("Number of Kinks\n(curvature > 3× median)")
    ax.grid(True, alpha=0.3, axis="y")

    # Mechanism summary
    ax = fig.add_subplot(gs[2, 3])
    ax.axis("off")

    relu_r = results["relu"]
    silu_r = results["silu"]
    tv_ratio = relu_r["mean_norm_tv"] / max(silu_r["mean_norm_tv"], 1e-10)
    curv_ratio = relu_r["mean_max_curvature"] / max(silu_r["mean_max_curvature"], 1e-10)
    kink_ratio = relu_r["mean_kinks"] / max(silu_r["mean_kinks"], 1)

    text = (
        f"MECHANISM SUMMARY\n"
        f"{'─' * 30}\n\n"
        f"Output smoothness (TV):\n"
        f"  ReLU {tv_ratio:.1f}× rougher than SiLU\n\n"
        f"Peak curvature:\n"
        f"  ReLU {curv_ratio:.1f}× spikier than SiLU\n\n"
        f"Kinks (discontinuities):\n"
        f"  ReLU: {relu_r['mean_kinks']:.0f} per trajectory\n"
        f"  SiLU: {silu_r['mean_kinks']:.0f} per trajectory\n\n"
        f"ReLU creates sharp kinks at\n"
        f"angles where neurons flip.\n"
        f"SiLU varies smoothly —\n"
        f"no preferred orientations."
    )
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5", alpha=0.9))

    fig.suptitle(
        f"Why SiLU Achieves Rotation Invariance (dim={dim}, κ={kappa:.0f})\n"
        f"Output smoothness under continuous input rotation",
        fontsize=14, fontweight="bold"
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved: {output_path}")


def print_summary(results):
    relu = results["relu"]
    silu = results["silu"]

    print("\n" + "=" * 70)
    print("SUMMARY: Rotation Smoothness Mechanism")
    print("=" * 70)

    tv_ratio = relu["mean_norm_tv"] / max(silu["mean_norm_tv"], 1e-10)
    curv_ratio = relu["mean_max_curvature"] / max(silu["mean_max_curvature"], 1e-10)

    print(f"\n  {'Metric':<25} | {'ReLU':>10} | {'SiLU':>10} | {'Ratio':>8}")
    print(f"  {'-'*25}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")
    print(f"  {'Norm Total Variation':<25} | {relu['mean_norm_tv']:>10.2f} | {silu['mean_norm_tv']:>10.2f} | {tv_ratio:>7.1f}×")
    print(f"  {'Mean Curvature':<25} | {relu['mean_curvature']:>10.2f} | {silu['mean_curvature']:>10.2f} | {relu['mean_curvature']/max(silu['mean_curvature'],1e-10):>7.1f}×")
    print(f"  {'Max Curvature':<25} | {relu['mean_max_curvature']:>10.2f} | {silu['mean_max_curvature']:>10.2f} | {curv_ratio:>7.1f}×")
    print(f"  {'Kinks per trajectory':<25} | {relu['mean_kinks']:>10.0f} | {silu['mean_kinks']:>10.0f} | {relu['mean_kinks']/max(silu['mean_kinks'],1):>7.1f}×")
    print(f"  {'Reconstruction MSE':<25} | {relu['mse']:>10.6f} | {silu['mse']:>10.6f} | {relu['mse']/max(silu['mse'],1e-10):>7.1f}×")

    print(f"\n  CONCLUSION:")
    if tv_ratio > 1.2:
        print(f"  ReLU output is {tv_ratio:.1f}× rougher under rotation, with {curv_ratio:.1f}× spikier curvature.")
        print(f"  This confirms the mechanism: ReLU's binary neuron switching creates")
        print(f"  orientation-dependent kinks that make the loss landscape rotation-sensitive.")
        print(f"  SiLU's smooth gating produces continuous output trajectories.")
    else:
        print(f"  Smoothness difference is small ({tv_ratio:.1f}×). The mechanism may operate")
        print(f"  through the optimization landscape rather than the function itself.")


def main():
    os.makedirs("results", exist_ok=True)
    results, dim, kappa = run_experiment(dim=8, n_test_inputs=10, n_planes=3)
    plot_results(results, dim, kappa, "results/obj-023-rotation-mechanism.png")
    print_summary(results)


if __name__ == "__main__":
    main()
