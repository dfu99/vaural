"""
Generate publication-quality figures from adaptation benchmark results.
"""

import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import sys

matplotlib.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 9,
    'figure.dpi': 150,
})

COLORS = {
    'sequential': '#2196F3',
    'sequential_fast': '#64B5F6',
    'joint': '#FF9800',
    'monolithic': '#F44336',
}

LABELS = {
    'sequential': 'Sequential (ours)',
    'sequential_full': 'Sequential (ours)',
    'sequential_fast': 'Sequential (fast)',
    'joint': 'Joint',
    'monolithic': 'Monolithic',
}


def load_results(path="results/adaptation_benchmark.json"):
    with open(path) as f:
        return json.load(f)


def fig1_adaptation_mse_comparison(results, save_dir="figures"):
    """Bar chart: adaptation MSE across dimensions for each method."""
    adapt_data = results["adaptation"]
    dims = [r["dim"] for r in adapt_data]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)

    for i, r in enumerate(adapt_data):
        ax = axes[i]
        adapt = r["adaptation"]

        methods = ["sequential_full", "sequential_fast", "joint", "monolithic"]
        mses = [adapt[m]["final_mse"] for m in methods]
        params = [adapt[m]["trainable_params"] for m in methods]
        colors = [COLORS.get(m, COLORS.get(m.replace("_full", ""), '#888')) for m in methods]
        labels = [LABELS[m] for m in methods]

        bars = ax.bar(range(len(methods)), mses, color=colors, edgecolor='white', linewidth=0.5)

        # Add param count labels on bars
        for bar, p in zip(bars, params):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mses)*0.02,
                    f'{p:,}p', ha='center', va='bottom', fontsize=8, color='#555')

        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=9)
        ax.set_title(f'dim={r["dim"]}', fontweight='bold')
        ax.set_ylabel('Test MSE (↓ better)' if i == 0 else '')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle('Channel Adaptation: MSE After Switching to New Channel', fontweight='bold', y=1.02)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "adaptation_mse_comparison.png")
    fig.savefig(path, bbox_inches='tight', dpi=150)
    print(f"Saved {path}")
    return path


def fig2_convergence_curves(results, save_dir="figures"):
    """Convergence curves during adaptation for each dimension."""
    adapt_data = results["adaptation"]

    fig, axes = plt.subplots(1, len(adapt_data), figsize=(14, 4.5), sharey=False)
    if len(adapt_data) == 1:
        axes = [axes]

    for i, r in enumerate(adapt_data):
        ax = axes[i]
        adapt = r["adaptation"]

        for method in ["sequential_full", "joint", "monolithic"]:
            epochs = adapt[method]["eval_epochs"]
            mses = adapt[method]["eval_mse"]
            color = COLORS.get(method, COLORS.get(method.replace("_full", ""), '#888'))
            ax.semilogy(epochs, mses, '-o', color=color, label=LABELS[method],
                       markersize=3, linewidth=1.5)

        ax.set_xlabel('Adaptation Epochs')
        ax.set_ylabel('Test MSE (log scale)' if i == 0 else '')
        ax.set_title(f'dim={r["dim"]}', fontweight='bold')
        ax.legend(loc='upper right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Convergence During Channel Adaptation', fontweight='bold', y=1.02)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "adaptation_convergence.png")
    fig.savefig(path, bbox_inches='tight', dpi=150)
    print(f"Saved {path}")
    return path


def fig3_parameter_efficiency(results, save_dir="figures"):
    """Scatter plot: MSE vs trainable parameters (the money shot)."""
    adapt_data = results["adaptation"]

    fig, ax = plt.subplots(figsize=(8, 6))

    for r in adapt_data:
        dim = r["dim"]
        adapt = r["adaptation"]

        for method in ["sequential_full", "sequential_fast", "joint", "monolithic"]:
            mse = adapt[method]["final_mse"]
            params = adapt[method]["trainable_params"]
            color = COLORS.get(method, COLORS.get(method.replace("_full", ""), '#888'))

            marker = {8: 'o', 16: 's', 32: 'D'}.get(dim, 'o')
            size = {8: 60, 16: 100, 32: 140}.get(dim, 80)

            ax.scatter(params, mse, c=color, marker=marker, s=size,
                      edgecolors='white', linewidths=0.5, zorder=5)

    # Custom legend for methods
    from matplotlib.lines import Line2D
    method_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['sequential'],
               markersize=10, label='Sequential (ours)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['sequential_fast'],
               markersize=10, label='Sequential (fast)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['joint'],
               markersize=10, label='Joint'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['monolithic'],
               markersize=10, label='Monolithic'),
    ]
    dim_handles = [
        Line2D([0], [0], marker='o', color='gray', markersize=6, linestyle='None', label='dim=8'),
        Line2D([0], [0], marker='s', color='gray', markersize=8, linestyle='None', label='dim=16'),
        Line2D([0], [0], marker='D', color='gray', markersize=10, linestyle='None', label='dim=32'),
    ]

    first_legend = ax.legend(handles=method_handles, loc='upper right', title='Method')
    ax.add_artist(first_legend)
    ax.legend(handles=dim_handles, loc='center right', title='Dimension')

    ax.set_xlabel('Trainable Parameters During Adaptation')
    ax.set_ylabel('Test MSE After Adaptation (↓ better)')
    ax.set_title('Parameter Efficiency: MSE vs Trainable Parameters\n(lower-left = better)',
                fontweight='bold')
    ax.set_yscale('log')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3)

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "parameter_efficiency.png")
    fig.savefig(path, bbox_inches='tight', dpi=150)
    print(f"Saved {path}")
    return path


def fig4_sample_efficiency(results, save_dir="figures"):
    """Line plot: MSE vs number of training samples."""
    se = results.get("sample_efficiency")
    if not se:
        print("No sample efficiency data, skipping fig4")
        return None

    fig, ax = plt.subplots(figsize=(8, 5))

    sample_sizes = se["sample_sizes"]
    for method in ["sequential", "joint", "monolithic"]:
        mses = se[method]
        color = COLORS[method]
        ax.semilogy(sample_sizes, mses, '-o', color=color, label=LABELS[method],
                   markersize=6, linewidth=2)

    ax.set_xlabel('Training Samples')
    ax.set_ylabel('Test MSE (log scale)')
    ax.set_title('Sample Efficiency (dim=8)', fontweight='bold')
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3)

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "sample_efficiency.png")
    fig.savefig(path, bbox_inches='tight', dpi=150)
    print(f"Saved {path}")
    return path


def fig5_speedup_summary(results, save_dir="figures"):
    """Bar chart showing adaptation speedup and param reduction."""
    adapt_data = results["adaptation"]

    dims = [r["dim"] for r in adapt_data]

    # Compute ratios relative to joint
    param_ratios = []
    mse_ratios = []
    time_ratios = []

    for r in adapt_data:
        adapt = r["adaptation"]
        seq = adapt["sequential_full"]
        jnt = adapt["joint"]

        param_ratios.append(jnt["trainable_params"] / seq["trainable_params"])
        mse_ratios.append(jnt["final_mse"] / max(seq["final_mse"], 1e-10))
        time_ratios.append(jnt["time_s"] / max(seq["time_s"], 0.01))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # Param reduction
    axes[0].bar(range(len(dims)), param_ratios, color='#2196F3', edgecolor='white')
    axes[0].set_xticks(range(len(dims)))
    axes[0].set_xticklabels([f'dim={d}' for d in dims])
    axes[0].set_ylabel('Ratio (Joint / Sequential)')
    axes[0].set_title('Parameter Reduction\n(higher = Sequential uses fewer params)', fontweight='bold')
    axes[0].axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    # MSE ratio
    colors_mse = ['#4CAF50' if r > 1 else '#F44336' for r in mse_ratios]
    axes[1].bar(range(len(dims)), mse_ratios, color=colors_mse, edgecolor='white')
    axes[1].set_xticks(range(len(dims)))
    axes[1].set_xticklabels([f'dim={d}' for d in dims])
    axes[1].set_ylabel('Ratio (Joint MSE / Sequential MSE)')
    axes[1].set_title('MSE Comparison\n(>1 = Sequential wins)', fontweight='bold')
    axes[1].axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    # Time ratio
    axes[2].bar(range(len(dims)), time_ratios, color='#FF9800', edgecolor='white')
    axes[2].set_xticks(range(len(dims)))
    axes[2].set_xticklabels([f'dim={d}' for d in dims])
    axes[2].set_ylabel('Ratio (Joint Time / Sequential Time)')
    axes[2].set_title('Training Time Reduction\n(higher = Sequential is faster)', fontweight='bold')
    axes[2].axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    axes[2].spines['top'].set_visible(False)
    axes[2].spines['right'].set_visible(False)

    fig.suptitle('Sequential Two-Phase Advantage Over Joint Training During Adaptation',
                fontweight='bold', y=1.05)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "adaptation_speedup.png")
    fig.savefig(path, bbox_inches='tight', dpi=150)
    print(f"Saved {path}")
    return path


if __name__ == "__main__":
    results = load_results()
    paths = []
    paths.append(fig1_adaptation_mse_comparison(results))
    paths.append(fig2_convergence_curves(results))
    paths.append(fig3_parameter_efficiency(results))
    paths.append(fig4_sample_efficiency(results))
    paths.append(fig5_speedup_summary(results))
    print(f"\nGenerated {len([p for p in paths if p])} figures")
