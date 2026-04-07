"""Generate combined scale ablation figure from checkpoint files."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

COLORS = {'sequential': '#2196F3', 'joint': '#FF9800', 'matched': '#9C27B0'}

dims = [8, 16, 32]
results = {}
for d in dims:
    with open(f"results/scale_ablation_dim{d}.json") as f:
        results[d] = json.load(f)

# Convergence curves
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for i, d in enumerate(dims):
    ax = axes[i]
    data = results[d]
    for method, color, label in [
        ("sequential", COLORS["sequential"], "Sequential (pre-trained)"),
        ("joint", COLORS["joint"], "Joint (both trainable)"),
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
    h = data["hidden"]
    ep = data.get("emitter_params", "?")
    ax.set_title(f'dim={d}, h={h}\n({ep} emitter params)', fontweight='bold')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.suptitle('Two-Phase Training Across Dimensions (3 seeds, mean±std)',
            fontweight='bold', y=1.02)
plt.tight_layout()
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/pub_scale_ablation.png", bbox_inches='tight', dpi=150)
print("Saved figures/pub_scale_ablation.png")

# Summary bar chart
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(dims))
width = 0.25
for j, (method, color, label) in enumerate([
    ("sequential", COLORS["sequential"], "Sequential"),
    ("joint", COLORS["joint"], "Joint"),
    ("matched", COLORS["matched"], "Matched (random)"),
]):
    means = [results[d][method]["final_mse_mean"] for d in dims]
    stds = [results[d][method]["final_mse_std"] for d in dims]
    offset = (j - 1) * width
    bars = ax.bar(x + offset, means, width, yerr=stds, label=label,
                 color=color, edgecolor='white', capsize=3)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                f'{m:.4f}', ha='center', fontsize=7, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels([f'dim={d}' for d in dims])
ax.set_ylabel('Final Test MSE')
ax.set_yscale('log')
ax.set_title('Scale Ablation: Sequential vs Joint vs Matched-Params', fontweight='bold')
ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, alpha=0.2, axis='y')
plt.tight_layout()
fig.savefig("figures/pub_scale_summary.png", bbox_inches='tight', dpi=150)
print("Saved figures/pub_scale_summary.png")
