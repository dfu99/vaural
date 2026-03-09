import os

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_loss_curve(
    losses: list[float], title: str, filepath: str, plot_every: int = 1
) -> None:
    """Plot loss over epochs with log-scale y-axis."""
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(losses) + 1)
    ax.plot(epochs, losses, linewidth=1.5)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (log scale)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"  Saved: {filepath}")


def plot_token_comparison(
    original: torch.Tensor,
    decoded: torch.Tensor,
    filepath: str,
    n_tokens: int = 5,
    n_dims: int = 8,
) -> None:
    """Grouped bar chart comparing original vs decoded tokens."""
    original = original[:n_tokens, :n_dims].detach().cpu().numpy()
    decoded = decoded[:n_tokens, :n_dims].detach().cpu().numpy()

    fig, axes = plt.subplots(1, n_tokens, figsize=(3 * n_tokens, 4), sharey=True)
    x = np.arange(n_dims)
    width = 0.35

    for i, ax in enumerate(axes):
        ax.bar(x - width / 2, original[i], width, label="Original", alpha=0.8)
        ax.bar(x + width / 2, decoded[i], width, label="Decoded", alpha=0.8)
        ax.set_title(f"Token {i}")
        ax.set_xlabel("Dimension")
        if i == 0:
            ax.set_ylabel("Value")
            ax.legend(fontsize=8)

    fig.suptitle("Original vs Decoded Sound Tokens", fontsize=14)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"  Saved: {filepath}")


def plot_environment_matrix(weight: torch.Tensor, filepath: str) -> None:
    """Heatmap of the environment matrix with diverging colormap."""
    matrix = weight.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(6, 5))
    vmax = np.abs(matrix).max()
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    fig.colorbar(im, ax=ax)
    ax.set_title("Environment Matrix")
    ax.set_xlabel("Input dimension")
    ax.set_ylabel("Output dimension")
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"  Saved: {filepath}")


def plot_pipeline_trace(trace: dict[str, torch.Tensor], filepath: str) -> None:
    """Show intermediate representations at each pipeline stage for a single sample."""
    stages = ["input", "action", "signal", "received", "decoded"]
    fig, axes = plt.subplots(1, len(stages), figsize=(3 * len(stages), 4), sharey=True)

    for ax, stage in zip(axes, stages):
        values = trace[stage][0].detach().cpu().numpy()
        ax.bar(range(len(values)), values, alpha=0.8)
        ax.set_title(stage.capitalize())
        ax.set_xlabel("Dimension")
        if stage == "input":
            ax.set_ylabel("Value")

    fig.suptitle("Pipeline Trace (single sample)", fontsize=14)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"  Saved: {filepath}")
