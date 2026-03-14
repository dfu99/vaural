"""Experiment: Pure Rotational Invariance Test.

obj-013 showed channel condition number κ dominates reconstruction quality.
But that experiment conflated two factors:
  1. Singular value spectrum (σ₁, ..., σ_d) — how spread the channel gains are
  2. Rotation matrices (U, V) — which directions get which gains

This experiment isolates factor (2) by holding the spectrum FIXED and varying
ONLY the rotation matrices. If the system is rotationally invariant, performance
should depend only on the spectrum, not on the specific rotations.

Design:
  - Fix 3 spectra: flat (κ=1), moderate (κ=10), steep (κ=100)
  - For each spectrum, sample N_ROTATIONS different random (U, V) pairs
  - Train full pipeline (sequential) for each configuration
  - Measure: MSE variance across rotations at fixed spectrum

If MSE variance is near-zero for each spectrum → system is rotationally invariant
If MSE varies significantly → rotation matters, and we investigate why
"""

import os
import sys

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from components import Emitter, ActionToSignal, Environment, Receiver, Pipeline
from train import pretrain_receiver, train_emitter


def make_orthogonal(dim, seed):
    """Random orthogonal matrix via QR."""
    gen = torch.Generator().manual_seed(seed)
    M = torch.randn(dim, dim, generator=gen)
    Q, R = torch.linalg.qr(M)
    Q = Q * torch.sign(torch.diag(R)).unsqueeze(0)
    return Q


def make_channel_from_svd(dim, sigmas, seed_u, seed_v):
    """Build channel M = U @ diag(sigmas) @ V^T with specific rotations.

    Returns ActionToSignal and Environment components such that
    env.weight @ a2s.weight = M.

    We set a2s.weight = diag(sigmas) @ V^T and env.weight = U.
    Then M = U @ diag(sigmas) @ V^T as desired.
    """
    U = make_orthogonal(dim, seed_u)
    V = make_orthogonal(dim, seed_v)

    # a2s maps action → signal: weight is (signal_dim, action_dim)
    # env maps signal → received: weight is (signal_dim, signal_dim)
    # Combined: env.weight @ a2s.weight = U @ diag(sigmas) @ V^T
    a2s = ActionToSignal(dim, dim, seed=0)  # placeholder
    env = Environment(dim, seed=0)

    a2s.weight.copy_(torch.diag(sigmas) @ V.T)
    env.weight.copy_(U)

    M = env.weight @ a2s.weight
    return a2s, env, M


def train_and_evaluate(a2s, env, cfg, n_test=2000, seed=42):
    """Train pipeline and return test MSE."""
    torch.manual_seed(seed)
    receiver = Receiver(cfg.signal_dim, cfg.sound_dim, cfg.hidden_dim)
    recv_losses = pretrain_receiver(a2s, env, receiver, cfg)
    receiver.requires_grad_(False)

    emitter = Emitter(cfg.sound_dim, cfg.action_dim, cfg.hidden_dim)
    pipeline = Pipeline(emitter, a2s, env, receiver)
    emit_losses = train_emitter(pipeline, cfg)

    # Evaluate
    torch.manual_seed(99)
    sounds = torch.randn(n_test, cfg.sound_dim)
    pipeline.eval()
    with torch.no_grad():
        reconstructed = pipeline(sounds)
        mse = (reconstructed - sounds).pow(2).mean().item()

    return mse, recv_losses, emit_losses


def run_experiment(dim=8, n_rotations=8, epochs_recv=400, epochs_emit=500,
                   n_samples=2000):
    """Run the pure rotational invariance test."""
    cfg = Config(
        sound_dim=dim, action_dim=dim, signal_dim=dim,
        hidden_dim=64,
        receiver_lr=1e-3, receiver_epochs=epochs_recv,
        receiver_samples=n_samples, receiver_batch_size=64,
        emitter_lr=1e-3, emitter_epochs=epochs_emit,
        emitter_samples=n_samples, emitter_batch_size=64,
        plot_every=200,
    )

    # Define spectra
    spectra = {
        "flat (κ=1)": torch.ones(dim),
        "moderate (κ=10)": torch.logspace(0, -1, dim),
        "steep (κ=100)": torch.logspace(0, -2, dim),
    }

    results = {}
    for spec_name, sigmas in spectra.items():
        kappa = (sigmas[0] / sigmas[-1]).item()
        print(f"\n{'='*60}")
        print(f"Spectrum: {spec_name}  (σ range: {sigmas[0]:.3f} → {sigmas[-1]:.3f}, κ={kappa:.1f})")
        print(f"{'='*60}")

        mses = []
        recv_final_losses = []
        emit_final_losses = []

        for rot_idx in range(n_rotations):
            seed_u = 1000 + rot_idx * 7
            seed_v = 2000 + rot_idx * 13

            print(f"\n  Rotation {rot_idx+1}/{n_rotations} (seeds: U={seed_u}, V={seed_v})")

            a2s, env, M = make_channel_from_svd(dim, sigmas, seed_u, seed_v)

            # Verify SVD is correct
            _, S_check, _ = torch.linalg.svd(M)
            assert torch.allclose(S_check, sigmas, atol=1e-4), \
                f"SVD mismatch: {S_check} vs {sigmas}"

            mse, recv_losses, emit_losses = train_and_evaluate(
                a2s, env, cfg, seed=42
            )
            mses.append(mse)
            recv_final_losses.append(recv_losses[-1])
            emit_final_losses.append(emit_losses[-1])

            print(f"    MSE: {mse:.6f}")

        mses = np.array(mses)
        results[spec_name] = {
            "sigmas": sigmas.numpy(),
            "kappa": kappa,
            "mses": mses,
            "mean_mse": mses.mean(),
            "std_mse": mses.std(),
            "cv_mse": mses.std() / mses.mean() if mses.mean() > 0 else 0,
            "min_mse": mses.min(),
            "max_mse": mses.max(),
            "recv_final_losses": np.array(recv_final_losses),
            "emit_final_losses": np.array(emit_final_losses),
        }

        print(f"\n  Summary: MSE = {mses.mean():.6f} ± {mses.std():.6f} "
              f"(CV={results[spec_name]['cv_mse']:.2%})")
        print(f"  Range: [{mses.min():.6f}, {mses.max():.6f}] "
              f"(max/min = {mses.max()/max(mses.min(), 1e-10):.2f}×)")

    return results, dim


def plot_results(results, dim, output_path):
    """Visualize the rotational invariance test results."""
    spec_names = list(results.keys())
    n_specs = len(spec_names)
    colors = ["#4CAF50", "#2196F3", "#E91E63"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # --- Panel 1: MSE distribution per spectrum (the key result) ---
    ax = axes[0, 0]
    positions = range(n_specs)
    for i, (name, r) in enumerate(results.items()):
        mses = r["mses"]
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(mses))
        ax.scatter(np.full_like(mses, i) + jitter, mses, color=colors[i],
                   s=80, alpha=0.7, edgecolors="white", linewidth=0.5, zorder=3)
        ax.errorbar(i, r["mean_mse"], yerr=r["std_mse"], fmt="D",
                    color="black", markersize=8, capsize=8, capthick=2, zorder=4)
    ax.set_xticks(range(n_specs))
    ax.set_xticklabels([f"{name}\nκ={r['kappa']:.0f}" for name, r in results.items()],
                       fontsize=9)
    ax.set_ylabel("Test MSE")
    ax.set_yscale("log")
    ax.set_title("MSE Distribution Across Random Rotations\n(fixed spectrum, varying U,V)")
    ax.grid(True, alpha=0.3, axis="y")

    # --- Panel 2: Coefficient of variation (relative spread) ---
    ax = axes[0, 1]
    cvs = [r["cv_mse"] for r in results.values()]
    max_min_ratios = [r["max_mse"] / max(r["min_mse"], 1e-10) for r in results.values()]
    bars = ax.bar(range(n_specs), cvs, color=colors, alpha=0.8)
    ax.set_xticks(range(n_specs))
    ax.set_xticklabels([f"κ={r['kappa']:.0f}" for r in results.values()])
    ax.set_ylabel("Coefficient of Variation (σ/μ)")
    ax.set_title("MSE Variability Due to Rotation Alone")
    for bar, cv, ratio in zip(bars, cvs, max_min_ratios):
        ax.text(bar.get_x() + bar.get_width() / 2, cv + max(cvs) * 0.03,
                f"CV={cv:.1%}\nmax/min={ratio:.1f}×",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # --- Panel 3: MSE vs κ with error bands ---
    ax = axes[1, 0]
    kappas = [r["kappa"] for r in results.values()]
    means = [r["mean_mse"] for r in results.values()]
    stds = [r["std_mse"] for r in results.values()]
    mins = [r["min_mse"] for r in results.values()]
    maxs = [r["max_mse"] for r in results.values()]

    ax.fill_between(kappas, mins, maxs, alpha=0.2, color="#666")
    ax.errorbar(kappas, means, yerr=stds, fmt="o-", color="black",
                markersize=8, capsize=6, linewidth=2, label="Mean ± std")
    ax.scatter(kappas, mins, marker="v", color="#4CAF50", s=60, zorder=3, label="Min")
    ax.scatter(kappas, maxs, marker="^", color="#E91E63", s=60, zorder=3, label="Max")

    ax.set_xlabel("Condition number κ")
    ax.set_ylabel("Test MSE")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("MSE vs Condition Number\n(shaded = range across rotations)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panel 4: Summary table ---
    ax = axes[1, 1]
    ax.axis("off")
    table_data = []
    for name, r in results.items():
        table_data.append([
            f"κ={r['kappa']:.0f}",
            f"{r['mean_mse']:.6f}",
            f"{r['std_mse']:.6f}",
            f"{r['cv_mse']:.1%}",
            f"{r['max_mse']/max(r['min_mse'], 1e-10):.2f}×",
        ])
    table = ax.table(
        cellText=table_data,
        colLabels=["Spectrum", "Mean MSE", "Std MSE", "CV", "Max/Min"],
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Color the spectrum column
    for i in range(len(table_data)):
        table[i + 1, 0].set_facecolor(colors[i])
        table[i + 1, 0].set_text_props(color="white", fontweight="bold")

    verdict = "ROTATIONALLY INVARIANT" if all(
        r["cv_mse"] < 0.1 for r in results.values()
    ) else "NOT ROTATIONALLY INVARIANT"
    ax.set_title(f"Summary — Verdict: {verdict}", fontsize=12, fontweight="bold",
                 pad=20)

    fig.suptitle(
        f"Pure Rotational Invariance Test (dim={dim})\n"
        f"Fixed singular value spectrum, varying rotation matrices U, V",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved: {output_path}")


def main():
    os.makedirs("results", exist_ok=True)

    print("=" * 60)
    print("EXPERIMENT: Pure Rotational Invariance Test")
    print("Fixed spectrum, varying rotations — does U,V matter?")
    print("=" * 60)

    results, dim = run_experiment(
        dim=8, n_rotations=8, epochs_recv=400, epochs_emit=500, n_samples=2000
    )

    plot_results(results, dim, "results/obj-015-pure-rotational-invariance.png")

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL VERDICT")
    print("=" * 60)
    for name, r in results.items():
        print(f"  {name}: MSE = {r['mean_mse']:.6f} ± {r['std_mse']:.6f} "
              f"(CV = {r['cv_mse']:.2%}, max/min = {r['max_mse']/max(r['min_mse'],1e-10):.2f}×)")

    all_cvs = [r["cv_mse"] for r in results.values()]
    if all(cv < 0.1 for cv in all_cvs):
        print("\n  → System IS rotationally invariant (CV < 10% for all spectra)")
        print("    Performance depends only on singular value spectrum, not rotation.")
    else:
        high_cv = [(name, r["cv_mse"]) for name, r in results.items() if r["cv_mse"] >= 0.1]
        print(f"\n  → System is NOT rotationally invariant")
        for name, cv in high_cv:
            print(f"    {name}: CV = {cv:.1%}")
        print("    Specific rotation matrices U, V affect reconstruction quality.")


if __name__ == "__main__":
    main()
