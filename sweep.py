"""Parameter sweep and scaling experiments for vaural pipeline.

Uses reduced epoch counts for fast iteration. Key findings can then
be validated with full training via main.py config changes.
"""

import json
import os
import time

import torch
import torch.nn as nn

from config import Config
from components import Emitter, ActionToSignal, Environment, Receiver, Pipeline
from train import pretrain_receiver, train_emitter


def run_experiment(cfg: Config, label: str = "") -> dict:
    """Run a full training pipeline and return metrics."""
    t0 = time.time()
    torch.manual_seed(cfg.seed)

    action_to_signal = ActionToSignal(cfg.action_dim, cfg.signal_dim, seed=100)
    environment = Environment(cfg.signal_dim, seed=200)
    receiver = Receiver(cfg.signal_dim, cfg.sound_dim, cfg.hidden_dim)

    receiver_losses = pretrain_receiver(action_to_signal, environment, receiver, cfg)
    receiver.requires_grad_(False)
    receiver.eval()

    emitter = Emitter(cfg.sound_dim, cfg.action_dim, cfg.hidden_dim)
    pipeline = Pipeline(emitter, action_to_signal, environment, receiver)
    emitter_losses = train_emitter(pipeline, cfg)

    pipeline.eval()
    with torch.no_grad():
        test_normal = torch.randn(200, cfg.sound_dim)
        decoded_normal = pipeline(test_normal)
        mse_normal = nn.functional.mse_loss(decoded_normal, test_normal).item()

        test_uniform = torch.rand(200, cfg.sound_dim) * 2 - 1
        decoded_uniform = pipeline(test_uniform)
        mse_uniform = nn.functional.mse_loss(decoded_uniform, test_uniform).item()

    elapsed = time.time() - t0
    emitter_params = sum(p.numel() for p in pipeline.emitter.parameters())
    receiver_params = sum(p.numel() for p in pipeline.receiver.parameters())

    result = {
        "label": label,
        "receiver_final_loss": receiver_losses[-1],
        "emitter_final_loss": emitter_losses[-1],
        "test_mse_normal": mse_normal,
        "test_mse_uniform": mse_uniform,
        "emitter_params": emitter_params,
        "receiver_params": receiver_params,
        "elapsed_s": round(elapsed, 1),
    }
    print(f"  [{label}] MSE={mse_normal:.6f} uniform={mse_uniform:.6f} "
          f"recv={receiver_losses[-1]:.6f} emit={emitter_losses[-1]:.6f} "
          f"({elapsed:.0f}s)")
    return result


# Fast training defaults
FAST_RECV_EPOCHS = 500
FAST_EMIT_EPOCHS = 800
FAST_SAMPLES = 3000


def main():
    os.makedirs("outputs/sweep", exist_ok=True)
    all_results = {}

    # ── Experiment 1: Scale dimensions ──
    print("=" * 60)
    print("Experiment 1: Dimension scaling (sound=action=signal)")
    print("=" * 60)
    results = []
    for dim in [8, 16, 32, 64]:
        hidden = max(64, dim * 4)
        cfg = Config(
            sound_dim=dim, action_dim=dim, signal_dim=dim,
            hidden_dim=hidden,
            receiver_epochs=FAST_RECV_EPOCHS, emitter_epochs=FAST_EMIT_EPOCHS,
            receiver_samples=FAST_SAMPLES, emitter_samples=FAST_SAMPLES,
            plot_every=9999,
        )
        r = run_experiment(cfg, f"dim={dim},h={hidden}")
        r["dim"] = dim
        r["hidden_dim"] = hidden
        results.append(r)
    all_results["dim_scaling"] = results

    # ── Experiment 2: Hidden dim at dim=32 ──
    print("\n" + "=" * 60)
    print("Experiment 2: Hidden dim scaling (dim=32)")
    print("=" * 60)
    results = []
    for hidden in [64, 128, 256]:
        cfg = Config(
            sound_dim=32, action_dim=32, signal_dim=32,
            hidden_dim=hidden,
            receiver_epochs=FAST_RECV_EPOCHS, emitter_epochs=FAST_EMIT_EPOCHS,
            receiver_samples=FAST_SAMPLES, emitter_samples=FAST_SAMPLES,
            plot_every=9999,
        )
        r = run_experiment(cfg, f"hidden={hidden}")
        r["hidden_dim"] = hidden
        results.append(r)
    all_results["hidden_scaling"] = results

    # ── Experiment 3: Learning rate sweep ──
    print("\n" + "=" * 60)
    print("Experiment 3: LR sweep (dim=32, hidden=128)")
    print("=" * 60)
    results = []
    for lr in [3e-4, 1e-3, 3e-3]:
        cfg = Config(
            sound_dim=32, action_dim=32, signal_dim=32,
            hidden_dim=128, receiver_lr=lr, emitter_lr=lr,
            receiver_epochs=FAST_RECV_EPOCHS, emitter_epochs=FAST_EMIT_EPOCHS,
            receiver_samples=FAST_SAMPLES, emitter_samples=FAST_SAMPLES,
            plot_every=9999,
        )
        r = run_experiment(cfg, f"lr={lr}")
        r["lr"] = lr
        results.append(r)
    all_results["lr_sweep"] = results

    # ── Experiment 4: Asymmetric dimensions ──
    print("\n" + "=" * 60)
    print("Experiment 4: Asymmetric dims")
    print("=" * 60)
    results = []
    for name, sound, action, signal in [
        ("bottleneck", 32, 16, 32),
        ("matched",    32, 32, 32),
        ("expansion",  32, 64, 32),
    ]:
        cfg = Config(
            sound_dim=sound, action_dim=action, signal_dim=signal,
            hidden_dim=128,
            receiver_epochs=FAST_RECV_EPOCHS, emitter_epochs=FAST_EMIT_EPOCHS,
            receiver_samples=FAST_SAMPLES, emitter_samples=FAST_SAMPLES,
            plot_every=9999,
        )
        r = run_experiment(cfg, name)
        r.update({"sound_dim": sound, "action_dim": action, "signal_dim": signal})
        results.append(r)
    all_results["asymmetric"] = results

    # Save
    path = "outputs/sweep/results.json"
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2)

    # ── Summary table ──
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for section, rows in all_results.items():
        print(f"\n{section}:")
        print(f"  {'Label':<25} {'TestMSE':>10} {'Uniform':>10} {'Time':>6}")
        for r in rows:
            print(f"  {r['label']:<25} {r['test_mse_normal']:>10.6f} "
                  f"{r['test_mse_uniform']:>10.6f} {r['elapsed_s']:>5.0f}s")

    print(f"\nResults saved to {path}")


if __name__ == "__main__":
    main()
