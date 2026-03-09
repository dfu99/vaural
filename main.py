import os

import torch

from config import Config
from components import Emitter, ActionToSignal, Environment, Receiver, Pipeline
from train import pretrain_receiver, train_emitter
from visualize import (
    plot_loss_curve,
    plot_token_comparison,
    plot_environment_matrix,
    plot_pipeline_trace,
)


def main():
    cfg = Config()

    # Seed
    torch.manual_seed(cfg.seed)

    # Output directory
    os.makedirs(cfg.output_dir, exist_ok=True)

    # ── Phase 0: Instantiate fixed components ──
    print("Instantiating fixed components...")
    action_to_signal = ActionToSignal(cfg.action_dim, cfg.signal_dim, seed=100)
    environment = Environment(cfg.signal_dim, seed=200)

    # ── Phase 1: Pre-train Receiver ──
    print("\n=== Phase 1: Pre-training Receiver ===")
    receiver = Receiver(cfg.signal_dim, cfg.sound_dim, cfg.hidden_dim)
    receiver_losses = pretrain_receiver(action_to_signal, environment, receiver, cfg)

    # Freeze receiver
    receiver.requires_grad_(False)
    receiver.eval()

    plot_loss_curve(
        receiver_losses,
        "Receiver Pre-training Loss",
        os.path.join(cfg.output_dir, "receiver_loss.png"),
    )

    # ── Phase 2: Train Emitter ──
    print("\n=== Phase 2: Training Emitter ===")
    emitter = Emitter(cfg.sound_dim, cfg.action_dim, cfg.hidden_dim)
    pipeline = Pipeline(emitter, action_to_signal, environment, receiver)
    emitter_losses = train_emitter(pipeline, cfg)

    plot_loss_curve(
        emitter_losses,
        "Emitter Training Loss",
        os.path.join(cfg.output_dir, "emitter_loss.png"),
    )

    # ── Phase 3: Evaluate ──
    print("\n=== Phase 3: Evaluation ===")
    pipeline.eval()
    test_sounds = torch.randn(100, cfg.sound_dim)

    with torch.no_grad():
        decoded = pipeline(test_sounds)
        test_mse = torch.nn.functional.mse_loss(decoded, test_sounds).item()

    print(f"\n  Final test MSE on 100 held-out tokens: {test_mse:.6f}")

    # Sanity check: gradient flow
    print("\n  Sanity check:")
    emitter_has_grad = any(
        p.grad is not None for p in pipeline.emitter.parameters()
    )
    receiver_has_grad = any(
        p.requires_grad for p in pipeline.receiver.parameters()
    )
    print(f"    Emitter params have gradients: {emitter_has_grad}")
    print(f"    Receiver params require grad:  {receiver_has_grad}")

    # Generate visualizations
    print("\n  Generating visualizations...")
    plot_token_comparison(
        test_sounds,
        decoded,
        os.path.join(cfg.output_dir, "token_comparison.png"),
    )
    plot_environment_matrix(
        environment.weight,
        os.path.join(cfg.output_dir, "environment_matrix.png"),
    )

    with torch.no_grad():
        trace = pipeline.forward_trace(test_sounds[:1])
    plot_pipeline_trace(trace, os.path.join(cfg.output_dir, "pipeline_trace.png"))

    print(f"\n  All outputs saved to {cfg.output_dir}/")
    print(f"  Final test MSE: {test_mse:.6f}")


if __name__ == "__main__":
    main()
