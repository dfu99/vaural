"""
Adaptation Benchmark: The Slam Dunk Experiment

Compares three training strategies on their ability to:
1. Achieve low MSE in initial training
2. Adapt to a NEW channel (different random seed) efficiently
3. Scale across dimensions

Strategies:
- Sequential (vaural's method): Pre-train Receiver on channel, freeze, train Emitter
- Joint: Train Emitter + Receiver simultaneously from scratch
- Monolithic: Single large MLP mapping sound→sound through the channel (no decomposition)

The key metric: when the channel changes, how fast and cheaply can each method recover?
"""

import torch
import torch.nn as nn
import numpy as np
import json
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components import Emitter, ActionToSignal, Environment, Receiver, Pipeline
from config import Config


class MonolithicModel(nn.Module):
    """Single MLP that maps sound→sound through the channel, no decomposition."""
    def __init__(self, sound_dim, signal_dim, hidden_dim, a2s_seed=100, env_seed=200):
        super().__init__()
        # Same channel
        a2s = ActionToSignal(sound_dim, signal_dim, seed=a2s_seed)
        env = Environment(signal_dim, seed=env_seed)
        self.register_buffer("channel_weight", env.weight @ a2s.weight)  # combined channel

        # Single large MLP: receives channel output, produces sound reconstruction
        total_hidden = hidden_dim * 2  # give it more capacity to be fair
        self.net = nn.Sequential(
            nn.Linear(signal_dim, total_hidden),
            nn.SiLU(),
            nn.Linear(total_hidden, total_hidden),
            nn.SiLU(),
            nn.Linear(total_hidden, sound_dim),
        )

    def forward(self, x):
        received = x @ self.channel_weight.T
        return self.net(received)


def count_params(model, trainable_only=True):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def train_sequential(cfg, a2s_seed=100, env_seed=200, max_epochs=None,
                     pretrained_receiver=None, verbose=False):
    """Train using vaural's two-phase approach. Returns (pipeline, metrics)."""
    torch.manual_seed(cfg.seed)

    a2s = ActionToSignal(cfg.action_dim, cfg.signal_dim, seed=a2s_seed)
    env = Environment(cfg.signal_dim, seed=env_seed)
    receiver = Receiver(cfg.signal_dim, cfg.sound_dim, cfg.hidden_dim)
    emitter = Emitter(cfg.sound_dim, cfg.action_dim, cfg.hidden_dim)

    recv_epochs = max_epochs or cfg.receiver_epochs
    emit_epochs = max_epochs or cfg.emitter_epochs

    metrics = {"method": "sequential", "recv_losses": [], "emit_losses": [],
               "eval_mse": [], "eval_epochs": []}

    # Phase 1: Pre-train receiver (or reuse)
    if pretrained_receiver is not None:
        receiver.load_state_dict(pretrained_receiver.state_dict())
        metrics["recv_epochs_used"] = 0
        metrics["recv_params"] = 0
    else:
        sounds = torch.randn(cfg.receiver_samples, cfg.sound_dim)
        with torch.no_grad():
            signals = a2s(sounds)
            received = env(signals)

        opt = torch.optim.Adam(receiver.parameters(), lr=cfg.receiver_lr)
        loss_fn = nn.MSELoss()
        for epoch in range(recv_epochs):
            perm = torch.randperm(sounds.size(0))
            epoch_loss = 0.0
            n = 0
            for i in range(0, sounds.size(0), cfg.receiver_batch_size):
                idx = perm[i:i + cfg.receiver_batch_size]
                decoded = receiver(received[idx])
                loss = loss_fn(decoded, sounds[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
                n += 1
            metrics["recv_losses"].append(epoch_loss / n)

        metrics["recv_epochs_used"] = recv_epochs
        metrics["recv_params"] = count_params(receiver)

    # Phase 2: Train emitter (frozen receiver)
    pipeline = Pipeline(emitter, a2s, env, receiver)
    pipeline.receiver.requires_grad_(False)
    pipeline.action_to_signal.requires_grad_(False)
    pipeline.environment.requires_grad_(False)

    sounds = torch.randn(cfg.emitter_samples, cfg.sound_dim)
    test_sounds = torch.randn(2000, cfg.sound_dim)

    opt = torch.optim.Adam(pipeline.emitter.parameters(), lr=cfg.emitter_lr)
    loss_fn = nn.MSELoss()

    metrics["emit_trainable_params"] = count_params(emitter)

    for epoch in range(emit_epochs):
        perm = torch.randperm(sounds.size(0))
        epoch_loss = 0.0
        n = 0
        for i in range(0, sounds.size(0), cfg.emitter_batch_size):
            idx = perm[i:i + cfg.emitter_batch_size]
            decoded = pipeline(sounds[idx])
            loss = loss_fn(decoded, sounds[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n += 1
        metrics["emit_losses"].append(epoch_loss / n)

        # Evaluate every 50 epochs
        if (epoch + 1) % 50 == 0 or epoch == 0:
            with torch.no_grad():
                test_out = pipeline(test_sounds)
                test_mse = nn.MSELoss()(test_out, test_sounds).item()
            metrics["eval_mse"].append(test_mse)
            metrics["eval_epochs"].append(epoch + 1)
            if verbose:
                print(f"  Sequential emit epoch {epoch+1}: test MSE = {test_mse:.6f}")

    return pipeline, metrics


def train_joint(cfg, a2s_seed=100, env_seed=200, max_epochs=None, verbose=False):
    """Train Emitter + Receiver jointly from scratch."""
    torch.manual_seed(cfg.seed)

    a2s = ActionToSignal(cfg.action_dim, cfg.signal_dim, seed=a2s_seed)
    env = Environment(cfg.signal_dim, seed=env_seed)
    receiver = Receiver(cfg.signal_dim, cfg.sound_dim, cfg.hidden_dim)
    emitter = Emitter(cfg.sound_dim, cfg.action_dim, cfg.hidden_dim)
    pipeline = Pipeline(emitter, a2s, env, receiver)

    epochs = max_epochs or (cfg.receiver_epochs + cfg.emitter_epochs)

    sounds = torch.randn(cfg.emitter_samples, cfg.sound_dim)
    test_sounds = torch.randn(2000, cfg.sound_dim)

    # Train both emitter and receiver
    trainable = list(emitter.parameters()) + list(receiver.parameters())
    opt = torch.optim.Adam(trainable, lr=cfg.emitter_lr)
    loss_fn = nn.MSELoss()

    metrics = {"method": "joint", "losses": [], "eval_mse": [], "eval_epochs": [],
               "trainable_params": count_params(emitter) + count_params(receiver)}

    for epoch in range(epochs):
        perm = torch.randperm(sounds.size(0))
        epoch_loss = 0.0
        n = 0
        for i in range(0, sounds.size(0), cfg.emitter_batch_size):
            idx = perm[i:i + cfg.emitter_batch_size]
            decoded = pipeline(sounds[idx])
            loss = loss_fn(decoded, sounds[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n += 1
        metrics["losses"].append(epoch_loss / n)

        if (epoch + 1) % 50 == 0 or epoch == 0:
            with torch.no_grad():
                test_out = pipeline(test_sounds)
                test_mse = nn.MSELoss()(test_out, test_sounds).item()
            metrics["eval_mse"].append(test_mse)
            metrics["eval_epochs"].append(epoch + 1)
            if verbose:
                print(f"  Joint epoch {epoch+1}: test MSE = {test_mse:.6f}")

    return pipeline, metrics


def train_monolithic(cfg, a2s_seed=100, env_seed=200, max_epochs=None, verbose=False):
    """Train a single large MLP end-to-end."""
    torch.manual_seed(cfg.seed)

    model = MonolithicModel(cfg.sound_dim, cfg.signal_dim, cfg.hidden_dim,
                            a2s_seed=a2s_seed, env_seed=env_seed)

    epochs = max_epochs or (cfg.receiver_epochs + cfg.emitter_epochs)

    sounds = torch.randn(cfg.emitter_samples, cfg.sound_dim)
    test_sounds = torch.randn(2000, cfg.sound_dim)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.emitter_lr)
    loss_fn = nn.MSELoss()

    metrics = {"method": "monolithic", "losses": [], "eval_mse": [], "eval_epochs": [],
               "trainable_params": count_params(model)}

    for epoch in range(epochs):
        perm = torch.randperm(sounds.size(0))
        epoch_loss = 0.0
        n = 0
        for i in range(0, sounds.size(0), cfg.emitter_batch_size):
            idx = perm[i:i + cfg.emitter_batch_size]
            decoded = model(sounds[idx])
            loss = loss_fn(decoded, sounds[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n += 1
        metrics["losses"].append(epoch_loss / n)

        if (epoch + 1) % 50 == 0 or epoch == 0:
            with torch.no_grad():
                test_out = model(test_sounds)
                test_mse = nn.MSELoss()(test_out, test_sounds).item()
            metrics["eval_mse"].append(test_mse)
            metrics["eval_epochs"].append(epoch + 1)
            if verbose:
                print(f"  Monolithic epoch {epoch+1}: test MSE = {test_mse:.6f}")

    return model, metrics


def run_adaptation_experiment(dim, hidden, epochs_initial, epochs_adapt, n_samples=5000, verbose=True):
    """
    Full experiment for a given dimension:
    1. Train all methods on Channel A
    2. Switch to Channel B (different env seed)
    3. Measure adaptation cost for each method
    """
    cfg = Config(
        sound_dim=dim, action_dim=dim, signal_dim=dim, hidden_dim=hidden,
        receiver_epochs=epochs_initial, emitter_epochs=epochs_initial,
        receiver_samples=n_samples, emitter_samples=n_samples,
        receiver_batch_size=64, emitter_batch_size=64,
        seed=42
    )

    channel_a = {"a2s_seed": 100, "env_seed": 200}
    channel_b = {"a2s_seed": 300, "env_seed": 400}  # different channel

    results = {"dim": dim, "hidden": hidden}

    if verbose:
        print(f"\n{'='*60}")
        print(f"DIM={dim}, HIDDEN={hidden}")
        print(f"{'='*60}")

    # === Phase 1: Initial training on Channel A ===
    if verbose:
        print(f"\n--- Initial Training (Channel A, {epochs_initial} epochs) ---")

    t0 = time.time()
    if verbose:
        print("Sequential...")
    seq_pipe, seq_metrics = train_sequential(cfg, **channel_a, max_epochs=epochs_initial, verbose=verbose)
    seq_time = time.time() - t0

    t0 = time.time()
    if verbose:
        print("Joint...")
    joint_pipe, joint_metrics = train_joint(cfg, **channel_a, max_epochs=epochs_initial, verbose=verbose)
    joint_time = time.time() - t0

    t0 = time.time()
    if verbose:
        print("Monolithic...")
    mono_model, mono_metrics = train_monolithic(cfg, **channel_a, max_epochs=epochs_initial, verbose=verbose)
    mono_time = time.time() - t0

    results["initial"] = {
        "sequential": {
            "final_mse": seq_metrics["eval_mse"][-1],
            "trainable_params_phase2": seq_metrics["emit_trainable_params"],
            "total_params": count_params(seq_pipe, trainable_only=False),
            "time_s": seq_time,
            "eval_mse": seq_metrics["eval_mse"],
            "eval_epochs": seq_metrics["eval_epochs"],
        },
        "joint": {
            "final_mse": joint_metrics["eval_mse"][-1],
            "trainable_params": joint_metrics["trainable_params"],
            "time_s": joint_time,
            "eval_mse": joint_metrics["eval_mse"],
            "eval_epochs": joint_metrics["eval_epochs"],
        },
        "monolithic": {
            "final_mse": mono_metrics["eval_mse"][-1],
            "trainable_params": mono_metrics["trainable_params"],
            "time_s": mono_time,
            "eval_mse": mono_metrics["eval_mse"],
            "eval_epochs": mono_metrics["eval_epochs"],
        },
    }

    # === Phase 2: Adaptation to Channel B ===
    if verbose:
        print(f"\n--- Adaptation to Channel B ({epochs_adapt} epochs) ---")

    # Sequential adaptation: reuse frozen receiver pre-trained on new channel,
    # retrain only the emitter
    adapt_cfg = Config(
        sound_dim=dim, action_dim=dim, signal_dim=dim, hidden_dim=hidden,
        receiver_epochs=epochs_adapt, emitter_epochs=epochs_adapt,
        receiver_samples=n_samples, emitter_samples=n_samples,
        seed=42
    )

    if verbose:
        print("Sequential adapt (retrain receiver on new channel, then emitter)...")
    t0 = time.time()
    seq_adapt_pipe, seq_adapt_metrics = train_sequential(
        adapt_cfg, **channel_b, max_epochs=epochs_adapt, verbose=verbose
    )
    seq_adapt_time = time.time() - t0

    # Sequential fast adapt: reuse existing receiver architecture, only retrain emitter
    # (receiver trained fresh on new channel with fewer epochs, then emitter)
    if verbose:
        print("Sequential fast adapt (fresh receiver + emitter, budget epochs)...")
    fast_cfg = Config(
        sound_dim=dim, action_dim=dim, signal_dim=dim, hidden_dim=hidden,
        receiver_epochs=epochs_adapt // 2, emitter_epochs=epochs_adapt // 2,
        receiver_samples=n_samples, emitter_samples=n_samples,
        seed=42
    )
    t0 = time.time()
    seq_fast_pipe, seq_fast_metrics = train_sequential(
        fast_cfg, **channel_b, max_epochs=epochs_adapt // 2, verbose=verbose
    )
    seq_fast_time = time.time() - t0

    # Joint adaptation from scratch on new channel
    if verbose:
        print("Joint adapt (from scratch on new channel)...")
    t0 = time.time()
    joint_adapt_pipe, joint_adapt_metrics = train_joint(
        adapt_cfg, **channel_b, max_epochs=epochs_adapt, verbose=verbose
    )
    joint_adapt_time = time.time() - t0

    # Monolithic retrain from scratch
    if verbose:
        print("Monolithic adapt (retrain from scratch)...")
    t0 = time.time()
    mono_adapt_model, mono_adapt_metrics = train_monolithic(
        adapt_cfg, **channel_b, max_epochs=epochs_adapt, verbose=verbose
    )
    mono_adapt_time = time.time() - t0

    results["adaptation"] = {
        "sequential_full": {
            "final_mse": seq_adapt_metrics["eval_mse"][-1],
            "epochs_used": epochs_adapt * 2,  # recv + emit
            "trainable_params": seq_adapt_metrics["emit_trainable_params"],
            "time_s": seq_adapt_time,
            "eval_mse": seq_adapt_metrics["eval_mse"],
            "eval_epochs": seq_adapt_metrics["eval_epochs"],
        },
        "sequential_fast": {
            "final_mse": seq_fast_metrics["eval_mse"][-1],
            "epochs_used": epochs_adapt,  # split between recv + emit
            "trainable_params": seq_fast_metrics["emit_trainable_params"],
            "time_s": seq_fast_time,
            "eval_mse": seq_fast_metrics["eval_mse"],
            "eval_epochs": seq_fast_metrics["eval_epochs"],
        },
        "joint": {
            "final_mse": joint_adapt_metrics["eval_mse"][-1],
            "epochs_used": epochs_adapt,
            "trainable_params": joint_adapt_metrics["trainable_params"],
            "time_s": joint_adapt_time,
            "eval_mse": joint_adapt_metrics["eval_mse"],
            "eval_epochs": joint_adapt_metrics["eval_epochs"],
        },
        "monolithic": {
            "final_mse": mono_adapt_metrics["eval_mse"][-1],
            "epochs_used": epochs_adapt,
            "trainable_params": mono_adapt_metrics["trainable_params"],
            "time_s": mono_adapt_time,
            "eval_mse": mono_adapt_metrics["eval_mse"],
            "eval_epochs": mono_adapt_metrics["eval_epochs"],
        },
    }

    return results


def run_sample_efficiency_experiment(dim=8, hidden=64, epochs=1000, verbose=True):
    """
    How many samples does each method need to reach a given MSE threshold?
    """
    sample_sizes = [100, 250, 500, 1000, 2500, 5000]
    results = {"dim": dim, "hidden": hidden, "sample_sizes": sample_sizes}

    if verbose:
        print(f"\n{'='*60}")
        print(f"SAMPLE EFFICIENCY: DIM={dim}")
        print(f"{'='*60}")

    for method_name, train_fn in [("sequential", train_sequential),
                                    ("joint", train_joint),
                                    ("monolithic", train_monolithic)]:
        mse_by_samples = []
        for n in sample_sizes:
            cfg = Config(
                sound_dim=dim, action_dim=dim, signal_dim=dim, hidden_dim=hidden,
                receiver_epochs=epochs, emitter_epochs=epochs,
                receiver_samples=n, emitter_samples=n,
                seed=42
            )
            if verbose:
                print(f"  {method_name}, n={n}...")
            _, metrics = train_fn(cfg, max_epochs=epochs)
            mse_by_samples.append(metrics["eval_mse"][-1])
        results[method_name] = mse_by_samples
        if verbose:
            print(f"  {method_name}: {[f'{m:.6f}' for m in mse_by_samples]}")

    return results


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    all_results = {}

    # Experiment 1: Adaptation benchmark across dimensions
    print("=" * 60)
    print("EXPERIMENT 1: ADAPTATION BENCHMARK")
    print("=" * 60)

    configs = [
        {"dim": 8, "hidden": 64, "epochs_initial": 1000, "epochs_adapt": 500},
        {"dim": 16, "hidden": 64, "epochs_initial": 1500, "epochs_adapt": 750},
        {"dim": 32, "hidden": 128, "epochs_initial": 2000, "epochs_adapt": 1000},
    ]

    adapt_results = []
    for c in configs:
        r = run_adaptation_experiment(**c, n_samples=5000, verbose=True)
        adapt_results.append(r)

    all_results["adaptation"] = adapt_results

    # Experiment 2: Sample efficiency at dim=8
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: SAMPLE EFFICIENCY")
    print("=" * 60)

    sample_results = run_sample_efficiency_experiment(dim=8, hidden=64, epochs=1000)
    all_results["sample_efficiency"] = sample_results

    # Save results
    # Convert for JSON serialization
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        return obj

    with open("results/adaptation_benchmark.json", "w") as f:
        json.dump(make_serializable(all_results), f, indent=2)

    print("\nResults saved to results/adaptation_benchmark.json")

    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for r in adapt_results:
        dim = r["dim"]
        print(f"\n--- dim={dim} ---")
        print(f"{'Method':<25} {'Initial MSE':>12} {'Adapt MSE':>12} {'Adapt Params':>12}")
        print("-" * 65)

        init = r["initial"]
        adapt = r["adaptation"]

        print(f"{'Sequential':<25} {init['sequential']['final_mse']:>12.6f} "
              f"{adapt['sequential_full']['final_mse']:>12.6f} "
              f"{adapt['sequential_full']['trainable_params']:>12d}")
        print(f"{'Sequential (fast)':<25} {'—':>12} "
              f"{adapt['sequential_fast']['final_mse']:>12.6f} "
              f"{adapt['sequential_fast']['trainable_params']:>12d}")
        print(f"{'Joint':<25} {init['joint']['final_mse']:>12.6f} "
              f"{adapt['joint']['final_mse']:>12.6f} "
              f"{adapt['joint']['trainable_params']:>12d}")
        print(f"{'Monolithic':<25} {init['monolithic']['final_mse']:>12.6f} "
              f"{adapt['monolithic']['final_mse']:>12.6f} "
              f"{adapt['monolithic']['trainable_params']:>12d}")
