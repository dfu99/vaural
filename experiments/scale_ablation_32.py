"""Quick dim=32 ablation with reduced epochs to complete on CPU."""

import torch
import torch.nn as nn
import numpy as np
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from components import Emitter, ActionToSignal, Environment, Receiver

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class SimplePipeline(nn.Module):
    def __init__(self, emitter, a2s, env, receiver):
        super().__init__()
        self.emitter = emitter
        self.action_to_signal = a2s
        self.environment = env
        self.receiver = receiver

    def forward(self, x):
        return self.receiver(self.environment(self.action_to_signal(self.emitter(x))))


def pretrain_receiver(dim, hidden, a2s, env, epochs, n_samples, seed=42):
    torch.manual_seed(seed)
    recv = Receiver(dim, dim, hidden)
    sounds = torch.randn(n_samples, dim)
    with torch.no_grad():
        received = env(a2s(sounds))
    opt = torch.optim.Adam(recv.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    for ep in range(epochs):
        perm = torch.randperm(n_samples)
        for i in range(0, n_samples, 128):
            idx = perm[i:i+128]
            loss = loss_fn(recv(received[idx]), sounds[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    return recv


def run_training(model, params, sounds, test_sounds, epochs, eval_every=25):
    opt = torch.optim.Adam(params, lr=1e-3)
    loss_fn = nn.MSELoss()
    curve = []
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(sounds.size(0))
        for i in range(0, sounds.size(0), 128):
            idx = perm[i:i+128]
            loss = loss_fn(model(sounds[idx]), sounds[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        if (epoch + 1) % eval_every == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                mse = loss_fn(model(test_sounds), test_sounds).item()
            curve.append((epoch + 1, mse))
    return curve


def serialize(obj):
    if isinstance(obj, dict):
        return {k: serialize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize(v) for v in obj]
    elif isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    return obj


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    dim, hidden, epochs, n_samples = 32, 256, 1500, 5000
    seeds = range(42, 45)

    print(f"DIM={dim}, hidden={hidden}, epochs={epochs}")
    methods = {"sequential": [], "joint": [], "matched": []}

    for seed in seeds:
        print(f"  seed={seed}...", end=" ", flush=True)
        t0 = time.time()
        torch.manual_seed(seed)
        test_sounds = torch.randn(2000, dim)
        sounds = torch.randn(n_samples, dim)

        # Sequential
        a2s = ActionToSignal(dim, dim, seed=300)
        env = Environment(dim, seed=400)
        recv = pretrain_receiver(dim, hidden, a2s, env, epochs, n_samples, seed)
        torch.manual_seed(seed + 1000)
        emit = Emitter(dim, dim, hidden)
        pipe = SimplePipeline(emit, a2s, env, recv)
        pipe.receiver.requires_grad_(False)
        pipe.action_to_signal.requires_grad_(False)
        pipe.environment.requires_grad_(False)
        c_seq = run_training(pipe, list(emit.parameters()), sounds, test_sounds, epochs)
        methods["sequential"].append(c_seq)

        # Joint
        a2s2 = ActionToSignal(dim, dim, seed=300)
        env2 = Environment(dim, seed=400)
        recv2 = Receiver(dim, dim, hidden)
        torch.manual_seed(seed + 1000)
        emit2 = Emitter(dim, dim, hidden)
        pipe2 = SimplePipeline(emit2, a2s2, env2, recv2)
        params = list(emit2.parameters()) + list(recv2.parameters())
        c_jnt = run_training(pipe2, params, sounds, test_sounds, epochs)
        methods["joint"].append(c_jnt)

        # Matched
        a2s3 = ActionToSignal(dim, dim, seed=300)
        env3 = Environment(dim, seed=400)
        recv3 = Receiver(dim, dim, hidden)
        torch.manual_seed(seed + 1000)
        emit3 = Emitter(dim, dim, hidden)
        pipe3 = SimplePipeline(emit3, a2s3, env3, recv3)
        pipe3.receiver.requires_grad_(False)
        pipe3.action_to_signal.requires_grad_(False)
        pipe3.environment.requires_grad_(False)
        c_mat = run_training(pipe3, list(emit3.parameters()), sounds, test_sounds, epochs)
        methods["matched"].append(c_mat)

        elapsed = time.time() - t0
        print(f"seq={c_seq[-1][1]:.6f}, jnt={c_jnt[-1][1]:.6f}, "
              f"matched={c_mat[-1][1]:.6f} ({elapsed/60:.1f}min)")

    # Aggregate and save
    emit_params = sum(p.numel() for p in emit.parameters() if p.requires_grad)
    result = {"dim": dim, "hidden": hidden, "epochs": epochs, "emitter_params": emit_params}
    for method, curves in methods.items():
        mse_matrix = np.array([[m for _, m in c] for c in curves])
        result[method] = {
            "mean_curve": mse_matrix.mean(axis=0).tolist(),
            "std_curve": mse_matrix.std(axis=0).tolist(),
            "epochs": [ep for ep, _ in curves[0]],
            "final_mse_mean": float(mse_matrix[:, -1].mean()),
            "final_mse_std": float(mse_matrix[:, -1].std()),
        }
        print(f"  {method}: MSE={result[method]['final_mse_mean']:.6f}"
              f"±{result[method]['final_mse_std']:.6f}")

    with open("results/scale_ablation_dim32.json", "w") as f:
        json.dump(serialize(result), f, indent=2)
    print("Saved results/scale_ablation_dim32.json")
