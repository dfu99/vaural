import torch
import torch.nn as nn

from config import Config
from components import ActionToSignal, Environment, Receiver, Emitter, Pipeline


def _make_batches(data: torch.Tensor, batch_size: int) -> list[torch.Tensor]:
    """Shuffle and split data into batches (pre-computed for speed)."""
    perm = torch.randperm(data.size(0))
    shuffled = data[perm]
    return list(shuffled.split(batch_size))


def pretrain_receiver(
    action_to_signal: ActionToSignal,
    environment: Environment,
    receiver: Receiver,
    cfg: Config,
) -> list[float]:
    """Pre-train the receiver using identity mapping (sound = action).

    The receiver learns to invert Environment(ActionToSignal(x)).
    """
    sounds = torch.randn(cfg.receiver_samples, cfg.sound_dim)

    # Pre-compute transformed inputs (fixed transforms, no grad needed)
    with torch.no_grad():
        signals = action_to_signal(sounds)
        received = environment(signals)

    optimizer = torch.optim.Adam(receiver.parameters(), lr=cfg.receiver_lr)
    loss_fn = nn.MSELoss()

    losses = []
    for epoch in range(cfg.receiver_epochs):
        perm = torch.randperm(sounds.size(0))
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, sounds.size(0), cfg.receiver_batch_size):
            idx = perm[i:i + cfg.receiver_batch_size]
            batch_recv = received[idx]
            batch_target = sounds[idx]

            decoded = receiver(batch_recv)
            loss = loss_fn(decoded, batch_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        if (epoch + 1) % cfg.plot_every == 0:
            print(f"  Receiver epoch {epoch + 1}/{cfg.receiver_epochs}, loss: {avg_loss:.6f}")

    return losses


def train_emitter(pipeline: Pipeline, cfg: Config) -> list[float]:
    """Train the emitter while receiver, action_to_signal, and environment are frozen.

    Gradients flow through the frozen components back into the emitter.
    """
    # Freeze everything except emitter
    pipeline.receiver.requires_grad_(False)
    pipeline.action_to_signal.requires_grad_(False)
    pipeline.environment.requires_grad_(False)

    sounds = torch.randn(cfg.emitter_samples, cfg.sound_dim)

    optimizer = torch.optim.Adam(pipeline.emitter.parameters(), lr=cfg.emitter_lr)
    loss_fn = nn.MSELoss()

    losses = []
    for epoch in range(cfg.emitter_epochs):
        perm = torch.randperm(sounds.size(0))
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, sounds.size(0), cfg.emitter_batch_size):
            idx = perm[i:i + cfg.emitter_batch_size]
            batch = sounds[idx]

            decoded = pipeline(batch)
            loss = loss_fn(decoded, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        if (epoch + 1) % cfg.plot_every == 0:
            print(f"  Emitter epoch {epoch + 1}/{cfg.emitter_epochs}, loss: {avg_loss:.6f}")

    return losses
