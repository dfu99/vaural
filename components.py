import torch
import torch.nn as nn


class Emitter(nn.Module):
    """Trainable 3-layer MLP. Maps sound tokens to actions."""

    def __init__(self, sound_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(sound_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VectorQuantizer(nn.Module):
    """Vector Quantization bottleneck with EMA codebook updates.

    Discretizes continuous vectors by snapping to the nearest codebook entry.
    Uses Exponential Moving Average for codebook updates (no codebook gradients)
    and straight-through gradient estimation for the encoder.
    """

    def __init__(self, num_codes: int, code_dim: int, commitment_cost: float = 0.25,
                 ema_decay: float = 0.99):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.commitment_cost = commitment_cost
        self.ema_decay = ema_decay

        self.register_buffer("codebook", torch.randn(num_codes, code_dim))
        self.register_buffer("ema_count", torch.zeros(num_codes))
        self.register_buffer("ema_sum", torch.randn(num_codes, code_dim))

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize input vectors.

        Args:
            z: (batch, code_dim) continuous vectors

        Returns:
            z_q: (batch, code_dim) quantized vectors (straight-through)
            commitment_loss: scalar loss to keep encoder close to codes
            indices: (batch,) codebook indices used
        """
        # Compute distances to codebook entries
        dists = (
            z.pow(2).sum(dim=-1, keepdim=True)
            + self.codebook.pow(2).sum(dim=-1)
            - 2 * z @ self.codebook.T
        )

        indices = dists.argmin(dim=-1)
        z_q = self.codebook[indices]

        # EMA codebook update (only during training)
        if self.training:
            one_hot = torch.zeros(z.size(0), self.num_codes, device=z.device)
            one_hot.scatter_(1, indices.unsqueeze(1), 1)

            self.ema_count.mul_(self.ema_decay).add_(one_hot.sum(0), alpha=1 - self.ema_decay)
            self.ema_sum.mul_(self.ema_decay).add_(
                one_hot.T @ z.detach(), alpha=1 - self.ema_decay
            )

            # Laplace smoothing to avoid division by zero
            n = self.ema_count.sum()
            count_smoothed = (
                (self.ema_count + 1e-5) / (n + self.num_codes * 1e-5) * n
            )
            self.codebook.copy_(self.ema_sum / count_smoothed.unsqueeze(1))

        # Commitment loss only (encoder stays close to codes)
        commitment_loss = self.commitment_cost * (z - z_q.detach()).pow(2).mean()

        # Straight-through estimator: copy gradients from z_q to z
        z_q = z + (z_q - z).detach()

        return z_q, commitment_loss, indices


class ActionToSignal(nn.Module):
    """Fixed random linear transform. Maps action to signal."""

    def __init__(self, action_dim: int, signal_dim: int, seed: int = 100):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        weight = torch.randn(signal_dim, action_dim, generator=gen)
        self.register_buffer("weight", weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T


class Environment(nn.Module):
    """Fixed random matrix transform. Maps signal to received signal."""

    def __init__(self, signal_dim: int, seed: int = 200):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        weight = torch.randn(signal_dim, signal_dim, generator=gen)
        self.register_buffer("weight", weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T


class Receiver(nn.Module):
    """Pre-trained then frozen 3-layer MLP. Maps received signal to sound."""

    def __init__(self, signal_dim: int, sound_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(signal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, sound_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Pipeline(nn.Module):
    """Full pipeline: Emitter -> ActionToSignal -> Environment -> Receiver."""

    def __init__(
        self,
        emitter: Emitter,
        action_to_signal: ActionToSignal,
        environment: Environment,
        receiver: Receiver,
    ):
        super().__init__()
        self.emitter = emitter
        self.action_to_signal = action_to_signal
        self.environment = environment
        self.receiver = receiver

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        action = self.emitter(x)
        signal = self.action_to_signal(action)
        received = self.environment(signal)
        decoded = self.receiver(received)
        return decoded

    def forward_trace(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return intermediate representations at each stage."""
        action = self.emitter(x)
        signal = self.action_to_signal(action)
        received = self.environment(signal)
        decoded = self.receiver(received)
        return {
            "input": x,
            "action": action,
            "signal": signal,
            "received": received,
            "decoded": decoded,
        }
