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
