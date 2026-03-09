from dataclasses import dataclass


@dataclass
class Config:
    # Dimensions
    sound_dim: int = 16
    action_dim: int = 16
    signal_dim: int = 16
    hidden_dim: int = 64

    # Receiver pre-training
    receiver_lr: float = 1e-3
    receiver_epochs: int = 2000
    receiver_batch_size: int = 64
    receiver_samples: int = 10000

    # Emitter training
    emitter_lr: float = 1e-3
    emitter_epochs: int = 3000
    emitter_batch_size: int = 64
    emitter_samples: int = 10000

    # General
    seed: int = 42
    plot_every: int = 100
    output_dir: str = "outputs"
