# Vaural

Sensorimotor vocal communication in PyTorch. An Emitter learns to produce actions that reconstruct sound tokens after passing through an unknown physical channel, guided only by a pre-trained Receiver's gradient signal.

```
Sound ──▶ Emitter ──▶ ActionToSignal ──▶ Environment ──▶ Receiver (frozen) ──▶ Decoded Sound
             ▲            (fixed)           (fixed)            │
             └──────────────── gradient flow ──────────────────┘
```

## Usage

```bash
pip install torch matplotlib numpy pytest

python main.py            # train and evaluate
pytest test_vaural.py -v  # 27 tests
```

Training runs in two phases: (1) pre-train the Receiver to invert the channel, freeze it; (2) train the Emitter end-to-end through the frozen pipeline. Outputs save to `outputs/`.

## Result

The system achieves near-perfect reconstruction (MSE < 0.001) across dimensions 8–16. SiLU activation is critical — it eliminates the rotational sensitivity that ReLU introduces through axis-aligned decision boundaries.

| dim | MSE |
|-----|-----|
| 8   | 0.000033 |
| 16  | 0.000156 |

## Example

```python
from config import Config
from components import Emitter, ActionToSignal, Environment, Receiver, Pipeline
from train import pretrain_receiver, train_emitter

cfg = Config(sound_dim=8, action_dim=8, signal_dim=8)

a2s = ActionToSignal(cfg.action_dim, cfg.signal_dim)
env = Environment(cfg.signal_dim)
receiver = Receiver(cfg.signal_dim, cfg.sound_dim, cfg.hidden_dim)

pretrain_receiver(a2s, env, receiver, cfg)
receiver.requires_grad_(False)

emitter = Emitter(cfg.sound_dim, cfg.action_dim, cfg.hidden_dim)
pipeline = Pipeline(emitter, a2s, env, receiver)
train_emitter(pipeline, cfg)
```

## Dependencies

Python 3.10+, PyTorch, matplotlib, numpy, pytest
