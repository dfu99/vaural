# Vaural

A PyTorch simulation of sensorimotor vocal communication. An **Emitter** learns to encode sound tokens into motor actions that pass through a fixed physical channel and are decoded by a pre-trained **Receiver** back into sound tokens — modeling how biological organisms learn to vocalize by listening to their own output.

## Motivation

When infants learn to speak, they rely on auditory feedback to refine their vocalizations — a tightly coupled sensorimotor loop. Vaural is a computational sandbox for studying this coupling: can a production system (Emitter) learn to communicate through an unknown channel by relying solely on a perception system (Receiver) that has already learned to listen?

This connects to a broader research question: **should audio generation and audio processing share a joint embedding space**, rather than existing as entirely separate models? The Emitter's gradient signal flows backward through the frozen Receiver, meaning the Emitter implicitly learns to produce outputs that align with the Receiver's internal representations — a form of emergent sensorimotor coupling.

## Architecture

```
Sound Tokens ──> Emitter (trainable) ──> ActionToSignal ──> Environment ──> Receiver (frozen) ──> Decoded Sound
                     ^                        (fixed)         (fixed)            |
                     └──────────── gradient flow (backpropagation) ──────────────┘
                                                                            MSE Loss
```

![Architecture Diagram](outputs/architecture_diagram.png)

Training happens in two phases:

1. **Phase 1 — Receiver pre-training**: The Receiver (3-layer MLP) learns to invert the combined ActionToSignal + Environment transform. After convergence, it is frozen.
2. **Phase 2 — Emitter training**: The Emitter (3-layer MLP) learns to map sound tokens to actions such that the full pipeline reconstructs the original sound. Gradients flow through the frozen channel and Receiver back into the Emitter.

The fixed components (ActionToSignal, Environment) are seeded random linear transforms — deterministic but unknown to the learners, simulating an opaque physical channel.

## Current Progress

### Baseline Results

| Configuration | Test MSE | Status |
|---------------|----------|--------|
| dim=8, h=64 (300+500 epochs) | 0.000168 | Converged |
| dim=16, h=64 (2k+3k epochs) | 0.000254 | Converged (baseline) |
| dim=32, h=128 (300+500 epochs) | 0.079817 | Needs more training |

### Visualizations

| Plot | Description |
|------|-------------|
| ![Receiver Loss](outputs/receiver_loss.png) | Phase 1: Receiver pre-training loss converges to near-zero |
| ![Emitter Loss](outputs/emitter_loss.png) | Phase 2: Emitter training loss drops steadily |
| ![Token Comparison](outputs/token_comparison.png) | Original vs decoded sound tokens — near-perfect reconstruction |
| ![Pipeline Trace](outputs/pipeline_trace.png) | Signal at each stage for a single sample |
| ![Environment Matrix](outputs/environment_matrix.png) | Heatmap of the fixed environment transform |

### Key Findings

- The pipeline achieves near-perfect reconstruction (MSE < 0.001) at dim=16 with sufficient training
- `sound_dim` must equal `action_dim` due to the identity-mapping receiver pre-training strategy
- dim=8 converges easily even with few epochs; dim >= 16 requires 1000+ epochs
- Pre-computing fixed transforms and using direct index slicing significantly speeds up CPU training

## Quickstart

```bash
# Install dependencies
pip install torch matplotlib numpy pytest

# Run full training pipeline
python main.py

# Run tests
pytest test_vaural.py -v
```

Outputs (loss curves, comparison plots, traces) are saved to `outputs/`.

## Project Structure

| File | Role |
|------|------|
| `config.py` | `Config` dataclass with all hyperparameters |
| `components.py` | All `nn.Module` classes: Emitter, ActionToSignal, Environment, Receiver, Pipeline |
| `train.py` | `pretrain_receiver()` and `train_emitter()` training loops |
| `visualize.py` | Matplotlib plotting utilities |
| `main.py` | Orchestrates the full pipeline |
| `sweep.py` | Parameter sweep experiments |
| `test_vaural.py` | Pytest suite (26 tests) |

## Research Context

Vaural sits at the intersection of several research threads:

- **Neuroscience**: The DIVA model (Guenther, BU) describes how auditory feedback drives motor learning during babbling — structurally identical to Vaural's gradient-based Emitter training
- **Machine Speech Chain** (Hori et al., 2017): Joint ASR/TTS training in a closed loop, the closest prior art to Vaural's approach
- **Emergent Communication**: Lewis signaling games study how agents develop communication protocols; Vaural uses continuous representations through physical transforms rather than discrete symbols
- **Self-supervised Speech**: HuBERT and wav2vec 2.0 discover phoneme-like units without labels — suggesting that discrete structure can emerge from continuous sensorimotor loops

See [`tasks/research.md`](tasks/research.md) for the full literature review and hypothesis analysis.

## Milestones

### Completed

- [x] Core pipeline: Emitter, Channel, Receiver with two-phase training
- [x] Visualization suite: loss curves, token comparison, environment heatmap, pipeline trace
- [x] Test suite: 26 tests covering shapes, determinism, convergence, end-to-end reconstruction
- [x] Parameter sweep framework
- [x] Baseline results at multiple dimensions

### In Progress

- [ ] Scale to larger dimensions (dim=64, 128) with GPU support
- [ ] Full parameter sweep (hidden dim, learning rate, signal dim)
- [ ] Early stopping and LR scheduling

### Future Directions

- [ ] **Joint training**: Unfreeze Receiver during Emitter training to test true sensorimotor coupling vs. sequential training
- [ ] **VQ bottleneck**: Add vector quantization to the Emitter output to produce emergent discrete "phoneme-like" units
- [ ] **Asymmetric dimensions**: Alternative pre-training strategy to decouple sound_dim from action_dim (information bottleneck experiments)
- [ ] **Internal feedback signal**: Add a DIVA-inspired auditory error map loss that compares Emitter output to Receiver expectations in latent space
- [ ] **Multi-agent communication**: Multiple Emitter-Receiver pairs developing shared protocols through a common channel
- [ ] **Noise and distortion**: Add realistic channel noise to test robustness of learned encodings
- [ ] **Temporal sequences**: Extend from single token reconstruction to sequence-level communication

## Dependencies

- Python 3.10+
- PyTorch
- matplotlib
- numpy
- pytest
