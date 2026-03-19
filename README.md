# Vaural

A PyTorch simulation of sensorimotor vocal communication. An **Emitter** learns to encode sound tokens into motor actions that pass through a fixed physical channel and are decoded by a pre-trained **Receiver** back into sound tokens — modeling how biological organisms learn to vocalize by listening to their own output.

## Motivation

When infants learn to speak, they rely on auditory feedback to refine their vocalizations — a tightly coupled sensorimotor loop. Vaural is a computational sandbox for studying this coupling: can a production system (Emitter) learn to communicate through an unknown channel by relying solely on a perception system (Receiver) that has already learned to listen?

This connects to a broader research question across three sibling projects (WorldNN, vaural, CorticalNN): **how does an agent learn to act effectively through a lossy, unknown channel?** The Emitter's gradient signal flows backward through the frozen Receiver, meaning the Emitter implicitly learns to produce outputs that align with the Receiver's internal representations — a form of emergent sensorimotor coupling.

## Architecture

```
Sound Tokens ──> Emitter (SiLU MLP) ──> ActionToSignal ──> Environment ──> Receiver (frozen SiLU MLP) ──> Decoded Sound
                     ^                       (fixed)          (fixed)              |
                     └──────────── gradient flow (backpropagation) ────────────────┘
                                                                               MSE Loss
```

Training happens in two phases (sequential mode):

1. **Phase 1 — Receiver pre-training**: The Receiver (3-layer MLP with SiLU activations) learns to invert the combined ActionToSignal + Environment transform. After convergence, it is frozen.
2. **Phase 2 — Emitter training**: The Emitter (3-layer MLP with SiLU) learns to map sound tokens to actions such that the full pipeline reconstructs the original sound. Gradients flow through the frozen channel and Receiver back into the Emitter.

The fixed components (ActionToSignal, Environment) are seeded random linear transforms — deterministic but unknown to the learners, simulating an opaque physical channel.

## Key Results

### Baseline Reconstruction

| Configuration | Test MSE | Notes |
|---------------|----------|-------|
| dim=8, h=64, SiLU | 0.000033 | Near-perfect reconstruction |
| dim=16, h=64, SiLU | 0.000156 | Validated |
| dim=16, h=64, ReLU (legacy) | 0.000365 | SiLU is 2.3× better |

### Rotational Invariance (obj-013 → obj-024, 12 experiments)

The channel transform M = U·diag(σ)·V^T has both a spectrum (σ) and rotation (U, V). A comprehensive investigation found:

| Finding | Experiment |
|---------|-----------|
| System is NOT rotationally invariant with ReLU (CV 13-23%) | obj-015 |
| **SiLU eliminates rotation sensitivity** (CV ~2%, residual is training noise) | obj-017, obj-020 |
| ReLU creates 18 kinks per output trajectory; SiLU creates 0 | obj-023 |
| Emitter adapts to channel rotation in ~50 epochs (2.7× oracle) | obj-021 |
| Joint fine-tuning from warm start closes 93% of accent gap (1.10× oracle) | obj-022 |
| SiLU advantage validated at dim=16 (CV 1.9% vs 5.1% ReLU) | obj-024 |

**Mechanism**: ReLU's element-wise max(0, x) creates decision boundaries along coordinate hyperplanes, producing orientation-dependent kinks. SiLU's smooth gating (x·σ(x)) has no axis bias, yielding smooth output trajectories under rotation.

### Joint Training vs Sequential

| Mode | dim=8 MSE | Rotation CV | Notes |
|------|-----------|-------------|-------|
| Sequential (default) | 0.000033 | 8.8% | Stable, best rotation invariance |
| Joint (from scratch) | 0.008300 | 43.7% | Underfitted at matched epochs |
| Joint (warm-started) | 0.000229 | — | 1.10× oracle for channel adaptation |

Joint training from scratch hurts (obj-018), but joint fine-tuning from a warm-started Receiver nearly matches oracle quality (obj-022). The key variable is initialization quality.

### VQ Bottleneck

Tested codebook sizes [4, 8, 16, 32, 64, 128, 256] at dim=8 (obj-012). All codebooks achieve near-max entropy utilization, but even 256 codes (MSE=0.44) is far from continuous (MSE=0.002). Random continuous inputs don't benefit from discrete codes — structured/clustered inputs needed for meaningful "phoneme emergence."

### Cross-Project: Coordination Quality C_i

From the WorldNN sensory-motor alignment framework: C_i = cos(emitter(s), M⁻¹·s) measures what fraction of the Emitter's output aligns with the optimal action. Currently being validated across channel conditions (PACE job 5055141).

## Quickstart

```bash
# Install dependencies
pip install torch matplotlib numpy pytest

# Run full training pipeline
python main.py

# Run tests (27 tests)
pytest test_vaural.py -v
```

Outputs (loss curves, comparison plots, traces) are saved to `outputs/`.

## Project Structure

| File | Role |
|------|------|
| `config.py` | `Config` dataclass with all hyperparameters |
| `components.py` | All `nn.Module` classes: Emitter, ActionToSignal, Environment, Receiver, Pipeline (SiLU default) |
| `train.py` | `pretrain_receiver()` and `train_emitter()` training loops |
| `visualize.py` | Matplotlib plotting utilities |
| `main.py` | Orchestrates the full pipeline |
| `test_vaural.py` | Pytest suite (27 tests) |
| `experiments/` | Individual experiment scripts (rotation invariance, VQ, adaptation, C_i) |
| `results/` | Generated figures for each objective |

## Research Context

Vaural sits at the intersection of several research threads:

- **Neuroscience**: The DIVA model (Guenther, BU) describes how auditory feedback drives motor learning during babbling — structurally identical to Vaural's gradient-based Emitter training
- **Machine Speech Chain** (Hori et al., 2017): Joint ASR/TTS training in a closed loop, the closest prior art
- **Emergent Communication**: Lewis signaling games with continuous representations through physical transforms rather than discrete symbols
- **Self-supervised Speech**: HuBERT and wav2vec 2.0 discover phoneme-like units without labels
- **Sensory-Motor Alignment**: Cross-project framework (WorldNN) formalizing perception as projection learning — the Emitter learns an alignment operator R through an unknown channel

See [`tasks/research.md`](tasks/research.md) for the full literature review.

## Completed Investigations

- [x] Core pipeline with two-phase sequential training
- [x] Visualization suite and test suite (27 tests)
- [x] Parameter sweep framework
- [x] Joint training ablation (obj-011): 2.7× improvement at dim=8
- [x] VQ bottleneck capacity curve (obj-012)
- [x] **Rotational invariance arc** (obj-013 → obj-024): 12 experiments establishing SiLU as the solution, characterizing adaptation dynamics, and identifying the ReLU kink mechanism
- [x] SiLU default activation switch

## Current & Future Directions

- [ ] **Coordination quality C_i** (obj-025): Cross-project metric validation (submitted to PACE)
- [ ] **Structured VQ input**: Clustered/categorical inputs for emergent phoneme discovery
- [ ] **Internal feedback loss**: DIVA-inspired auditory error map
- [ ] **Multi-agent communication**: Multiple Emitter-Receiver pairs through a common channel
- [ ] **Temporal sequences**: Extend from single token to sequence-level communication
- [ ] **GPU scaling**: Full training at dim=32+ with proper compute budget

## Dependencies

- Python 3.10+
- PyTorch
- matplotlib
- numpy
- pytest
