# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Vaural is a PyTorch simulation of a vocal communication pipeline. An **Emitter** learns to encode sound tokens into actions that pass through fixed random transforms (ActionToSignal, Environment) and are decoded by a pre-trained **Receiver** back into sound tokens. The goal is end-to-end reconstruction with low MSE.

## Commands

```bash
# Run full training pipeline (outputs plots to outputs/)
python main.py

# Run all tests
pytest test_vaural.py -v

# Run a single test class or test
pytest test_vaural.py::TestShapes -v
pytest test_vaural.py::TestEndToEnd::test_round_trip_low_mse -v
```

## Architecture

The pipeline is: `Sound → Emitter → ActionToSignal → Environment → Receiver → Sound`

Training happens in two phases:
1. **Phase 1 — Receiver pre-training**: The Receiver (3-layer MLP) learns to invert the combined ActionToSignal+Environment transform using an identity mapping (sound=action). After training, the Receiver is frozen.
2. **Phase 2 — Emitter training**: The Emitter (3-layer MLP) learns to map sound tokens to actions such that the full pipeline reconstructs the original sound. Gradients flow through the frozen components back into the Emitter.

**Fixed components** (ActionToSignal, Environment) are seeded random linear transforms registered as buffers (no learnable parameters). They are deterministic given the same seed.

### Key files

| File | Role |
|------|------|
| `config.py` | `Config` dataclass with all hyperparameters |
| `components.py` | All `nn.Module` classes: Emitter, ActionToSignal, Environment, Receiver, Pipeline |
| `train.py` | `pretrain_receiver()` and `train_emitter()` training loops |
| `visualize.py` | Matplotlib plotting utilities (loss curves, token comparison, environment heatmap, pipeline trace) |
| `main.py` | Orchestrates the full training and evaluation pipeline |
| `test_vaural.py` | Pytest suite covering shapes, determinism, training convergence, and end-to-end reconstruction |

## Dependencies

- PyTorch, matplotlib, numpy, pytest

## Task Files

| File | Consult when |
|------|-------------|
| `tasks/planning.md` | Starting any work session; checking priorities |
| `tasks/lessons.md` | Before modifying training, components, or tests |
| `tasks/research.md` | Before designing new experiments or changing architecture direction |
