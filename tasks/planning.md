# Planning — vaural

## Current Priorities

1. **Complete parameter sweep** — sweep.py is ready but too slow on CPU. Run it on GPU or in smaller batches. Key experiments to finish:
   - Dimension scaling: dim=64 with h=256 (dim=8/16/32 fast results below)
   - Hidden dim scaling at dim=32: h=[64, 128, 256]
   - LR sweep at dim=16: lr=[3e-4, 1e-3, 3e-3, 1e-2]
   - Signal dim scaling (sound=action=16, signal varies): sig=[8, 16, 32]
2. **Scale to larger dimensions** — once sweep identifies best hidden/LR, run dim=64 and dim=128 with full training epochs
3. **Consider GPU support** — add device handling to Config/training if GPU available; CPU training is very slow at dim≥32

## Partial Sweep Results (300 recv + 500 emit epochs, 2k samples — fast/underfitted)

| Config | Test MSE | Uniform MSE | Notes |
|--------|----------|-------------|-------|
| dim=8, h=64 | 0.000168 | 0.000043 | Excellent even with few epochs |
| dim=16, h=64 | 0.161204 | 0.049080 | Needs more training |
| dim=32, h=128 | 0.079817 | 0.025893 | Needs more training |

## Baseline (full training: 2k recv + 3k emit epochs, 10k samples)

| Config | Test MSE |
|--------|----------|
| dim=16, h=64 (defaults) | 0.000254 |

## Architecture Constraint Discovered

- **sound_dim must equal action_dim** — the identity-mapping receiver pre-training assumes sound=action. Asymmetric action dims cause shape mismatches. Added assertion in `pretrain_receiver()`. Supporting asymmetric dims would require a fundamentally different pre-training approach.

## Next Steps

- Run full training (2k+3k epochs) for dim=32 with h=128 and h=256 to compare
- If GPU available, run dim=64 and dim=128 experiments
- Add early stopping to training loops to avoid wasting compute
- Consider LR scheduling (cosine annealing or step decay)

## Recently Completed

- Initial project scaffolding verified: all 26 tests pass
- Git repo initialized with .gitignore
- Optimized training loops: replaced DataLoader with direct index slicing, pre-compute fixed transforms in receiver pre-training
- Added parameter sweep script (sweep.py) with 4 experiment types
- Discovered and documented sound_dim==action_dim constraint
- Added assertion + test for the constraint
- Ran baseline training: dim=16 test MSE = 0.000254
- Ran fast sweep for dim=8/16/32: dim=8 converges easily, larger dims need more training
