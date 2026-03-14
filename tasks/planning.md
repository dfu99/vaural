# Planning — vaural

## Current Priorities

1. **Structured input for VQ** — Random continuous inputs don't benefit from discrete codes. Next: use structured/clustered inputs where VQ can discover meaningful categories (true "phoneme emergence").
2. **Scale joint training to higher dims with more epochs** — dim=16 was underfitted at 600 epochs; run with 2k+ epochs to see if joint training advantage holds at scale
3. **Add GPU support** — add device handling to Config/training; CPU is too slow for dim≥16 experiments
4. **Complete parameter sweep** — sweep.py is ready but too slow on CPU. Key experiments:
   - Dimension scaling: dim=64 with h=256
   - Hidden dim scaling at dim=32: h=[64, 128, 256]
   - LR sweep at dim=16: lr=[3e-4, 1e-3, 3e-3, 1e-2]

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

### Research-informed experiments (see tasks/research.md)
- **Structured VQ input**: Use clustered/categorical inputs where VQ codes can discover meaningful phoneme-like categories
- **Internal feedback loss**: Add DIVA-inspired auditory error map loss term
- **Multi-agent communication**: Multiple Emitter-Receiver pairs through a common channel

### Scaling and optimization
- Scale joint training to dim=16/32 with 2k+ epochs (needs GPU or patience)
- Add early stopping to training loops to avoid wasting compute
- Consider LR scheduling (cosine annealing or step decay)

## Recently Completed

- Completed sensorimotor hypothesis research review (tasks/research.md)
- Created README.md with architecture diagram, results, milestones, and research context
- **Joint training experiment**: Joint beats sequential 2.7x at dim=8 (MSE 0.000710 vs 0.001909); near-equal at dim=16 (both underfitted)
- **VQ bottleneck capacity curve**: Codebook sizes 4-256 at dim=8; all fully utilized with near-max entropy; MSE decreases monotonically but even 256 codes far from continuous baseline
- **Rotational invariance experiment** (obj-013): System is NOT rotationally invariant. Orthogonal channels 17x easier than random, 471x easier than ill-conditioned. Receiver does all inversion (Jacobian ≈ M⁻¹), Emitter stays near-identity. Error concentrates on weak singular directions (~1/σ²).
