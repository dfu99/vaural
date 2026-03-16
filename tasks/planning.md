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

## Rotational Invariance Investigation Summary (obj-013 → obj-020)

Eight experiments systematically characterized rotational invariance:

1. **obj-013**: Channel κ dominates; Receiver inverts (Jacobian ≈ M⁻¹), Emitter ≈ identity
2. **obj-014**: Joint training reduces channel sensitivity 94% (474× → 28× ill/ortho ratio)
3. **obj-015**: Pure rotation test — system NOT rotationally invariant (CV 13-23% at fixed spectrum). ReLU axis-alignment causes inherent MLP orientation bias.
4. **obj-016**: Joint training halves rotation CV at κ=10 (23% → 10.5%) but is underfitted at κ=100 (CV 52%, MSE 85× worse). Sweet spot is moderate conditioning.
5. **obj-017**: Activation function test — ReLU hypothesis CONFIRMED. All smooth activations (GELU, SiLU, Tanh) reduce rotation CV from 19% to 6-9%. SiLU wins on both MSE (5.5× better) and invariance.
6. **obj-018**: 2×2 factorial (activation × training mode) — SiLU+Sequential is the best combination. Joint training makes rotation sensitivity WORSE (CV 28-44%), not better.
7. **obj-019**: Channel rotation adaptation — Emitter CAN learn the pre-rotation M₂⁻¹M₁ when forced to compensate (Jacobian 50-100× closer to target than identity). ~3× MSE penalty vs oracle. SiLU enables consistent adaptation (CV 10%) vs ReLU (CV 66%).
8. **obj-020**: Residual rotation sensitivity diagnosis — 80% of SiLU's remaining CV is training noise (SGD stochasticity), only 20% is true rotation dependence. Wider networks (h=128, 256) improve MSE but NOT invariance. LayerNorm hurts MSE 6× without improving CV.

**Key conclusions**: The rotational invariance investigation is **definitively complete**. SiLU with sequential training effectively solves rotation invariance — the residual CV (~2%) is dominated by training noise, not structural rotation bias. No further architectural intervention (wider networks, normalization) helps. The recommendation is simple: **use SiLU activation**. The default ReLU in components.py should be switched to SiLU.

**Open questions for future work**:
- How few epochs does adaptation need? (adaptation speed curve)
- Does κ affect adaptation quality? (harder channels = harder to adapt?)
- Can Receiver fine-tuning on the new channel close the 3× gap? (accent accommodation)
- Validate findings at dim=16+ (requires GPU)

## Recently Completed

- **Residual rotation sensitivity** (obj-020): 80% of SiLU's remaining CV is training noise, not rotation structure. Wider nets and LayerNorm don't help. SiLU effectively solves rotational invariance.
- **Channel rotation adaptation** (obj-019): Emitter learns pre-rotation M₂⁻¹M₁ when Receiver is frozen from different channel. ~3× MSE penalty. SiLU adaptation CV 10% vs ReLU 66%.
- **SiLU+Joint factorial** (obj-018): SiLU+Seq wins the 2×2 factorial. Joint training worsens rotation CV. Sequential is the right training mode.
- **Activation × rotational invariance** (obj-017): ReLU hypothesis confirmed — smooth activations reduce rotation CV from 19% → 6-9%. SiLU best overall (5.5× MSE, 8.8% CV).
- **Joint training × rotation sensitivity** (obj-016): Joint halves rotation CV at κ=10 but is underfitted at κ=100. Sweet spot is moderate conditioning.
- **Pure rotational invariance test** (obj-015): System NOT rotationally invariant — CV 13-23% across rotations at fixed spectrum. ReLU axis-alignment bias.
- **Joint vs channel geometry** (obj-014): Joint training reduces channel sensitivity by 94%.
- **Rotational invariance experiment** (obj-013): Receiver inverts channel, Emitter stays near-identity. Error ∝ 1/σ².
- **Joint training experiment** (obj-011): Joint beats sequential 2.7x at dim=8
- **VQ bottleneck capacity curve** (obj-012): All codebooks fully utilized; discretization bottleneck is fundamental
- Completed sensorimotor hypothesis research review (tasks/research.md)
