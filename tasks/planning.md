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

## Rotational Invariance Investigation Summary (obj-013 → obj-023)

Eleven experiments systematically characterized rotational invariance:

1. **obj-013**: Channel κ dominates; Receiver inverts (Jacobian ≈ M⁻¹), Emitter ≈ identity
2. **obj-014**: Joint training reduces channel sensitivity 94% (474× → 28× ill/ortho ratio)
3. **obj-015**: Pure rotation test — system NOT rotationally invariant (CV 13-23% at fixed spectrum). ReLU axis-alignment causes inherent MLP orientation bias.
4. **obj-016**: Joint training halves rotation CV at κ=10 (23% → 10.5%) but is underfitted at κ=100 (CV 52%, MSE 85× worse). Sweet spot is moderate conditioning.
5. **obj-017**: Activation function test — ReLU hypothesis CONFIRMED. All smooth activations (GELU, SiLU, Tanh) reduce rotation CV from 19% to 6-9%. SiLU wins on both MSE (5.5× better) and invariance.
6. **obj-018**: 2×2 factorial (activation × training mode) — SiLU+Sequential is the best combination. Joint training makes rotation sensitivity WORSE (CV 28-44%), not better.
7. **obj-019**: Channel rotation adaptation — Emitter CAN learn the pre-rotation M₂⁻¹M₁ when forced to compensate (Jacobian 50-100× closer to target than identity). ~3× MSE penalty vs oracle. SiLU enables consistent adaptation (CV 10%) vs ReLU (CV 66%).
8. **obj-020**: Residual rotation sensitivity diagnosis — 80% of SiLU's remaining CV is training noise (SGD stochasticity), only 20% is true rotation dependence. Wider networks (h=128, 256) improve MSE but NOT invariance. LayerNorm hurts MSE 6× without improving CV.
9. **obj-021**: Adaptation speed curve — Emitter reaches functional communication (~2.7× oracle) within 50 epochs. Two-phase pattern: rapid discovery (0-50 epochs, 16×→2.7×) then slow refinement (50-200 epochs, 2.7×→2.3×). The 2.3× residual is the "accent effect."
10. **obj-022**: Accent accommodation — Joint fine-tuning (warm-started from M₁ Receiver) closes 93% of the accent gap (1.10× oracle). Sequential fine-tuning closes 69% (1.44× oracle). Key insight: joint training from warm start >> joint training from scratch (reconciling obj-018's negative result with obj-011's positive one).
11. **obj-023**: Mechanistic analysis — ReLU produces 18 kinks per output trajectory (5.7× higher max curvature) where neurons flip at specific rotation angles. SiLU produces 0 kinks. The mechanism: ReLU's binary neuron switching creates orientation-dependent loss landscape features; SiLU's smooth gating has no such structure.

**Key conclusions**: The rotational invariance investigation is **complete**. The full story:
- **Mechanism**: ReLU creates 18 kinks per trajectory vs 0 for SiLU — binary neuron switching at rotation-dependent angles creates orientation-dependent optimization landscapes
- **Static invariance**: SiLU activation eliminates rotation sensitivity (CV ~2%, residual is training noise)
- **Dynamic adaptation**: When channel rotates, Emitter adapts in ~50 epochs to 2.7× oracle (accent effect)
- **Accent accommodation**: Joint fine-tuning from warm start closes 93% of the gap, reaching 1.10× oracle
- **Architecture**: SiLU is now the default in components.py. No other architectural change needed.

**dim=16 validation (obj-024)**: SiLU advantage confirmed at dim=16 (CV 1.9% vs 5.1% ReLU, MSE 2.3× better). Both activations show less rotation sensitivity at higher dims (averaging over more directions), but SiLU consistently wins. No remaining open questions.

## Recently Completed

- **Coordination quality C_i** (obj-025): C_i ≈ 0 across all conditions — Emitter does NOT align with optimal action M⁻¹·s. Consistent with Emitter ≈ identity (obj-013). Magnitude ratio (R=0.69) is a better predictor than cosine (R=0.16). C_i is not useful for vaural's sequential training; may work in WorldNN's bottleneck regime.
- **dim=16 validation** (obj-024): SiLU advantage holds at dim=16 (CV 1.9% vs 5.1%, MSE 2.3× better). Both less sensitive at higher dims. No open questions remain.
- **Rotation mechanism** (obj-023): ReLU has 18 kinks per trajectory (5.7× spikier curvature) vs 0 for SiLU. Binary neuron switching at rotation-dependent angles is the root cause.
- **Accent accommodation** (obj-022): Joint fine-tuning closes 93% of accent gap (1.10× oracle). Sequential FT closes 69% (1.44×). Joint from warm start >> joint from scratch.
- **Adaptation speed curve** (obj-021): Emitter reaches functional adaptation in ~50 epochs (2.7× oracle). Two-phase: rapid discovery then slow refinement. Residual 2.3× penalty = "accent effect."
- **SiLU default switch**: Changed Emitter and Receiver activations from ReLU to SiLU in components.py. All 27 tests pass.
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
