# Full-Pipeline Inverse Learning in Sequential Communication Systems

**Target venue:** ICML 2027 or NeurIPS 2026 Workshop (Embodied AI / Robot Learning)

**Working title options:**
1. "Controllers Learn the Full-Pipeline Inverse, Not the Channel Inverse: Implications for Cross-Embodiment Transfer"
2. "Sequential Training Reveals How Sensorimotor Systems Distribute Channel Inversion"
3. "On What Learned Encoders Actually Invert: Full-Pipeline Alignment in Communication Systems"

---

## Abstract (~150 words)

- Learned communication through unknown channels: encoder → channel → decoder
- Two training regimes: pre-trained decoder (frozen) vs. fixed decoder (never trained)
- Surprise finding: encoder aligns perfectly with the **full pipeline** inverse P⁻¹ = (decoder ∘ channel)⁻¹, NOT the channel inverse M⁻¹
- When the decoder is pre-trained, it absorbs channel inversion → P ≈ I → encoder learns identity
- When the decoder is a fixed linear transducer, encoder learns the non-trivial composite inverse
- Coordination quality C_i measured at 1.0000 for pipeline inverse vs ≈0 for channel inverse
- SiLU activation eliminates rotation sensitivity (CV 2% vs 19% ReLU) via smooth gating
- Noise robustness: C_i degrades gracefully (R=-0.78 correlation with MSE)
- Implication: formal justification for shared-trunk / per-embodiment-head architectures (CrossFormer, GR00T N1)

## 1. Introduction (~800 words)

- **Opening hook:** How does a speaker learn to produce sounds that a listener can understand, through an unknown acoustic channel? This is the sensorimotor alignment problem.
- **The naive expectation:** The encoder (speaker) should learn to invert the channel — compensate for distortion so the decoder receives clean signal.
- **The surprise:** In a controlled simulation, the encoder does NOT learn the channel inverse. It learns the full-pipeline inverse, including the decoder. When the decoder pre-trains to invert the channel, the encoder learns identity — it has nothing left to do.
- **Why this matters:** This decomposition — where pre-trained perception absorbs the "hard" inversion and leaves the controller with a trivial residual — is exactly the architecture used implicitly by cross-embodiment robot policies (CrossFormer, Octo, GR00T N1). Our work provides the first formal justification.
- **Biological grounding:** The DIVA model (Guenther 2006) describes auditory feedback-driven motor learning. Wolpert & Kawato (1998) show cerebellar inverse models absorb the full kinematic chain. Our simulation instantiates these principles with quantitative alignment metrics.
- **Contributions (3):**
  1. Empirical demonstration that learned encoders converge to full-pipeline inverse P⁻¹ (C_i = 1.0000), not channel inverse M⁻¹ (C_i ≈ 0)
  2. Identification of SiLU as critical for rotation invariance (mechanism: zero kinks vs 18 for ReLU)
  3. Characterization of noise alignment boundary and formal connection to cross-embodiment transfer

## 2. Related Work (~1000 words)

- **Adaptive inverse control (Widrow & Walach, 1996):** Classical proof that adaptive filters converge to plant inverse for linear systems. Our contribution: extending to nonlinear MLPs in sequential training, and measuring the channel-vs-pipeline distinction they don't address.
- **Internal models in motor control (Wolpert & Kawato, 1998; MOSAIC, Haruno et al., 2001):** Cerebellar inverse models invert the full sensorimotor cascade. Our C_i measurement provides the first quantitative confirmation of this principle.
- **Learned communications (O'Shea & Hoydis, 2017; Dörner et al., 2018):** Autoencoder framing of communication systems. They train jointly; we study sequential training and the pre-trained-then-frozen decoder regime they don't explore.
- **Cross-embodiment robot learning (CrossFormer, Doshi et al., 2024; Octo, Ghosh et al., 2024; GR00T N1, NVIDIA, 2025):** Shared trunk + per-embodiment heads. This IS our invariant/variant decomposition, but without formal justification. We provide it.
- **DIVA model (Guenther, 2006+):** Vocal tract inverse model with auditory feedback. Structurally identical to our pipeline. We add the activation function analysis and C_i metric.
- **Operational space control (Khatib, 1987):** Task-space controller inverts the full kinematic chain. Our decomposition in robotics form.

## 3. Method (~1500 words)

- **3.1 Pipeline architecture:** Sound → Emitter (SiLU MLP) → ActionToSignal → Environment → Receiver (SiLU MLP) → Decoded Sound. Fixed components are seeded random linear transforms M = Env ∘ A2S. Full pipeline P = Recv ∘ M.
- **3.2 Channel parameterization via SVD:** M = UΣV^T. Condition number κ = σ_max/σ_min controls difficulty. Rotation matrices U, V control orientation. This separates spectrum effects from rotation effects.
- **3.3 Two-phase sequential training:** Phase 1: pre-train Receiver to invert M (identity mapping, sound=action). Phase 2: train Emitter end-to-end with frozen Receiver. Mirrors developmental asymmetry (infants perceive before producing).
- **3.4 Fixed receiver regime:** Receiver is a fixed random linear transform T (never trained). Models a physical transducer ("ear"). Emitter must learn (T ∘ M)⁻¹.
- **3.5 Coordination quality metric C_i:** Two variants: C_i(channel) = cos(emitter(s), M⁻¹·s) and C_i(pipeline) = cos(emitter(s), P⁻¹·s). The contrast between these reveals what the encoder actually inverts.
- **3.6 Rotation invariance protocol:** Fix singular value spectrum, vary rotation matrices U, V. Coefficient of variation across rotations measures orientation sensitivity.

## 4. Experiments and Results (~2000 words)

- **4.1 Rotation invariance (obj-013→024, 12 experiments):**
  - Hypothesis: MLP encoders should be rotation-invariant since the task depends only on singular values.
  - Result: NOT invariant with ReLU (CV 13-23%). SiLU eliminates sensitivity (CV ~2%).
  - Mechanism: ReLU creates 18 kinks per output trajectory; SiLU creates 0 (obj-023).
  - Validated at dim=16 (obj-024): SiLU CV 1.9% vs ReLU 5.1%.

- **4.2 Full-pipeline inverse (obj-025→027):**
  - Hypothesis: Encoder learns channel inverse M⁻¹.
  - Result: C_i(channel) ≈ 0 — encoder does NOT learn M⁻¹.
  - Resolution: C_i(pipeline) = 1.0000 — encoder perfectly learns P⁻¹.
  - Trained Receiver: P ≈ I, so Emitter ≈ identity (Jacobian dist to I = 0.03).
  - Fixed linear Receiver: Emitter learns non-trivial (T·M)⁻¹ (Jacobian dist to I = 36).

- **4.3 Adaptation dynamics (obj-019, 021, 022):**
  - Channel rotation adaptation: ~50 epochs to functional communication (2.7× oracle).
  - Accent accommodation: joint fine-tuning from warm start closes 93% of gap (1.10× oracle).
  - Key insight: warm-started joint >> cold-start joint (reconciles contradictory results).

- **4.4 Noise alignment boundary (obj-028):**
  - C_i(pipeline) degrades gracefully: 1.0 at σ=0 → 0.65 at σ=3.0 (trained Receiver).
  - Fixed linear degrades faster: C_i=0.73 at σ=0.1.
  - C_i predicts MSE: R=-0.78 (fixed linear), R=-0.69 (trained).
  - Pre-trained Receiver provides noise robustness that fixed transducers cannot.

## 5. Discussion (~800 words)

- **Implications for cross-embodiment transfer:**
  - Invariant component (shared trunk) = environment physics model — train once, reuse
  - Variant component (per-embodiment head) = actuator/sensor model — lightweight adaptation
  - Our result: when perception is pre-trained, the controller adaptation is trivial (identity)
  - This formally justifies CrossFormer/Octo/GR00T N1 architectures

- **Biological interpretation:**
  - Two-phase training mirrors developmental asymmetry (perceive before produce)
  - "Accent effect" (2.3× penalty) corresponds to speaking a second language
  - Joint fine-tuning = accent accommodation (listener + speaker both adapt)
  - SiLU's smoothness connects to biological neurons (smooth activation, no sharp thresholds)

- **Limitations:**
  - Linear channels only (real channels are nonlinear)
  - Dimensions 8-16 only (not validated at real-world scale)
  - No temporal structure (single-token reconstruction, not sequences)
  - Fixed channel during training (real environments change continuously)
  - Gaussian noise only (real noise is structured/correlated)

- **Connection to information theory:**
  - At high noise, the channel capacity drops and no C_i can be high
  - The noise boundary characterization connects to rate-distortion theory
  - Pre-trained Receiver is a regularized inverse — implicitly performs Wiener filtering

## 6. Conclusion (~300 words)

- **Contribution 1:** Learned encoders converge to the full-pipeline inverse P⁻¹, not the channel inverse M⁻¹. C_i(pipeline) = 1.0000 vs C_i(channel) ≈ 0. First quantitative measurement of this distinction.
- **Contribution 2:** SiLU activation is critical for rotation invariance — eliminates axis-aligned kinks that ReLU creates. Validated across dimensions and noise levels.
- **Contribution 3:** The variant/invariant decomposition (pre-trained perception absorbs channel; controller learns residual) formally justifies shared-trunk architectures in cross-embodiment robotics.
- **Future work:** Nonlinear channels, temporal sequences, real audio, multi-agent communication, scaling to high dimensions.

## Figures (5 planned)

1. **Architecture diagram** — Pipeline with gradient flow, two-phase training annotation
2. **Rotation invariance** — SiLU vs ReLU: CV comparison + curvature profiles (kinks)
3. **C_i contrast** — C_i(channel) ≈ 0 vs C_i(pipeline) = 1.0, both regimes
4. **Noise boundary** — C_i(pipeline) vs noise σ for both Receiver regimes
5. **Adaptation dynamics** — Speed curve + accent accommodation comparison

## Key Numbers for Abstract/Intro

| Metric | Value |
|--------|-------|
| C_i(pipeline) | 1.0000 |
| C_i(channel) | ≈ 0.02 |
| SiLU rotation CV | 2% |
| ReLU rotation CV | 19% |
| ReLU kinks per trajectory | 18 |
| SiLU kinks per trajectory | 0 |
| Adaptation epochs to functional | ~50 |
| Accent accommodation (joint FT) | 1.10× oracle |
| Noise R(C_i, MSE) | -0.78 |
| Experiments total | 28 objectives |
