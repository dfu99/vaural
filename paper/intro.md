# Abstract

When a learned encoder transmits through an unknown channel to a learned decoder, what does the encoder actually learn to invert? The naive expectation — that it learns the channel inverse — is wrong. We show empirically that the encoder converges to the *full-pipeline inverse*, including the decoder, with cosine alignment C_i = 1.0000. When the decoder is pre-trained to invert the channel, the pipeline collapses to identity and the encoder learns a trivial pass-through. When the decoder is a fixed linear transducer (modeling a physical sensor), the encoder learns the non-trivial composite inverse. We identify SiLU activation as critical for rotation invariance (CV 2% vs 19% for ReLU), trace the mechanism to ReLU's axis-aligned kinks (18 per trajectory vs 0 for SiLU), and characterize how alignment degrades gracefully under channel noise (R = -0.78 with reconstruction error). These findings formally justify the shared-trunk / per-embodiment-head architectures used in cross-embodiment robot learning.

# 1. Introduction

How does a speaker learn to produce sounds that a listener can understand through an unknown acoustic channel? The speaker cannot observe the channel directly — air pressure, room acoustics, and the listener's ear all transform the signal in ways the speaker does not control. Yet human speakers learn to communicate reliably, adapting to new environments within seconds. Understanding how this works is a foundational question in both neuroscience and machine learning.

We study this problem in a controlled setting: a learned encoder (Emitter) must transmit information through a fixed, unknown linear channel to a decoder (Receiver) that reconstructs the original signal. The system is trained sequentially — the decoder first learns to invert the channel, then is frozen while the encoder learns to produce signals the decoder can reconstruct. This two-phase protocol mirrors a developmental asymmetry observed in human speech acquisition: infants perceive speech months before they can produce it (Guenther, 2006).

The naive expectation is that the encoder should learn to compensate for channel distortion — to produce actions a = M⁻¹s that pre-invert the channel matrix M so the decoder receives a clean signal. We test this by measuring the coordination quality C_i, defined as the cosine alignment between the encoder's output and the theoretical optimal action.

**The surprise.** C_i measured against the channel inverse is approximately zero. The encoder does not learn M⁻¹. Instead, it learns the inverse of the *full downstream pipeline* — the composite of decoder and channel. When the decoder has been pre-trained to invert the channel (g_φ ≈ M⁻¹), the full pipeline P = g_φ ∘ M ≈ I, and the encoder correctly learns P⁻¹ ≈ I: the identity map. The encoder is not failing; it is solving the right problem. The decoder already did the hard work.

We confirm this by measuring C_i against the full-pipeline inverse, obtaining C_i^{pipe} = 1.0000 — perfect alignment in both direction and magnitude. The contrast with C_i^{chan} ≈ 0 is the first empirical demonstration that learned controllers converge to the composite inverse of their downstream system, not the plant inverse alone.

To verify this is not an artifact of the pre-trained decoder, we introduce a *fixed receiver regime* where the decoder is a random linear projection that is never trained — modeling a physical sensor like an ear. In this setting, the encoder must learn the non-trivial composite inverse (T · M)⁻¹, and it does so successfully (C_i^{pipe} = 1.0000, MSE = 0.0016). The encoder's Jacobian moves far from identity (Frobenius distance 36 vs 0.03 for the trained decoder), confirming it has learned an active compensation strategy.

Along the way, we discover that activation function choice is critical for robustness. Standard ReLU activations create orientation-dependent behavior: reconstruction quality varies by up to 2.3× across channel rotations at fixed spectrum. We trace this to ReLU's axis-aligned decision boundaries, which produce 18 sharp curvature spikes ("kinks") per output trajectory when the input is continuously rotated. Replacing ReLU with SiLU (smooth, axis-free gating) eliminates all kinks and reduces rotation sensitivity from CV = 19% to 2%, where the residual is dominated by SGD stochasticity rather than geometric bias.

Finally, we characterize the noise alignment boundary: under additive Gaussian channel noise, C_i^{pipe} degrades gracefully from 1.0 at σ = 0 to 0.65 at σ = 3.0, correlating with reconstruction error at R = -0.78. Pre-trained decoders provide substantial noise robustness that fixed transducers lack, offering a formal argument for learning perception rather than relying on fixed sensors.

These findings have practical implications for cross-embodiment robot learning. Systems like CrossFormer (Doshi et al., 2024) and GR00T N1 (Bjorck et al., 2025) use shared perception backbones with per-embodiment action heads — an architecture that implicitly decomposes the problem into invariant (shared) and variant (per-robot) components. Our pipeline inverse result provides the missing formal justification: when perception is pre-trained and shared, the per-embodiment controller need only learn a near-identity residual. When transferring to a new embodiment with different sensors, the controller must learn the full composite inverse — harder, but tractable for linear transducers.

Our contributions are:

1. **Empirical demonstration** that learned encoders converge to the full-pipeline inverse P⁻¹ (C_i = 1.0000), not the channel inverse M⁻¹ (C_i ≈ 0) — the first quantitative measurement of this distinction.

2. **Identification of SiLU** as critical for rotation invariance, with a mechanistic explanation: ReLU creates 18 axis-aligned kinks per trajectory; SiLU creates zero.

3. **Formal justification** for shared-trunk / per-embodiment-head architectures in cross-embodiment robotics, via the invariant/variant decomposition implied by the pipeline inverse.
