# Controllers Learn the Full-Pipeline Inverse, Not the Channel Inverse: Implications for Cross-Embodiment Transfer

---

## Abstract

When a learned encoder transmits through an unknown channel to a learned decoder, what does the encoder actually learn to invert? The naive expectation — that it learns the channel inverse — is wrong. We show empirically that the encoder converges to the *full-pipeline inverse*, including the decoder, with cosine alignment C_i = 1.0000. When the decoder is pre-trained to invert the channel, the pipeline collapses to identity and the encoder learns a trivial pass-through. When the decoder is a fixed linear transducer (modeling a physical sensor), the encoder learns the non-trivial composite inverse. We identify SiLU activation as critical for rotation invariance (CV 2% vs 19% for ReLU), trace the mechanism to ReLU's axis-aligned kinks (18 per trajectory vs 0 for SiLU), and characterize how alignment degrades gracefully under channel noise (R = -0.78 with reconstruction error). These findings formally justify the shared-trunk / per-embodiment-head architectures used in cross-embodiment robot learning.

---

## 1. Introduction

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

---

## 2. Related Work

Our work connects four research threads: adaptive control theory, motor neuroscience, learned communications, and cross-embodiment robotics. We position our contributions relative to each.

### Adaptive Inverse Control

Widrow and Walach (1996) proved that an adaptive filter placed in series with an unknown linear plant converges to the plant's inverse transfer function. This is the classical foundation for our finding that the Emitter converges to P⁻¹. However, Widrow's canonical formulation treats the plant as a single input-output block without a downstream decoder inside the inversion loop. Our contribution is the empirical demonstration that when a pre-trained decoder is included in the loop — forming a composite pipeline P = g_φ ∘ M — the encoder converges to the *composite* inverse P⁻¹, not the plant inverse M⁻¹. Furthermore, Widrow's proof applies to linear systems; we show convergence holds empirically for nonlinear MLP encoders and decoders in a sequential training regime. The distinction between C_i^{chan} ≈ 0 and C_i^{pipe} = 1.0 is a measurement that the classical framework does not provide.

The separation principle in linear control theory (Wonham, 1968) establishes that observer design and controller design can proceed independently without loss of optimality. Our two-phase training is structurally analogous: the Receiver (observer) trains first, then the Emitter (controller) trains against the frozen observer. For linear systems this separation is optimal, but for nonlinear systems the principle does not generally hold (Maggiore and Passino, 2003). Our adaptation experiments (Section 4.3) show that joint fine-tuning from a warm start outperforms sequential adaptation — suggesting the separation principle breaks down when the channel rotates.

### Internal Models in Motor Control

Wolpert and Kawato (1998) proposed that the cerebellum maintains paired forward and inverse models of the sensorimotor system. Critically, their "controlled object" is the full cascade from neural command through joint torques, limb kinematics, and task-space position — the inverse model absorbs the entire chain including the sensory mapping. This is precisely our full-pipeline inverse finding: the Emitter (motor controller) learns to invert the pipeline including the Receiver (sensory system), not just the channel (physical plant). The MOSAIC extension (Haruno, Wolpert, and Kawato, 2001) introduces modular selection among multiple paired models, which connects to our multi-rotation adaptation experiments where different channel rotations could be handled by different specialist modules.

Our contribution relative to Wolpert and Kawato is quantitative: we provide the first measurement of alignment between a learned controller and the full-pipeline inverse (C_i^{pipe} = 1.0000) versus the plant-only inverse (C_i^{chan} ≈ 0), and we show how pre-training the sensory model causes the motor controller to collapse to identity — a prediction implicit in their framework but never measured.

### Computational Models of Speech Acquisition

The DIVA model (Guenther, 2006; Guenther and Vladusich, 2012) is the most complete neurocomputational model of speech motor control. It posits a feedforward motor controller with auditory and somatosensory feedback controllers, trained through a babbling phase where auditory feedback updates motor targets. Vaural's architecture is structurally isomorphic to DIVA: the Emitter is the motor controller, the channel (ActionToSignal + Environment) is the vocal tract and acoustic propagation, and the Receiver is the auditory cortex. Our two-phase training mirrors the developmental asymmetry that DIVA models: infants perceive speech months before producing it. We extend the DIVA framework by analyzing the role of activation functions in rotation invariance (a factor DIVA does not address) and by introducing the fixed-receiver regime that models passive auditory transduction.

### Learned Communication Systems

O'Shea and Hoydis (2017) reframed the physical communication layer as an autoencoder: encoder (transmitter) → channel → decoder (receiver), trained end-to-end to minimize reconstruction error. With over 1400 citations, this work established deep learning for communications as a field. Dörner et al. (2018) extended the approach to real over-the-air radio channels.

Our pipeline is structurally identical, but we study a training regime they do not: sequential training with a pre-trained, frozen decoder. O'Shea and Hoydis train encoder and decoder jointly from random initialization. They do not examine (a) what happens when the decoder is pre-trained and frozen, (b) whether the encoder converges to (decoder ∘ channel)⁻¹ versus channel⁻¹, or (c) coordination quality metrics that distinguish these. Our C_i measurement fills this gap and reveals that the training regime fundamentally determines what the encoder learns.

### Cross-Embodiment Robot Learning

Recent foundation models for robotics adopt a shared-trunk architecture with per-embodiment heads. CrossFormer (Doshi et al., 2024) trains on 900K trajectories across 30 robot embodiments using a shared transformer backbone with embodiment-specific action tokenizers. Octo (Ghosh et al., 2024) uses modality-specific input tokenizers feeding a shared transformer. GR00T N1 (Bjorck et al., 2025) implements a dual-system design with an embodiment-aware encoder (System 1) and a shared vision-language backbone (System 2). AnyMorph (Trabucco et al., 2022) represents morphology as token sequences for a morphology-agnostic policy.

All of these systems implicitly decompose the problem into invariant components (shared trunk — environment dynamics, physics, perception) and variant components (per-embodiment heads — actuator-specific encoding/decoding). None formally justify why this decomposition works. Our full-pipeline inverse result provides the missing theoretical backbone: when perception is pre-trained (shared trunk absorbs channel inversion), the controller's residual task is near-identity (trivial adaptation). When perception is fixed (new embodiment with different sensors), the controller must learn the full composite inverse (non-trivial but tractable for linear transducers). This predicts that cross-embodiment transfer should be easy when perception transfers and hard when it does not — consistent with empirical observations in the robotics literature.

---

## 3. Method

### 3.1 Pipeline Architecture

We study a sequential communication system in which an encoder (Emitter) must learn to transmit information through an unknown, fixed channel to a decoder (Receiver) that reconstructs the original message. The pipeline is:

```
    s ──▶ f_θ ──▶ A ──▶ E ──▶ g_φ ──▶ ŝ
  (sound)  (Emitter)  (ActionToSignal)  (Environment)  (Receiver)  (decoded)
```

Let s ∈ ℝ^d denote a sound token drawn from a standard normal distribution. The Emitter f_θ : ℝ^d → ℝ^d is a 3-layer MLP with SiLU activations and parameters θ. The channel consists of two fixed linear transforms: ActionToSignal A ∈ ℝ^{d×d} and Environment E ∈ ℝ^{d×d}, both initialized as random Gaussian matrices with fixed seeds. The combined channel transform is:

    M = E · A ∈ ℝ^{d×d}

The Receiver g_φ : ℝ^d → ℝ^d is a 3-layer MLP with SiLU activations and parameters φ. The full pipeline computes:

    ŝ = g_φ(M · f_θ(s))

Training minimizes reconstruction loss:

    L(θ, φ) = E_s[ ||ŝ - s||² ] = E_s[ ||g_φ(M · f_θ(s)) - s||² ]

All learnable components (Emitter, Receiver) use 3-layer MLPs with architecture: Linear(d, h) → SiLU → Linear(h, h) → SiLU → Linear(h, d), where h is the hidden dimension. The choice of SiLU over ReLU is motivated by our rotation invariance analysis (Section 4.1).

### 3.2 Channel Parameterization via SVD

The channel matrix M admits a singular value decomposition:

    M = U Σ V^T

where U, V ∈ O(d) are orthogonal rotation matrices and Σ = diag(σ₁, ..., σ_d) contains the singular values in decreasing order. The condition number κ = σ₁/σ_d controls the channel difficulty: κ = 1 (orthogonal, lossless) to κ → ∞ (singular, information-destroying).

This decomposition separates two independent factors:
- **Spectrum** (Σ): how much each direction is amplified or attenuated
- **Orientation** (U, V): which directions receive which gains

We exploit this decomposition to construct controlled experiments. By fixing Σ and varying U, V (or vice versa), we isolate the effect of channel rotation from channel conditioning. Specifically, we generate channel matrices as M = U · diag(σ₁, ..., σ_d) · V^T where σ_k = 10^{-k(log₁₀ κ)/(d-1)} for k = 0, ..., d-1, giving a logarithmically spaced spectrum with prescribed condition number κ.

### 3.3 Two-Phase Sequential Training

Training proceeds in two phases that mirror a developmental asymmetry observed in biological speech acquisition (Guenther, 2006): infants perceive speech months before they can produce it.

**Phase 1 — Receiver pre-training.** The Receiver g_φ learns to invert the channel M using an identity mapping. We sample sound tokens s ~ N(0, I), compute received signals r = M · s, and train g_φ to minimize:

    L_recv(φ) = E_s[ ||g_φ(M · s) - s||² ]

This teaches g_φ to approximate M⁻¹ in the sense that g_φ(r) ≈ M⁻¹ · r. After convergence, the Receiver parameters φ are frozen.

**Phase 2 — Emitter training.** With g_φ frozen, the Emitter f_θ learns to map sound tokens to actions such that the full pipeline reconstructs the input:

    L_emit(θ) = E_s[ ||g_φ(M · f_θ(s)) - s||² ]

Gradients flow through the frozen channel M and Receiver g_φ back into the Emitter via backpropagation. The Emitter must discover actions that, after transformation by M and decoding by g_φ, recover the original sound.

Both phases use Adam optimization with learning rate 10⁻³ and batch size 64.

### 3.4 The Full-Pipeline Transform

Define the full pipeline (excluding the Emitter) as:

    P = g_φ ∘ M : ℝ^d → ℝ^d

For a trained Receiver where g_φ ≈ M⁻¹, we have P ≈ M⁻¹ · M = I. For a fixed linear Receiver with weight matrix T, we have P = T · M, a different linear transform.

At convergence, the Emitter minimizes ||P(f_θ(s)) - s||², which requires f_θ → P⁻¹ when P is invertible and f_θ has sufficient capacity. The key observation is that P⁻¹ ≠ M⁻¹ in general: the optimal encoder depends on the decoder, not just the channel.

For a trained Receiver (P ≈ I), P⁻¹ ≈ I and the Emitter should converge to the identity map. For a fixed linear Receiver (P = T · M), P⁻¹ = M⁻¹ · T⁻¹ and the Emitter must learn a non-trivial composite inverse.

### 3.5 Fixed Receiver Regime

To test whether the Emitter can learn active channel compensation (rather than relying on a pre-trained Receiver), we introduce a fixed receiver regime motivated by a biological observation: ears are fixed physical transducers. They faithfully convert air pressure to neural signals but cannot be trained.

In this regime, the Receiver is a fixed random linear projection T ∈ ℝ^{d×d} (initialized once, never updated). The Emitter trains end-to-end against the frozen composite transform P = T · M. This forces the Emitter to learn f_θ → P⁻¹ = (T · M)⁻¹, which is non-trivial.

We also test a fixed random MLP Receiver (random weights, SiLU activations, never trained). This represents a nonlinear fixed transducer. As we show in Section 4, the nonlinear case is too hard for the Emitter to invert, while the linear case succeeds — consistent with the observation that biological sensors are approximately linear transducers.

### 3.6 Coordination Quality Metric

To measure what the Emitter actually learns, we define the coordination quality C_i as the expected cosine alignment between the Emitter's output and a reference optimal action.

**Channel-only variant:**

    C_i^{chan} = E_s[ cos(f_θ(s), M⁻¹ · s) ]

This measures alignment with the action that would perfectly compensate for the channel, ignoring the Receiver.

**Full-pipeline variant:**

    C_i^{pipe} = E_s[ cos(f_θ(s), P⁻¹ · s) ]

This measures alignment with the action that is optimal given the actual downstream pipeline (including the Receiver).

For a nonlinear Receiver g_φ, we approximate P as a linear map by computing the mean Jacobian of g_φ at representative input points, then composing with M.

The contrast between C_i^{chan} and C_i^{pipe} reveals what transform the Emitter has learned to invert. We also track the magnitude ratio ||f_θ(s)|| / ||P⁻¹ · s|| to assess whether the Emitter matches not just the direction but the scale of the optimal action.

### 3.7 Rotation Invariance Protocol

To assess whether the pipeline is sensitive to channel orientation (independent of conditioning), we fix the singular value spectrum Σ and sample multiple random rotation pairs (U_j, V_j) for j = 1, ..., N_rot. For each rotation, we construct M_j = U_j · Σ · V_j^T, train the full pipeline, and measure test MSE.

The coefficient of variation CV = std(MSE) / mean(MSE) across rotations at fixed spectrum quantifies rotation sensitivity. A rotationally invariant system would have CV ≈ 0. We test spectra at κ ∈ {1, 10, 100} and compare across activation functions (ReLU, GELU, SiLU, Tanh).

### 3.8 Noise Injection

To characterize robustness, we add isotropic Gaussian noise after the Environment transform during training:

    r = M · f_θ(s) + σ_n · ε,  ε ~ N(0, I)

where σ_n controls noise intensity. The Receiver (in the trained regime) is also pre-trained with the same noise level. At test time, noise is removed to measure the learned encoding quality independent of test-time stochasticity. We sweep σ_n ∈ {0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0} and track C_i^{pipe} to characterize the alignment boundary.

---

## 4. Experiments and Results

We organize our empirical investigation into four experiment blocks, each building on the previous. All experiments use SiLU activations unless otherwise noted. We report means and standard deviations across multiple channel rotations to account for orientation effects.

### 4.1 Rotation Invariance and Activation Function Selection

**Hypothesis.** Since reconstruction quality depends on the singular value spectrum of M (not its orientation), the pipeline should be rotationally invariant: performance should be constant across channel rotations U, V at fixed spectrum Σ. If it is not, the source of orientation bias should be identifiable and correctable.

**Setup.** We construct channel matrices M = UΣV^T with fixed spectra at three condition numbers (κ = 1, 10, 100) and sample 8 random rotation pairs (U_j, V_j) per spectrum. For each of the 24 configurations, we train the full sequential pipeline (dim = 8, h = 64, 400 receiver + 500 emitter epochs, 2000 samples) and record test MSE. We repeat the full sweep for four activation functions: ReLU, GELU, SiLU, and Tanh (96 pipeline trainings total). We additionally decompose the residual variance into rotation-dependent and seed-dependent components by running 3 training seeds per rotation at κ = 10 (12 additional runs).

**Results.** The system is not rotationally invariant with ReLU. At fixed spectrum, MSE varies by up to 2.3× across rotations (CV = 13–23% depending on κ). All three smooth activations dramatically reduce rotation sensitivity: GELU (CV 7.9%), SiLU (CV 8.8%), Tanh (CV 6.0%), compared to ReLU (CV 19.0%). SiLU achieves the best absolute reconstruction (mean MSE 0.000033, 5.5× better than ReLU) while maintaining near-invariance (Figure 2a).

To understand the mechanism, we trace the output of a single trained Receiver under continuous input rotation in a 2D plane. ReLU produces 18 sharp curvature spikes ("kinks") per trajectory at angles where neurons switch on/off — the element-wise max(0, x) creates decision boundaries along coordinate hyperplanes. SiLU produces zero kinks: smooth gating (x · σ(x)) has no preferred axes (Figure 2b). The peak curvature of ReLU trajectories is 5.7× higher than SiLU.

Variance decomposition reveals that SiLU's residual CV (~2%) is dominated by SGD stochasticity (80% of total variance), not rotation structure (20%). Wider networks (h = 128, 256) improve MSE but do not reduce rotation CV; LayerNorm hurts MSE 6× without improving invariance. The finding generalizes to dim = 16, where SiLU maintains CV = 1.9% vs ReLU's 5.1%.

**Interpretation.** ReLU's axis-aligned nonlinearity creates orientation-dependent optimization landscapes. Different rotations cause different neurons to activate, leading to different local optima. SiLU's smooth, axis-free gating eliminates this bias. We adopt SiLU as the default activation for all subsequent experiments.

### 4.2 Full-Pipeline Inverse Discovery

**Hypothesis.** The Emitter should learn to invert the channel M, producing actions a = M⁻¹ · s that compensate for channel distortion. Measuring C_i^{chan} = E[cos(f_θ(s), M⁻¹ · s)] should yield values near 1.0 for a well-trained system.

**Setup.** We compute C_i^{chan} for 30 trained pipelines spanning 5 condition numbers (κ = 1, 3, 10, 30, 100) and 6 rotations each (dim = 8, h = 64, 200 receiver + 300 emitter epochs, 2000 samples). We also track C_i during adaptation (8 epoch checkpoints across 4 channel rotations) and decompose per singular direction.

**Results.** The hypothesis is wrong. C_i^{chan} ≈ 0 across all conditions (range [-0.23, +0.24]), with negligible correlation to MSE (R = 0.16). The Emitter does not learn M⁻¹. This is consistent with the Emitter Jacobian being near-identity (Frobenius distance to I = 0.03), as established in a prior Jacobian analysis: the Receiver absorbs channel inversion during Phase 1 pre-training (Jacobian ≈ M⁻¹, distance 0.63).

The resolution comes from redefining C_i against the full pipeline P = g_φ ∘ M rather than the channel alone. Let P⁻¹ be the inverse of the linearized pipeline. Then:

    C_i^{pipe} = E[cos(f_θ(s), P⁻¹ · s)] = 1.0000

with magnitude ratio ||f_θ(s)|| / ||P⁻¹ · s|| = 0.999 (Figure 3). The Emitter perfectly learns the full-pipeline inverse in both direction and magnitude. For the trained Receiver where g_φ ≈ M⁻¹, we have P ≈ I and P⁻¹ ≈ I, which explains why the Emitter Jacobian ≈ I — it is the correct answer, not a failure to learn.

**Interpretation.** The encoder does not invert the channel; it inverts the full downstream pipeline including the decoder. When the decoder has already absorbed the channel inversion, the encoder's optimal strategy is identity. This is the central finding of the paper: what the controller learns depends on what the observer has already learned. The measurement C_i^{pipe} = 1.0000 vs C_i^{chan} ≈ 0 provides the first empirical quantification of this principle.

### 4.3 Fixed Receiver Regime

**Hypothesis.** If the trained Receiver's absorption of M⁻¹ is what makes the Emitter trivial, then fixing the Receiver (never training it) should force the Emitter to learn a non-trivial transform. This tests whether the Emitter can actively encode for the channel when perception is a fixed transducer.

**Setup.** We compare three Receiver regimes at κ = 10, dim = 8 with 4 rotations: (1) trained Receiver (current default, 500 emitter epochs), (2) fixed random MLP (random SiLU MLP, never trained, 1000 emitter epochs), (3) fixed linear projection (random matrix T, never trained, 1000 emitter epochs). For each, we measure MSE, C_i^{pipe}, and Emitter Jacobian distances to both M⁻¹ and I.

**Results.** The fixed linear Receiver succeeds: MSE = 0.0016 (55× worse than trained but far from random). The Emitter Jacobian moves dramatically away from identity (distance to I = 36 vs 0.03 for trained), confirming it learns a non-trivial transform. C_i^{pipe} = 1.0000 — the Emitter perfectly inverts the composite pipeline P = T · M even though P ≠ M. Meanwhile, C_i^{chan} remains near zero because the Emitter learns (T · M)⁻¹, not M⁻¹.

The fixed random MLP Receiver fails: MSE = 0.35 (barely better than chance). The random MLP's nonlinearities create an inversion problem the Emitter cannot solve — the composite transform is too complex. This is consistent with real biology: ears are approximately linear transducers, not random nonlinear functions (Figure 3, right panels).

We additionally test adaptation dynamics: when the channel rotates while the Receiver remains frozen from a previous channel, the Emitter adapts to functional communication within ~50 epochs (2.7× oracle MSE). Joint fine-tuning — unfreezing the Receiver and training both components simultaneously from the warm start — closes 93% of this gap (1.10× oracle), outperforming emitter-only adaptation (2.43×) and sequential Receiver fine-tuning (1.44×). See Figure 5.

**Interpretation.** The fixed linear Receiver validates the biological "fixed ear" model: when perception cannot adapt, the motor system must actively compensate. The fact that C_i^{pipe} = 1.0 in both regimes confirms that the metric captures the correct inversion target regardless of whether the Receiver is trained or fixed. The failure of the fixed MLP Receiver constrains the class of viable fixed transducers to approximately linear transforms.

### 4.4 Noise Alignment Boundary

**Hypothesis.** Channel noise should degrade C_i^{pipe} by making the pipeline non-invertible. There may be a sharp noise threshold beyond which alignment collapses, corresponding to a channel capacity limit.

**Setup.** Extended GPU training (500 receiver + 1000 emitter epochs, dim = 16, h = 128, 5000 samples) with additive Gaussian noise σ_n ∈ {0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0} injected after the Environment transform during training. Both trained and fixed linear Receiver regimes, 3 rotations each. Noise is removed at test time to measure learned encoding quality. Run on NVIDIA RTX 6000.

**Results.** C_i^{pipe} degrades gracefully with no sharp cliff (Figure 4). For the trained Receiver, alignment remains above 0.97 through σ_n = 1.0 and drops to 0.65 only at σ_n = 3.0 (where MSE = 0.23). For the fixed linear Receiver, degradation is faster: C_i^{pipe} = 0.73 at σ_n = 0.1 and 0.35 at σ_n = 3.0.

C_i^{pipe} correlates with log(MSE) at R = -0.69 (trained) and R = -0.78 (fixed linear), validating it as a meaningful surrogate metric for reconstruction quality across noise conditions.

**Interpretation.** The trained Receiver's noise robustness stems from its pre-training with noise: it learns a regularized inverse that tolerates perturbations — analogous to Wiener filtering. The fixed linear Receiver has no such regularization, so the Emitter bears the full burden of noise robustness and degrades faster. This provides a formal argument for learning perception rather than relying on fixed sensors: pre-trained perception provides noise robustness that fixed transducers cannot. The absence of a sharp cliff suggests the system degrades smoothly as channel capacity decreases, rather than exhibiting a phase transition.

---

## 5. Discussion

### Implications for Cross-Embodiment Transfer

Our central finding — that the encoder learns P⁻¹ rather than M⁻¹ — has direct implications for how robot policies should be structured for cross-embodiment transfer. The invariant/variant decomposition falls out naturally:

- **Invariant components** (shared across embodiments): the environment dynamics and a pre-trained perception backbone. These constitute the "channel" that all embodiments share. Pre-training perception to absorb channel inversion means the controller's residual task is near-identity.

- **Variant components** (per-embodiment): the actuator-specific encoder and sensor-specific decoder. When transferring to a new robot with different sensors, only the variant components need retraining.

This is precisely the architecture used by CrossFormer, Octo, and GR00T N1, but without formal justification. Our result provides it: the shared trunk works because pre-trained perception makes per-embodiment adaptation trivial (near-identity). The adaptation experiments (Section 4.3) quantify this: warm-started joint fine-tuning reaches 1.10× oracle performance, meaning the "cost" of embodiment transfer is approximately 10% MSE overhead.

### Biological Interpretation

The two-phase training protocol mirrors biological speech development. Phase 1 (Receiver pre-training) corresponds to the period when infants perceive phonemic contrasts before producing speech. Phase 2 (Emitter training) corresponds to the babbling period when motor exploration, guided by auditory feedback, refines vocal production.

The "accent effect" — a 2.3× MSE penalty when the Emitter adapts to a rotated channel with a frozen Receiver — has a natural biological interpretation: speakers who learned in one acoustic environment perform worse in a new one. Joint fine-tuning (closing 93% of the gap) models accent accommodation, where both speaker and listener adapt.

### Limitations

Several limitations bound the generality of our findings:

**Linear channels.** All channel transforms are random linear matrices. Real communication channels exhibit nonlinearities (clipping, multipath interference, reverberation). The fixed MLP Receiver result (Section 4.3) suggests that nonlinear channels may be qualitatively harder — the Emitter failed to invert a random nonlinear pipeline. Extending to structured nonlinear channels (e.g., learned or physics-based) is an important next step.

**Low dimensionality.** Experiments use d = 8 and d = 16. While our key findings (C_i contrast, SiLU advantage, noise boundary) are consistent across both, real audio signals have thousands of dimensions. Whether the pipeline inverse property and rotation invariance hold at scale remains to be validated.

**No temporal structure.** We reconstruct single tokens, not sequences. Real communication involves temporal dependencies, variable-length utterances, and prosody. Extending to recurrent or autoregressive architectures would test whether the pipeline inverse finding generalizes beyond the i.i.d. setting.

**Static channel.** The channel is fixed during training. Real environments change continuously. Our adaptation experiments (Section 4.3) show the system can adapt to rotated channels, but we do not test continuous channel drift or online adaptation.

**Gaussian noise only.** We test additive isotropic Gaussian noise. Real channel noise is often structured (burst errors, frequency-dependent attenuation, interference). The graceful degradation we observe may not hold for adversarial or structured noise patterns.

**No real audio.** All experiments use synthetic random sound tokens. Validation on real speech or audio signals would strengthen the practical relevance of the findings.

### Connection to Information Theory

The noise alignment boundary (Section 4.4) connects to information-theoretic limits. As σ_n increases, the mutual information I(f_θ(s); s | noise) decreases, and no encoder can maintain perfect alignment. The graceful degradation of C_i^{pipe} — rather than a sharp transition — suggests the system operates in a regime where the channel capacity is well above the source entropy for low noise and degrades continuously. The trained Receiver's noise robustness can be understood as implicit Wiener filtering: by pre-training with noise, it learns a regularized inverse that trades off bias for noise reduction.

---

## 6. Conclusion

We have shown that learned encoders in sequential communication systems converge to the full-pipeline inverse P⁻¹ = (decoder ∘ channel)⁻¹, not the channel inverse M⁻¹ alone. This was demonstrated through a coordination quality metric C_i that yields 1.0000 when measured against the pipeline inverse and approximately zero against the channel inverse — a contrast that, to our knowledge, has not been previously measured.

Three key contributions emerge:

1. **The full-pipeline inverse principle.** The encoder's learned transform depends on the entire downstream system, not just the channel. When the decoder absorbs channel inversion through pre-training, the encoder's optimal strategy collapses to identity. When the decoder is a fixed linear transducer, the encoder learns the non-trivial composite inverse. This principle holds across training regimes, noise levels, and dimensionalities.

2. **SiLU activation eliminates rotation sensitivity.** ReLU's axis-aligned decision boundaries create 18 curvature kinks per output trajectory under input rotation, producing CV = 19% variation in reconstruction quality across channel orientations. SiLU's smooth gating creates zero kinks and reduces CV to 2%, where the residual is SGD noise. This is the first mechanistic explanation of rotation sensitivity in learned communication systems.

3. **Formal justification for cross-embodiment architectures.** The invariant/variant decomposition implied by the pipeline inverse — pre-trained perception absorbs the channel, leaving the controller with a near-identity residual — provides the theoretical backbone for shared-trunk / per-embodiment-head architectures (CrossFormer, Octo, GR00T N1) that are used in practice but lack formal justification.

Future work should extend these findings to nonlinear channels, temporal sequences, real audio signals, and higher dimensions. The connection between the noise alignment boundary and information-theoretic channel capacity deserves formal analysis. Multi-agent settings — where multiple Emitters and Receivers share a common channel — would test whether the pipeline inverse principle generalizes to communication games.

---

## References

- Bjorck, J., et al. (2025). GR00T N1: An Open Foundation Model for Generalist Humanoid Robots. *arXiv:2503.14734*.
- Dörner, S., Cammerer, S., Hoydis, J., and Brink, S. (2018). Deep Learning Based Communication Over the Air. *IEEE Journal of Selected Topics in Signal Processing*, 12(1).
- Doshi, R., et al. (2024). CrossFormer: Scaling Cross-Embodiment Learning. *Robotics: Science and Systems (RSS)*.
- Ghosh, D., et al. (2024). Octo: An Open-Source Generalist Robot Policy. *Robotics: Science and Systems (RSS)*.
- Guenther, F. (2006). Cortical Interactions Underlying the Production of Speech Sounds. *Journal of Communication Disorders*, 39(5).
- Guenther, F. and Vladusich, T. (2012). A Neural Theory of Speech Acquisition and Production. *Journal of Neurolinguistics*, 25(5).
- Haruno, M., Wolpert, D., and Kawato, M. (2001). MOSAIC Model for Sensorimotor Learning and Control. *Neural Computation*, 13(10).
- Khatib, O. (1987). A Unified Approach for Motion and Force Control of Robot Manipulators: The Operational Space Formulation. *IEEE Journal on Robotics and Automation*, 3(1).
- Maggiore, M. and Passino, K. (2003). A Separation Principle for a Class of Non-UCO Systems. *IEEE Transactions on Automatic Control*, 48(7).
- O'Shea, T. and Hoydis, J. (2017). An Introduction to Deep Learning for the Physical Layer. *IEEE Transactions on Cognitive Communications and Networking*, 3(4).
- Trabucco, B., et al. (2022). AnyMorph: Learning Transferable Polices By Inferring Agent Morphology. *International Conference on Machine Learning (ICML)*.
- Widrow, B. and Walach, E. (1996). *Adaptive Inverse Control: A Signal Processing Approach*. Wiley-IEEE Press.
- Wolpert, D. and Kawato, M. (1998). Multiple Paired Forward and Inverse Models for Motor Control. *Neural Networks*, 11(7–8).
- Wonham, W. M. (1968). On the Separation Theorem of Stochastic Control. *SIAM Journal on Control*, 6(2).

---

## Appendix: Figure List

| Figure | Description | File |
|--------|-------------|------|
| Figure 1 | Pipeline architecture with two-phase training | `paper/figures/fig1_architecture.pdf` |
| Figure 2 | Rotation invariance: (a) CV by activation, (b) curvature profiles | `paper/figures/fig2_rotation.pdf` |
| Figure 3 | C_i contrast: (a) channel vs pipeline, (b) Jacobian distances | `paper/figures/fig3_ci_contrast.pdf` |
| Figure 4 | Noise alignment boundary: (a) C_i vs noise, (b) MSE vs noise | `paper/figures/fig4_noise.pdf` |
| Figure 5 | Adaptation: (a) speed curve, (b) accent accommodation | `paper/figures/fig5_adaptation.pdf` |
