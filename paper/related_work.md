# 2. Related Work

Our work connects four research threads: adaptive control theory, motor neuroscience, learned communications, and cross-embodiment robotics. We position our contributions relative to each.

## Adaptive Inverse Control

Widrow and Walach (1996) proved that an adaptive filter placed in series with an unknown linear plant converges to the plant's inverse transfer function. This is the classical foundation for our finding that the Emitter converges to P⁻¹. However, Widrow's canonical formulation treats the plant as a single input-output block without a downstream decoder inside the inversion loop. Our contribution is the empirical demonstration that when a pre-trained decoder is included in the loop — forming a composite pipeline P = g_φ ∘ M — the encoder converges to the *composite* inverse P⁻¹, not the plant inverse M⁻¹. Furthermore, Widrow's proof applies to linear systems; we show convergence holds empirically for nonlinear MLP encoders and decoders in a sequential training regime. The distinction between C_i^{chan} ≈ 0 and C_i^{pipe} = 1.0 is a measurement that the classical framework does not provide.

The separation principle in linear control theory (Wonham, 1968) establishes that observer design and controller design can proceed independently without loss of optimality. Our two-phase training is structurally analogous: the Receiver (observer) trains first, then the Emitter (controller) trains against the frozen observer. For linear systems this separation is optimal, but for nonlinear systems the principle does not generally hold (Maggiore and Passino, 2003). Our adaptation experiments (Section 4.3) show that joint fine-tuning from a warm start outperforms sequential adaptation — suggesting the separation principle breaks down when the channel rotates.

## Internal Models in Motor Control

Wolpert and Kawato (1998) proposed that the cerebellum maintains paired forward and inverse models of the sensorimotor system. Critically, their "controlled object" is the full cascade from neural command through joint torques, limb kinematics, and task-space position — the inverse model absorbs the entire chain including the sensory mapping. This is precisely our full-pipeline inverse finding: the Emitter (motor controller) learns to invert the pipeline including the Receiver (sensory system), not just the channel (physical plant). The MOSAIC extension (Haruno, Wolpert, and Kawato, 2001) introduces modular selection among multiple paired models, which connects to our multi-rotation adaptation experiments where different channel rotations could be handled by different specialist modules.

Our contribution relative to Wolpert and Kawato is quantitative: we provide the first measurement of alignment between a learned controller and the full-pipeline inverse (C_i^{pipe} = 1.0000) versus the plant-only inverse (C_i^{chan} ≈ 0), and we show how pre-training the sensory model causes the motor controller to collapse to identity — a prediction implicit in their framework but never measured.

## Computational Models of Speech Acquisition

The DIVA model (Guenther, 2006; Guenther and Vladusich, 2012) is the most complete neurocomputational model of speech motor control. It posits a feedforward motor controller with auditory and somatosensory feedback controllers, trained through a babbling phase where auditory feedback updates motor targets. Vaural's architecture is structurally isomorphic to DIVA: the Emitter is the motor controller, the channel (ActionToSignal + Environment) is the vocal tract and acoustic propagation, and the Receiver is the auditory cortex. Our two-phase training mirrors the developmental asymmetry that DIVA models: infants perceive speech months before producing it. We extend the DIVA framework by analyzing the role of activation functions in rotation invariance (a factor DIVA does not address) and by introducing the fixed-receiver regime that models passive auditory transduction.

## Learned Communication Systems

O'Shea and Hoydis (2017) reframed the physical communication layer as an autoencoder: encoder (transmitter) → channel → decoder (receiver), trained end-to-end to minimize reconstruction error. With over 1400 citations, this work established deep learning for communications as a field. Dörner et al. (2018) extended the approach to real over-the-air radio channels.

Our pipeline is structurally identical, but we study a training regime they do not: sequential training with a pre-trained, frozen decoder. O'Shea and Hoydis train encoder and decoder jointly from random initialization. They do not examine (a) what happens when the decoder is pre-trained and frozen, (b) whether the encoder converges to (decoder ∘ channel)⁻¹ versus channel⁻¹, or (c) coordination quality metrics that distinguish these. Our C_i measurement fills this gap and reveals that the training regime fundamentally determines what the encoder learns.

## Cross-Embodiment Robot Learning

Recent foundation models for robotics adopt a shared-trunk architecture with per-embodiment heads. CrossFormer (Doshi et al., 2024) trains on 900K trajectories across 30 robot embodiments using a shared transformer backbone with embodiment-specific action tokenizers. Octo (Ghosh et al., 2024) uses modality-specific input tokenizers feeding a shared transformer. GR00T N1 (Bjorck et al., 2025) implements a dual-system design with an embodiment-aware encoder (System 1) and a shared vision-language backbone (System 2). AnyMorph (Trabucco et al., 2022) represents morphology as token sequences for a morphology-agnostic policy.

All of these systems implicitly decompose the problem into invariant components (shared trunk — environment dynamics, physics, perception) and variant components (per-embodiment heads — actuator-specific encoding/decoding). None formally justify why this decomposition works. Our full-pipeline inverse result provides the missing theoretical backbone: when perception is pre-trained (shared trunk absorbs channel inversion), the controller's residual task is near-identity (trivial adaptation). When perception is fixed (new embodiment with different sensors), the controller must learn the full composite inverse (non-trivial but tractable for linear transducers). This predicts that cross-embodiment transfer should be easy when perception transfers and hard when it does not — consistent with empirical observations in the robotics literature.

## Summary of Novelty

| Claim | Prior art | Our contribution |
|-------|-----------|-----------------|
| Controller → plant inverse | Widrow (1996) | Extends to nonlinear MLPs; measures composite vs. plant-only inverse |
| Autoencoder communication | O'Shea & Hoydis (2017) | Sequential training; pre-trained frozen decoder regime |
| Full-pipeline inverse model | Wolpert & Kawato (1998) | First quantitative C_i measurement (1.0 vs ≈ 0) |
| Developmental asymmetry | Guenther DIVA (2006) | Activation function × rotation invariance analysis |
| Shared trunk / per-embodiment heads | CrossFormer, GR00T N1 | Formal justification via pipeline inverse decomposition |
