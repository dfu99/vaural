# 4. Experiments and Results

We organize our empirical investigation into four experiment blocks, each building on the previous. All experiments use SiLU activations unless otherwise noted. We report means and standard deviations across multiple channel rotations to account for orientation effects.

## 4.1 Rotation Invariance and Activation Function Selection

**Hypothesis.** Since reconstruction quality depends on the singular value spectrum of M (not its orientation), the pipeline should be rotationally invariant: performance should be constant across channel rotations U, V at fixed spectrum Σ. If it is not, the source of orientation bias should be identifiable and correctable.

**Setup.** We construct channel matrices M = UΣV^T with fixed spectra at three condition numbers (κ = 1, 10, 100) and sample 8 random rotation pairs (U_j, V_j) per spectrum. For each of the 24 configurations, we train the full sequential pipeline (dim = 8, h = 64, 400 receiver + 500 emitter epochs, 2000 samples) and record test MSE. We repeat the full sweep for four activation functions: ReLU, GELU, SiLU, and Tanh (96 pipeline trainings total). We additionally decompose the residual variance into rotation-dependent and seed-dependent components by running 3 training seeds per rotation at κ = 10 (12 additional runs).

**Results.** The system is not rotationally invariant with ReLU. At fixed spectrum, MSE varies by up to 2.3× across rotations (CV = 13–23% depending on κ). All three smooth activations dramatically reduce rotation sensitivity: GELU (CV 7.9%), SiLU (CV 8.8%), Tanh (CV 6.0%), compared to ReLU (CV 19.0%). SiLU achieves the best absolute reconstruction (mean MSE 0.000033, 5.5× better than ReLU) while maintaining near-invariance (Figure 2a).

To understand the mechanism, we trace the output of a single trained Receiver under continuous input rotation in a 2D plane. ReLU produces 18 sharp curvature spikes ("kinks") per trajectory at angles where neurons switch on/off — the element-wise max(0, x) creates decision boundaries along coordinate hyperplanes. SiLU produces zero kinks: smooth gating (x · σ(x)) has no preferred axes (Figure 2b). The peak curvature of ReLU trajectories is 5.7× higher than SiLU.

Variance decomposition reveals that SiLU's residual CV (~2%) is dominated by SGD stochasticity (80% of total variance), not rotation structure (20%). Wider networks (h = 128, 256) improve MSE but do not reduce rotation CV; LayerNorm hurts MSE 6× without improving invariance. The finding generalizes to dim = 16, where SiLU maintains CV = 1.9% vs ReLU's 5.1%.

**Interpretation.** ReLU's axis-aligned nonlinearity creates orientation-dependent optimization landscapes. Different rotations cause different neurons to activate, leading to different local optima. SiLU's smooth, axis-free gating eliminates this bias. We adopt SiLU as the default activation for all subsequent experiments.

## 4.2 Full-Pipeline Inverse Discovery

**Hypothesis.** The Emitter should learn to invert the channel M, producing actions a = M⁻¹ · s that compensate for channel distortion. Measuring C_i^{chan} = E[cos(f_θ(s), M⁻¹ · s)] should yield values near 1.0 for a well-trained system.

**Setup.** We compute C_i^{chan} for 30 trained pipelines spanning 5 condition numbers (κ = 1, 3, 10, 30, 100) and 6 rotations each (dim = 8, h = 64, 200 receiver + 300 emitter epochs, 2000 samples). We also track C_i during adaptation (8 epoch checkpoints across 4 channel rotations) and decompose per singular direction.

**Results.** The hypothesis is wrong. C_i^{chan} ≈ 0 across all conditions (range [-0.23, +0.24]), with negligible correlation to MSE (R = 0.16). The Emitter does not learn M⁻¹. This is consistent with the Emitter Jacobian being near-identity (Frobenius distance to I = 0.03), as established in a prior Jacobian analysis: the Receiver absorbs channel inversion during Phase 1 pre-training (Jacobian ≈ M⁻¹, distance 0.63).

The resolution comes from redefining C_i against the full pipeline P = g_φ ∘ M rather than the channel alone. Let P⁻¹ be the inverse of the linearized pipeline. Then:

    C_i^{pipe} = E[cos(f_θ(s), P⁻¹ · s)] = 1.0000

with magnitude ratio ||f_θ(s)|| / ||P⁻¹ · s|| = 0.999 (Figure 3). The Emitter perfectly learns the full-pipeline inverse in both direction and magnitude. For the trained Receiver where g_φ ≈ M⁻¹, we have P ≈ I and P⁻¹ ≈ I, which explains why the Emitter Jacobian ≈ I — it is the correct answer, not a failure to learn.

**Interpretation.** The encoder does not invert the channel; it inverts the full downstream pipeline including the decoder. When the decoder has already absorbed the channel inversion, the encoder's optimal strategy is identity. This is the central finding of the paper: what the controller learns depends on what the observer has already learned. The measurement C_i^{pipe} = 1.0000 vs C_i^{chan} ≈ 0 provides the first empirical quantification of this principle.

## 4.3 Fixed Receiver Regime

**Hypothesis.** If the trained Receiver's absorption of M⁻¹ is what makes the Emitter trivial, then fixing the Receiver (never training it) should force the Emitter to learn a non-trivial transform. This tests whether the Emitter can actively encode for the channel when perception is a fixed transducer.

**Setup.** We compare three Receiver regimes at κ = 10, dim = 8 with 4 rotations: (1) trained Receiver (current default, 500 emitter epochs), (2) fixed random MLP (random SiLU MLP, never trained, 1000 emitter epochs), (3) fixed linear projection (random matrix T, never trained, 1000 emitter epochs). For each, we measure MSE, C_i^{pipe}, and Emitter Jacobian distances to both M⁻¹ and I.

**Results.** The fixed linear Receiver succeeds: MSE = 0.0016 (55× worse than trained but far from random). The Emitter Jacobian moves dramatically away from identity (distance to I = 36 vs 0.03 for trained), confirming it learns a non-trivial transform. C_i^{pipe} = 1.0000 — the Emitter perfectly inverts the composite pipeline P = T · M even though P ≠ M. Meanwhile, C_i^{chan} remains near zero because the Emitter learns (T · M)⁻¹, not M⁻¹.

The fixed random MLP Receiver fails: MSE = 0.35 (barely better than chance). The random MLP's nonlinearities create an inversion problem the Emitter cannot solve — the composite transform is too complex. This is consistent with real biology: ears are approximately linear transducers, not random nonlinear functions (Figure 3, right panels).

We additionally test adaptation dynamics: when the channel rotates while the Receiver remains frozen from a previous channel, the Emitter adapts to functional communication within ~50 epochs (2.7× oracle MSE). Joint fine-tuning — unfreezing the Receiver and training both components simultaneously from the warm start — closes 93% of this gap (1.10× oracle), outperforming emitter-only adaptation (2.43×) and sequential Receiver fine-tuning (1.44×).

**Interpretation.** The fixed linear Receiver validates the biological "fixed ear" model: when perception cannot adapt, the motor system must actively compensate. The fact that C_i^{pipe} = 1.0 in both regimes confirms that the metric captures the correct inversion target regardless of whether the Receiver is trained or fixed. The failure of the fixed MLP Receiver constrains the class of viable fixed transducers to approximately linear transforms.

## 4.4 Noise Alignment Boundary

**Hypothesis.** Channel noise should degrade C_i^{pipe} by making the pipeline non-invertible. There may be a sharp noise threshold beyond which alignment collapses, corresponding to a channel capacity limit.

**Setup.** Extended GPU training (500 receiver + 1000 emitter epochs, dim = 16, h = 128, 5000 samples) with additive Gaussian noise σ_n ∈ {0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0} injected after the Environment transform during training. Both trained and fixed linear Receiver regimes, 3 rotations each. Noise is removed at test time to measure learned encoding quality. Run on NVIDIA RTX 6000.

**Results.** C_i^{pipe} degrades gracefully with no sharp cliff (Figure 4). For the trained Receiver, alignment remains above 0.97 through σ_n = 1.0 and drops to 0.65 only at σ_n = 3.0 (where MSE = 0.23). For the fixed linear Receiver, degradation is faster: C_i^{pipe} = 0.73 at σ_n = 0.1 and 0.35 at σ_n = 3.0.

C_i^{pipe} correlates with log(MSE) at R = -0.69 (trained) and R = -0.78 (fixed linear), validating it as a meaningful surrogate metric for reconstruction quality across noise conditions.

**Interpretation.** The trained Receiver's noise robustness stems from its pre-training with noise: it learns a regularized inverse that tolerates perturbations — analogous to Wiener filtering. The fixed linear Receiver has no such regularization, so the Emitter bears the full burden of noise robustness and degrades faster. This provides a formal argument for learning perception rather than relying on fixed sensors: pre-trained perception provides noise robustness that fixed transducers cannot. The absence of a sharp cliff suggests the system degrades smoothly as channel capacity decreases, rather than exhibiting a phase transition.
