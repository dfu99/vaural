# 3. Method

## 3.1 Pipeline Architecture

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

## 3.2 Channel Parameterization via SVD

The channel matrix M admits a singular value decomposition:

    M = U Σ V^T

where U, V ∈ O(d) are orthogonal rotation matrices and Σ = diag(σ₁, ..., σ_d) contains the singular values in decreasing order. The condition number κ = σ₁/σ_d controls the channel difficulty: κ = 1 (orthogonal, lossless) to κ → ∞ (singular, information-destroying).

This decomposition separates two independent factors:
- **Spectrum** (Σ): how much each direction is amplified or attenuated
- **Orientation** (U, V): which directions receive which gains

We exploit this decomposition to construct controlled experiments. By fixing Σ and varying U, V (or vice versa), we isolate the effect of channel rotation from channel conditioning. Specifically, we generate channel matrices as M = U · diag(σ₁, ..., σ_d) · V^T where σ_k = 10^{-k(log₁₀ κ)/(d-1)} for k = 0, ..., d-1, giving a logarithmically spaced spectrum with prescribed condition number κ.

## 3.3 Two-Phase Sequential Training

Training proceeds in two phases that mirror a developmental asymmetry observed in biological speech acquisition (Guenther, 2006): infants perceive speech months before they can produce it.

**Phase 1 — Receiver pre-training.** The Receiver g_φ learns to invert the channel M using an identity mapping. We sample sound tokens s ~ N(0, I), compute received signals r = M · s, and train g_φ to minimize:

    L_recv(φ) = E_s[ ||g_φ(M · s) - s||² ]

This teaches g_φ to approximate M⁻¹ in the sense that g_φ(r) ≈ M⁻¹ · r. After convergence, the Receiver parameters φ are frozen.

**Phase 2 — Emitter training.** With g_φ frozen, the Emitter f_θ learns to map sound tokens to actions such that the full pipeline reconstructs the input:

    L_emit(θ) = E_s[ ||g_φ(M · f_θ(s)) - s||² ]

Gradients flow through the frozen channel M and Receiver g_φ back into the Emitter via backpropagation. The Emitter must discover actions that, after transformation by M and decoding by g_φ, recover the original sound.

Both phases use Adam optimization with learning rate 10⁻³ and batch size 64.

## 3.4 The Full-Pipeline Transform

Define the full pipeline (excluding the Emitter) as:

    P = g_φ ∘ M : ℝ^d → ℝ^d

For a trained Receiver where g_φ ≈ M⁻¹, we have P ≈ M⁻¹ · M = I. For a fixed linear Receiver with weight matrix T, we have P = T · M, a different linear transform.

At convergence, the Emitter minimizes ||P(f_θ(s)) - s||², which requires f_θ → P⁻¹ when P is invertible and f_θ has sufficient capacity. The key observation is that P⁻¹ ≠ M⁻¹ in general: the optimal encoder depends on the decoder, not just the channel.

For a trained Receiver (P ≈ I), P⁻¹ ≈ I and the Emitter should converge to the identity map. For a fixed linear Receiver (P = T · M), P⁻¹ = M⁻¹ · T⁻¹ and the Emitter must learn a non-trivial composite inverse.

## 3.5 Fixed Receiver Regime

To test whether the Emitter can learn active channel compensation (rather than relying on a pre-trained Receiver), we introduce a fixed receiver regime motivated by a biological observation: ears are fixed physical transducers. They faithfully convert air pressure to neural signals but cannot be trained.

In this regime, the Receiver is a fixed random linear projection T ∈ ℝ^{d×d} (initialized once, never updated). The Emitter trains end-to-end against the frozen composite transform P = T · M. This forces the Emitter to learn f_θ → P⁻¹ = (T · M)⁻¹, which is non-trivial.

We also test a fixed random MLP Receiver (random weights, SiLU activations, never trained). This represents a nonlinear fixed transducer. As we show in Section 4, the nonlinear case is too hard for the Emitter to invert, while the linear case succeeds — consistent with the observation that biological sensors are approximately linear transducers.

## 3.6 Coordination Quality Metric

To measure what the Emitter actually learns, we define the coordination quality C_i as the expected cosine alignment between the Emitter's output and a reference optimal action.

**Channel-only variant:**

    C_i^{chan} = E_s[ cos(f_θ(s), M⁻¹ · s) ]

This measures alignment with the action that would perfectly compensate for the channel, ignoring the Receiver.

**Full-pipeline variant:**

    C_i^{pipe} = E_s[ cos(f_θ(s), P⁻¹ · s) ]

This measures alignment with the action that is optimal given the actual downstream pipeline (including the Receiver).

For a nonlinear Receiver g_φ, we approximate P as a linear map by computing the mean Jacobian of g_φ at representative input points, then composing with M.

The contrast between C_i^{chan} and C_i^{pipe} reveals what transform the Emitter has learned to invert. We also track the magnitude ratio ||f_θ(s)|| / ||P⁻¹ · s|| to assess whether the Emitter matches not just the direction but the scale of the optimal action.

## 3.7 Rotation Invariance Protocol

To assess whether the pipeline is sensitive to channel orientation (independent of conditioning), we fix the singular value spectrum Σ and sample multiple random rotation pairs (U_j, V_j) for j = 1, ..., N_rot. For each rotation, we construct M_j = U_j · Σ · V_j^T, train the full pipeline, and measure test MSE.

The coefficient of variation CV = std(MSE) / mean(MSE) across rotations at fixed spectrum quantifies rotation sensitivity. A rotationally invariant system would have CV ≈ 0. We test spectra at κ ∈ {1, 10, 100} and compare across activation functions (ReLU, GELU, SiLU, Tanh).

## 3.8 Noise Injection

To characterize robustness, we add isotropic Gaussian noise after the Environment transform during training:

    r = M · f_θ(s) + σ_n · ε,  ε ~ N(0, I)

where σ_n controls noise intensity. The Receiver (in the trained regime) is also pre-trained with the same noise level. At test time, noise is removed to measure the learned encoding quality independent of test-time stochasticity. We sweep σ_n ∈ {0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0} and track C_i^{pipe} to characterize the alignment boundary.
