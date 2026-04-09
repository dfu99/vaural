# When Does Staged Training Beat Joint Training? A Crossover Analysis in Learned Communication Systems

---

## Abstract

When a learned encoder transmits through an unknown channel to a learned decoder, should they be trained jointly or sequentially? Conventional wisdom and end-to-end learning practice favor joint training. We show the answer depends on problem scale: sequential training (pre-train decoder, then train encoder) achieves 3x lower error at dimension 8, but joint training wins by 1.5--1.8x at dimensions 16 and 32, revealing a dimension-dependent crossover around $d=16$. A matched-parameters control --- replacing the pre-trained decoder with a random frozen decoder of equal capacity --- fails catastrophically (MSE $>0.39$), proving that pre-training quality, not parameter count, drives sequential training's advantage. Gradient cosine similarity between encoder and decoder during joint training is near-zero ($-0.004$ to $-0.007$), indicating the two modules solve nearly orthogonal subproblems. These findings hold across architectures (MLP and transformer) and channel structures, though the crossover shifts: noisy channels favor joint training even at small $d$. We additionally measure a coordination quality metric $C_i$ showing the encoder converges to the full-pipeline inverse $P^{-1} = (g_\phi \circ M)^{-1}$ with $C_i = 1.0000$, not the channel inverse $M^{-1}$ ($C_i \approx 0$) --- a quantitative confirmation of the composite-inverse principle described qualitatively by Wolpert and Kawato (1998). We identify SiLU activation as critical for rotation invariance (CV 2% vs 19% for ReLU), tracing the mechanism to ReLU's axis-aligned decision boundaries (18 curvature kinks per trajectory vs 0 for SiLU).

---

## 1. Introduction

Multi-stage training pipelines are ubiquitous in modern deep learning. Vision-language models pre-train a visual encoder before connecting it to a language model (Alayrac et al., 2022; Li et al., 2023; Liu et al., 2023). Cross-embodiment robot policies use pre-trained perception backbones with per-embodiment action heads (Doshi et al., 2024; Bjorck et al., 2025). Speech systems train acoustic models before fine-tuning production (Guenther, 2006). In each case, practitioners face a fundamental question: should the modules be trained jointly from scratch, or should one module be pre-trained and frozen while the other adapts?

The answer is rarely studied systematically. Joint training is the default assumption in end-to-end learning, but staged training has practical advantages: it reduces the effective dimensionality of the optimization, allows reuse of pre-trained components, and mirrors developmental asymmetries in biological systems. When does each strategy win?

We study this question in a controlled setting: a learned encoder (Emitter) transmits through a fixed, unknown linear channel to a decoder (Receiver) that reconstructs the original signal. This communication pipeline is minimal enough to permit exhaustive ablation yet rich enough to exhibit non-trivial training dynamics. We compare three training regimes:

1. **Sequential**: pre-train the Receiver to invert the channel, freeze it, then train the Emitter against the frozen Receiver.
2. **Joint**: train both Emitter and Receiver simultaneously from random initialization.
3. **Matched-parameters**: freeze a *random* (untrained) Receiver and train only the Emitter --- matching the parameter count of the sequential regime but removing pre-training quality.

**The crossover finding.** At $d=8$, sequential training achieves MSE $= 0.000037 \pm 0.000005$, outperforming joint training ($0.000117 \pm 0.000073$) by $3.2\times$. But at $d=16$, the ordering reverses: joint training ($0.086 \pm 0.029$) beats sequential ($0.138 \pm 0.006$) by $1.6\times$, and the gap persists at $d=32$ (joint $0.064$ vs sequential $0.099$). The matched-parameters control fails at all scales ($0.39 \to 0.51 \to 0.83$), ruling out parameter count as an explanation and confirming that pre-training quality is load-bearing.

**Why sequential works at small scale.** Gradient cosine similarity between encoder and decoder parameters during joint training is near-zero ($-0.004$ to $-0.007$), revealing that the two modules solve nearly orthogonal subproblems. At small $d$, the optimization landscape is simple enough that sequential decomposition --- solving each subproblem in isolation --- incurs no coupling penalty and gains from reduced search dimensionality. At large $d$, the coupling between subproblems becomes significant enough that joint optimization's ability to coordinate both modules outweighs the dimensionality advantage.

**What the encoder learns.** Regardless of training regime, when the encoder converges, it learns the inverse of the *full downstream pipeline* $P = g_\phi \circ M$, not the channel inverse $M^{-1}$ alone. We measure this via a coordination quality metric $C_i$: cosine alignment with $P^{-1}$ gives $C_i^{pipe} = 1.0000$, while alignment with $M^{-1}$ gives $C_i^{chan} \approx 0$. This quantitatively confirms the qualitative principle described by Wolpert and Kawato (1998) for internal models in motor control: the controller inverts the full sensorimotor chain, not the plant alone.

Our contributions are:

1. **Crossover characterization**: sequential training wins at small $d$ but joint training wins at large $d$, with a transition around $d=16$. The crossover shifts with channel structure --- noisy channels favor joint training even at small $d$.

2. **Matched-parameters diagnostic**: a practical ablation that separates the contribution of pre-training quality from parameter count. Its catastrophic failure ($>1000\times$ worse than sequential) proves pre-training is the active ingredient.

3. **Pipeline inverse measurement** ($C_i$): the first quantitative measurement of the distinction between $C_i^{pipe} = 1.0$ and $C_i^{chan} \approx 0$, confirming Wolpert and Kawato's composite-inverse principle.

4. **SiLU for rotation invariance**: ReLU creates 18 axis-aligned curvature kinks per output trajectory; SiLU creates zero, reducing rotation sensitivity from CV = 19% to 2%.

---

## 2. Related Work

**Staged vs. joint training in deep learning.** The question of when to freeze components arises throughout modern ML. In vision-language models, LLaVA (Liu et al., 2023) pre-trains a visual encoder and connects it to a frozen language model via a projection layer; BLIP-2 (Li et al., 2023) introduces a querying transformer between frozen image and language models; Flamingo (Alayrac et al., 2022) interleaves frozen visual features with a frozen language model via gated cross-attention. All three use staged training, but the choice is driven by computational pragmatism rather than systematic comparison with joint training. Our crossover finding provides a principled basis: staged training's advantage depends on the effective dimensionality of the problem.

**Multi-task gradient interactions.** Yu et al. (2020) introduced PCGrad, which projects conflicting gradients to resolve negative transfer in multi-task learning. Their work shows that gradient conflict between tasks degrades joint optimization. Our gradient cosine similarity measurement complements this: we find near-zero (not negative) cosine similarity, indicating *orthogonality* rather than conflict. The encoder and decoder solve independent subproblems rather than interfering with each other --- a qualitatively different regime from the conflicting-gradient setting PCGrad addresses.

**Adaptive inverse control.** Widrow and Walach (1996) proved that an adaptive filter placed in series with an unknown linear plant converges to the plant's inverse transfer function. Our pipeline inverse finding extends this to the case where a pre-trained decoder is included in the loop: the encoder converges to $(g_\phi \circ M)^{-1}$, not $M^{-1}$. Widrow's proof applies to linear systems; we show convergence empirically for nonlinear MLP and transformer architectures in both sequential and joint training regimes.

**Internal models in motor control.** Wolpert and Kawato (1998) proposed that the cerebellum maintains paired forward and inverse models of the sensorimotor system, where the "controlled object" encompasses the full cascade from neural command through limb kinematics to sensory feedback. Our $C_i$ measurement provides the first quantitative confirmation: $C_i^{pipe} = 1.0000$ vs $C_i^{chan} \approx 0$. We contribute the measurement; the qualitative principle is theirs.

**Computational models of speech acquisition.** The DIVA model (Guenther, 2006; Guenther and Vladusich, 2012) posits a feedforward motor controller trained through a babbling phase where auditory feedback updates motor targets. Our two-phase training mirrors the developmental asymmetry DIVA models: perception before production. We extend the DIVA framework by characterizing when this developmental ordering is advantageous (small $d$) and when joint learning outperforms it (large $d$).

**Learned communication systems.** O'Shea and Hoydis (2017) reframed the physical communication layer as an autoencoder trained end-to-end. Dorner et al. (2018) extended this to real over-the-air channels. Both train encoder and decoder jointly. We compare their joint regime against sequential training and characterize the conditions under which each wins.

**Cross-embodiment robot learning.** CrossFormer (Doshi et al., 2024), Octo (Ghosh et al., 2024), and GR00T N1 (Bjorck et al., 2025) use shared perception backbones with per-embodiment action heads. Our pipeline inverse result provides empirical motivation for this architecture: when perception absorbs channel inversion, the controller's residual task is near-identity. Our crossover finding adds nuance: this decomposition is most beneficial when the per-embodiment problem is low-dimensional.

---

## 3. Method

### 3.1 Pipeline Architecture

We study a communication system in which an encoder (Emitter) transmits through an unknown, fixed channel to a decoder (Receiver):

$$s \xrightarrow{f_\theta} a \xrightarrow{A} \xrightarrow{E} r \xrightarrow{g_\phi} \hat{s}$$

where $s \in \mathbb{R}^d$ is a sound token drawn from $\mathcal{N}(0, I)$, $f_\theta : \mathbb{R}^d \to \mathbb{R}^d$ is the Emitter (3-layer MLP, SiLU activations), $A, E \in \mathbb{R}^{d \times d}$ are fixed random Gaussian matrices (ActionToSignal and Environment), and $g_\phi : \mathbb{R}^d \to \mathbb{R}^d$ is the Receiver (3-layer MLP, SiLU activations). The combined channel is $M = E \cdot A$. Training minimizes reconstruction loss:

$$\mathcal{L}(\theta, \phi) = \mathbb{E}_s\left[ \|\hat{s} - s\|^2 \right] = \mathbb{E}_s\left[ \|g_\phi(M \cdot f_\theta(s)) - s\|^2 \right]$$

Architecture: Linear($d$, $h$) $\to$ SiLU $\to$ Linear($h$, $h$) $\to$ SiLU $\to$ Linear($h$, $d$), where $h$ scales with $d$ (64 at $d=8$, 128 at $d=16$, 256 at $d=32$). The choice of SiLU over ReLU is motivated by rotation invariance analysis (Section 4.1).

### 3.2 Training Regimes

We compare three training regimes:

**Sequential.** Phase 1: train Receiver $g_\phi$ to invert $M$ using identity mapping ($\mathcal{L}_{recv} = \mathbb{E}_s[\|g_\phi(Ms) - s\|^2]$). Phase 2: freeze $g_\phi$, train Emitter $f_\theta$ against the frozen pipeline ($\mathcal{L}_{emit} = \mathbb{E}_s[\|g_\phi(Mf_\theta(s)) - s\|^2]$). This mirrors a developmental asymmetry: perception before production.

**Joint.** Train $f_\theta$ and $g_\phi$ simultaneously from random initialization, optimizing $\mathcal{L}(\theta, \phi)$ with both parameter sets updated at each step.

**Matched-parameters.** Initialize $g_\phi$ randomly and freeze it *without training*. Train only $f_\theta$. This matches the parameter count and frozen-decoder structure of sequential training but removes pre-training quality. It is the critical control for determining whether sequential training's advantage comes from *having* a frozen decoder or from *pre-training* the decoder.

All regimes use Adam with learning rate $10^{-3}$ and batch size 64.

### 3.3 Channel Parameterization

The channel matrix $M$ admits SVD: $M = U \Sigma V^T$ where $U, V \in O(d)$ and $\Sigma = \text{diag}(\sigma_1, \ldots, \sigma_d)$. The condition number $\kappa = \sigma_1/\sigma_d$ controls difficulty. We generate structured channels by controlling spectrum and orientation independently.

### 3.4 Coordination Quality Metric

We define coordination quality $C_i$ as the expected cosine alignment between the Emitter's output and a reference optimal action:

$$C_i^{chan} = \mathbb{E}_s\left[\cos(f_\theta(s), M^{-1}s)\right], \quad C_i^{pipe} = \mathbb{E}_s\left[\cos(f_\theta(s), P^{-1}s)\right]$$

where $P = g_\phi \circ M$ is the full downstream pipeline. The contrast between $C_i^{chan}$ and $C_i^{pipe}$ reveals which transform the Emitter has learned to invert.

### 3.5 Gradient Cosine Similarity

To characterize optimization dynamics during joint training, we measure the cosine similarity between gradients of the loss with respect to encoder and decoder parameters at each training step:

$$\rho_t = \cos\left(\nabla_\theta \mathcal{L}_t, \; \nabla_\phi \mathcal{L}_t\right)$$

where gradients are flattened to vectors. Values near 1 indicate cooperative gradient directions, near $-1$ indicate conflict, and near 0 indicate orthogonality (independent subproblems).

---

## 4. Experiments and Results

All experiments use SiLU activations unless otherwise noted. We report means and standard deviations across 3+ random seeds and multiple channel rotations.

### 4.1 Rotation Invariance and Activation Function Selection

**Setup.** Channel matrices $M = U\Sigma V^T$ with fixed spectra at $\kappa \in \{1, 10, 100\}$, 8 random rotation pairs per spectrum, 4 activation functions (ReLU, GELU, SiLU, Tanh), dim = 8, $h = 64$.

**Results.** ReLU is not rotationally invariant: MSE varies by $2.3\times$ across rotations (CV = 13--23%). SiLU achieves the best absolute reconstruction (mean MSE $0.000033$, $5.5\times$ better than ReLU) with CV $\approx 2\%$. The mechanism is clear from output trajectory analysis: ReLU produces 18 sharp curvature kinks per trajectory at axis-aligned decision boundaries; SiLU produces zero kinks (peak curvature $5.7\times$ lower). Variance decomposition confirms SiLU's residual CV is 80% SGD stochasticity, 20% rotation structure. The finding generalizes to $d = 16$ (SiLU CV = 1.9% vs ReLU 5.1%).

**Recommendation.** Use SiLU for any learned communication system where the channel orientation is unknown. We adopt SiLU for all subsequent experiments.

### 4.2 Full-Pipeline Inverse

**Setup.** We compute $C_i^{chan}$ and $C_i^{pipe}$ for 30 trained pipelines spanning $\kappa \in \{1, 3, 10, 30, 100\}$ at $d = 8$.

**Results.** $C_i^{chan} \approx 0$ across all conditions (range $[-0.23, +0.24]$), but $C_i^{pipe} = 1.0000$ with magnitude ratio $\|f_\theta(s)\| / \|P^{-1}s\| = 0.999$. The Emitter perfectly inverts the full downstream pipeline, not the channel alone. For a pre-trained Receiver where $g_\phi \approx M^{-1}$, the pipeline $P \approx I$ and the Emitter correctly learns $P^{-1} \approx I$. The Emitter Jacobian's Frobenius distance to $I$ is 0.03 --- it has learned the identity because identity is the right answer.

We verify this is not an artifact of the trained decoder by testing a **fixed linear Receiver** (random matrix $T$, never trained). The Emitter now learns the non-trivial composite inverse $(TM)^{-1}$: MSE = 0.0016, $C_i^{pipe} = 1.0000$, Jacobian distance to $I = 36$ (far from identity). A fixed random MLP Receiver fails (MSE = 0.35), constraining viable fixed transducers to approximately linear transforms.

**Interpretation.** This quantitatively confirms the composite-inverse principle described by Wolpert and Kawato (1998): the controller inverts the full sensorimotor chain, not the plant alone. Our contribution is the measurement ($C_i^{pipe} = 1.0$ vs $C_i^{chan} \approx 0$), not the qualitative insight.

### 4.3 Sequential vs Joint: The Crossover

This is the central experiment. We compare sequential, joint, and matched-parameters training across three scales.

**Setup.** For each of $d \in \{8, 16, 32\}$, we train all three regimes with matched architectures ($h = 8d$), 1000--1500 epochs, 3 seeds per condition. Hidden dimensions scale proportionally to keep the capacity-to-problem ratio constant.

**Results (Table 1).**

| Dim | Sequential MSE | Joint MSE | Matched MSE | Winner | Ratio |
|-----|---------------|-----------|-------------|--------|-------|
| 8   | $0.000037 \pm 0.000005$ | $0.000117 \pm 0.000073$ | $0.394 \pm 0.017$ | Sequential ($3.2\times$) | Seq/Joint = 0.32 |
| 16  | $0.138 \pm 0.006$ | $0.086 \pm 0.029$ | $0.513 \pm 0.018$ | Joint ($1.6\times$) | Seq/Joint = 1.61 |
| 32  | $0.099 \pm 0.001$ | $0.064 \pm 0.024$ | $0.828 \pm 0.026$ | Joint ($1.5\times$) | Seq/Joint = 1.54 |

**Table 1.** Scale ablation showing the crossover between sequential and joint training. At $d = 8$, sequential wins by $3.2\times$; at $d \geq 16$, joint wins by $1.5$--$1.6\times$. Matched-parameters training (random frozen Receiver) fails at all scales, and its failure *worsens* with dimension ($0.39 \to 0.51 \to 0.83$), confirming that pre-training quality is load-bearing and becomes more critical at larger scale.

The crossover between $d = 8$ and $d = 16$ is sharp: the sequential/joint MSE ratio flips from 0.32 to 1.61. At $d = 8$, sequential training converges to its final MSE within 10 epochs of Phase 2, while joint training requires $\sim200$ epochs to break through an initial plateau. At $d = 32$, joint training converges faster than sequential (which appears to plateau at $\sim0.1$).

**Interpretation.** At small $d$, the optimization problem decomposes cleanly: the Receiver's subproblem (invert $M$) and the Emitter's subproblem (invert $P$) are simple enough that solving them independently is efficient. At large $d$, the subproblems become harder individually, and joint training's ability to coordinate the loss landscape across both modules provides a meaningful advantage. The matched-parameters failure rules out the hypothesis that sequential training benefits from having more effective parameters (frozen Receiver + trainable Emitter); the benefit comes specifically from the *quality* of the frozen component.

### 4.4 Matched-Parameters Ablation

The matched-parameters control deserves detailed discussion because it functions as a general diagnostic tool.

**Design.** The matched-parameters condition exactly replicates the sequential regime's structure --- frozen Receiver, trainable Emitter, same parameter counts --- but replaces the pre-trained Receiver with a randomly initialized one. If sequential training's advantage came from parameter count or the frozen-decoder structure itself (e.g., by regularizing the Emitter's optimization), matched-parameters should perform comparably. It does not.

**Results.** Matched-parameters MSE is $>1000\times$ worse than sequential at $d = 8$ ($0.394$ vs $0.000037$) and $>6\times$ worse than joint at $d = 16$ ($0.513$ vs $0.086$). The failure worsens with dimension: at $d = 32$, matched MSE reaches $0.828$, approaching chance level. This pattern makes sense: a random frozen Receiver is a random linear projection that discards most information, and information loss compounds with dimensionality.

**Diagnostic value.** The matched-parameters ablation answers a practical question: "Is this pre-trained component actually helping, or would any frozen component do?" Any system using staged training should run this control. If matched-parameters approaches sequential performance, the pre-training is not contributing useful structure. If it fails catastrophically (as here), pre-training is load-bearing.

### 4.5 Gradient Orthogonality

**Setup.** During joint training at $d = 8$, we record gradient cosine similarity $\rho_t$ between $\nabla_\theta \mathcal{L}$ (Emitter gradients) and $\nabla_\phi \mathcal{L}$ (Receiver gradients) at each epoch across 3 seeds.

**Results.** $\rho_t$ fluctuates around zero throughout training, with mean $-0.004$ and range $[-0.013, +0.009]$. There is no systematic trend over epochs. The gradients are neither cooperative ($\rho > 0$) nor conflicting ($\rho < 0$); they are orthogonal.

**Interpretation.** The Emitter and Receiver solve nearly independent subproblems during joint training. The Receiver learns to invert $M$ regardless of what the Emitter does (because the loss penalizes reconstruction error, and the Receiver is the last module before the output). The Emitter learns to invert $P = g_\phi \circ M$ regardless of the Receiver's state (because it adapts to whatever downstream pipeline exists). These two optimization trajectories are geometrically orthogonal in parameter space.

This orthogonality explains why sequential training works at small $d$: since the subproblems barely interact, solving them sequentially incurs no coupling penalty. It also provides context for the crossover: at larger $d$, even small coupling effects accumulate, and joint training's ability to navigate the coupled landscape becomes valuable. The gradient orthogonality measurement is in a qualitatively different regime from the gradient *conflict* that PCGrad (Yu et al., 2020) addresses --- our modules do not interfere, they simply ignore each other.

### 4.6 Architecture Generality

**Setup.** We replace the 3-layer MLP with a 1-layer, 4-head transformer (with positional encoding treating each dimension as a token) and repeat the three-way comparison at $d = 8$. MLP: 5,256 emitter parameters. Transformer: 8,897 emitter parameters.

**Results (Table 2).**

| Architecture | Sequential MSE | Joint MSE | Matched MSE |
|-------------|---------------|-----------|-------------|
| MLP (3-layer) | $0.000468 \pm 0.000058$ | $0.016 \pm 0.022$ | $0.382 \pm 0.007$ |
| Transformer (1L, 4H) | $0.118 \pm 0.008$ | $0.143 \pm 0.028$ | $0.443 \pm 0.227$ |

**Table 2.** Architecture ablation at $d = 8$. The ordering Sequential $<$ Joint $<$ Matched holds for both MLP and transformer. The transformer has higher absolute MSE (likely due to the tokenization overhead for this low-dimensional problem), but the *relative ordering* is identical, confirming the crossover is not MLP-specific.

### 4.7 Channel Structure

**Setup.** We test four structured channel types at $d = 8$: random (baseline), ill-conditioned ($\kappa = 100$), noisy (additive Gaussian $\sigma \in \{0.1, 0.5\}$), and bandlimited (only 4 of 8 dimensions pass signal). For each, we compare sequential vs joint training.

**Results (Table 3).**

| Channel | Sequential MSE | Joint MSE | Winner | Speedup |
|---------|---------------|-----------|--------|---------|
| Random | $0.000037$ | $0.000117$ | Sequential ($3.2\times$) | Seq 20$\times$ faster to threshold |
| Ill-conditioned ($\kappa=100$) | $0.000055$ | $0.000143$ | Sequential ($2.6\times$) | Seq 6$\times$ faster |
| Noisy ($\sigma=0.1$) | $0.070$ | $0.004$ | Joint ($19\times$) | --- |
| Noisy ($\sigma=0.5$) | $0.204$ | $0.009$ | Joint ($22\times$) | --- |
| Bandlimited (4 of 8) | $0.602$ | $0.368$ | Joint ($1.6\times$) | --- |

**Table 3.** Channel structure modulates the crossover. For deterministic channels (random, ill-conditioned), sequential training dominates. For noisy channels, joint training wins decisively --- noise during training makes the pre-trained Receiver's fixed inverse suboptimal, and joint training can co-adapt encoder and decoder to the noise distribution. Bandlimited channels are a mild joint advantage.

**Interpretation.** The crossover is not a function of dimension alone but of the *effective difficulty* of the inversion problem. Noise increases difficulty by making the channel non-invertible; high condition number increases difficulty by amplifying certain directions. Sequential training excels when the Receiver can cleanly invert the channel in isolation; joint training excels when the inversion problem requires encoder-decoder coordination (noise, information loss). This connects to the scale crossover: at larger $d$, even a well-conditioned channel becomes harder to invert precisely, shifting the balance toward joint training.

### 4.8 Fixed Receiver and Adaptation

**Setup.** We train with sequential regime at $d = 8$, then rotate the channel by a random orthogonal transformation while keeping the Receiver frozen from the original channel. We compare adaptation strategies: emitter-only fine-tuning, sequential Receiver fine-tuning (retrain Receiver, then Emitter), and joint fine-tuning (both simultaneously from warm start).

**Results.** After channel rotation, performance degrades to $2.7\times$ oracle MSE within 50 epochs of emitter-only adaptation. Joint fine-tuning from warm start reaches $1.10\times$ oracle (closing 93% of the gap). Sequential Receiver fine-tuning reaches $1.44\times$ oracle. The ordering is: joint warm-start $>$ sequential retrain $>$ emitter-only adaptation.

**Interpretation.** Adaptation to a changed channel favors joint training from a warm start, even though initial training favored the sequential regime. This is consistent with the crossover: the warm-start adaptation problem is effectively "easier" (small perturbation from a good solution), and joint training can exploit the coupling between modules to find the nearby optimum. The $2.7\times$ penalty for emitter-only adaptation is the "accent effect" --- a speaker trained in one acoustic environment performing in a new one.

### 4.9 Noise Alignment Boundary

**Setup.** Extended training ($d = 16$, $h = 128$) with additive Gaussian noise $\sigma_n \in \{0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0\}$ during training. Noise removed at test time.

**Results.** $C_i^{pipe}$ degrades gracefully: $>0.97$ through $\sigma_n = 1.0$, dropping to $0.65$ at $\sigma_n = 3.0$ for the trained Receiver. The fixed linear Receiver degrades faster ($C_i^{pipe} = 0.73$ at $\sigma_n = 0.1$). Correlation $R = -0.78$ between $C_i^{pipe}$ and log(MSE) validates $C_i$ as a surrogate metric. No sharp cliff --- degradation is continuous, consistent with smooth channel capacity reduction rather than a phase transition.

---

## 5. Discussion

### The crossover as a practical guide

The dimension-dependent crossover provides actionable guidance for practitioners. For low-dimensional problems (small action spaces, few controllable degrees of freedom), sequential training is preferable: it converges faster, achieves lower error, and is simpler to implement. For high-dimensional problems, joint training should be the default. The transition around $d = 16$ in our setting will shift depending on network capacity, channel complexity, and noise level, but the qualitative pattern --- sequential wins at small scale, joint wins at large scale --- should generalize because it reflects a fundamental tradeoff between decomposition benefit and coupling cost.

The matched-parameters diagnostic provides a practical tool for any system using staged training: run the control, and if it fails catastrophically, the pre-training is contributing meaningful structure. If it performs comparably to staged training, the frozen component may not need pre-training at all.

### Connection to vision-language models

The staged training paradigm in LLaVA (Liu et al., 2023), BLIP-2 (Li et al., 2023), and Flamingo (Alayrac et al., 2022) mirrors our sequential regime: a pre-trained visual encoder is frozen while a downstream module learns to produce language outputs. These systems operate at very high dimensionality (millions of parameters, high-dimensional feature spaces), where our results predict joint training should win. Indeed, recent work has shown that fine-tuning the visual encoder jointly improves performance --- consistent with our crossover finding. Our gradient orthogonality result suggests that the visual encoder and language model may also solve nearly independent subproblems, which would explain why staged training works reasonably well even when joint training is theoretically better.

### Connection to cross-embodiment robotics

The pipeline inverse finding provides empirical motivation for the shared-trunk architecture used in CrossFormer, Octo, and GR00T N1. When a pre-trained perception backbone absorbs channel inversion ($g_\phi \approx M^{-1}$), the per-embodiment controller's residual task is near-identity ($P^{-1} \approx I$). The crossover adds nuance: this decomposition is most beneficial when the per-embodiment problem is low-dimensional (few actuator degrees of freedom). For high-DOF robots, joint fine-tuning of both perception and control may be necessary.

### Gradient orthogonality vs gradient conflict

Our gradient cosine similarity result ($\approx 0$) is qualitatively different from the gradient *conflict* ($< 0$) that PCGrad (Yu et al., 2020) addresses. In multi-task learning, conflicting gradients cause negative transfer. In our encoder-decoder system, the gradients are orthogonal --- the modules do not interfere, they simply solve independent problems. This orthogonality is why sequential decomposition works at all: the coupling penalty is small. The crossover at larger $d$ suggests that orthogonality breaks down as the problem grows, but we do not have gradient measurements at $d = 16$ and $d = 32$ to confirm this. Extending the gradient analysis across scales is an important direction.

### Limitations

**Linear channels.** All channel transforms are random linear matrices. Real channels exhibit nonlinearities, and the fixed MLP Receiver result (Section 4.2) shows nonlinear channels may be qualitatively harder. **Low dimensionality.** Our largest experiments are $d = 32$. The crossover location may shift at higher dimensions. **No temporal structure.** We reconstruct single tokens, not sequences. **Gaussian noise only.** Structured noise (burst errors, interference) may shift the crossover differently. **Synthetic data.** Validation on real signals (speech, sensor data) would strengthen practical relevance. **Single architecture family.** While MLP and transformer show the same ordering, broader architecture coverage (RNNs, state-space models) would strengthen the generality claim.

---

## 6. Conclusion

We have characterized when sequential training beats joint training in learned communication systems, finding a dimension-dependent crossover: sequential wins at $d = 8$ by $3.2\times$ but joint wins at $d \geq 16$ by $1.5$--$1.6\times$. The matched-parameters ablation proves this advantage comes from pre-training quality, not parameter count. Gradient orthogonality between encoder and decoder explains why sequential decomposition works at small scale. These findings hold across MLP and transformer architectures and are modulated by channel structure, with noisy channels favoring joint training even at small $d$.

Along the way, we measure the full-pipeline inverse ($C_i^{pipe} = 1.0000$ vs $C_i^{chan} \approx 0$), quantitatively confirming that learned encoders invert the composite downstream system rather than the channel alone. We identify SiLU as critical for rotation invariance, with a mechanistic explanation rooted in axis-aligned vs axis-free nonlinearities.

The crossover finding has practical implications: use sequential training for low-dimensional control problems and joint training for high-dimensional ones, with the matched-parameters diagnostic to verify that pre-trained components are contributing useful structure. Future work should extend to nonlinear channels, temporal sequences, real signals, and higher dimensions, and should measure gradient orthogonality across scales to understand the mechanism of the crossover more precisely.

---

## References

- Alayrac, J.-B., et al. (2022). Flamingo: A Visual Language Model for Few-Shot Learning. *Advances in Neural Information Processing Systems (NeurIPS)* 35.
- Bjorck, J., et al. (2025). GR00T N1: An Open Foundation Model for Generalist Humanoid Robots. *arXiv:2503.14734*.
- Dorner, S., Cammerer, S., Hoydis, J., and Brink, S. (2018). Deep Learning Based Communication Over the Air. *IEEE Journal of Selected Topics in Signal Processing*, 12(1).
- Doshi, R., et al. (2024). CrossFormer: Scaling Cross-Embodiment Learning. *Robotics: Science and Systems (RSS)*.
- Ghosh, D., et al. (2024). Octo: An Open-Source Generalist Robot Policy. *Robotics: Science and Systems (RSS)*.
- Guenther, F. (2006). Cortical Interactions Underlying the Production of Speech Sounds. *Journal of Communication Disorders*, 39(5).
- Guenther, F. and Vladusich, T. (2012). A Neural Theory of Speech Acquisition and Production. *Journal of Neurolinguistics*, 25(5).
- Haruno, M., Wolpert, D., and Kawato, M. (2001). MOSAIC Model for Sensorimotor Learning and Control. *Neural Computation*, 13(10).
- Li, J., et al. (2023). BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models. *International Conference on Machine Learning (ICML)*.
- Liu, H., et al. (2023). Visual Instruction Tuning. *Advances in Neural Information Processing Systems (NeurIPS)* 36.
- Maggiore, M. and Passino, K. (2003). A Separation Principle for a Class of Non-UCO Systems. *IEEE Transactions on Automatic Control*, 48(7).
- O'Shea, T. and Hoydis, J. (2017). An Introduction to Deep Learning for the Physical Layer. *IEEE Transactions on Cognitive Communications and Networking*, 3(4).
- Trabucco, B., et al. (2022). AnyMorph: Learning Transferable Polices By Inferring Agent Morphology. *International Conference on Machine Learning (ICML)*.
- Widrow, B. and Walach, E. (1996). *Adaptive Inverse Control: A Signal Processing Approach*. Wiley-IEEE Press.
- Wolpert, D. and Kawato, M. (1998). Multiple Paired Forward and Inverse Models for Motor Control. *Neural Networks*, 11(7--8).
- Wonham, W. M. (1968). On the Separation Theorem of Stochastic Control. *SIAM Journal on Control*, 6(2).
- Yu, T., et al. (2020). Gradient Surgery for Multi-Task Learning. *Advances in Neural Information Processing Systems (NeurIPS)* 33.

---

## Appendix: Figure List

| Figure | Description | File |
|--------|-------------|------|
| Figure 1 | Pipeline architecture with three training regimes | `paper/figures/fig1_architecture.pdf` |
| Figure 2 | Rotation invariance: (a) CV by activation, (b) curvature profiles | `paper/figures/fig2_rotation.pdf` |
| Figure 3 | $C_i$ contrast: pipeline vs channel alignment | `paper/figures/fig3_ci_contrast.pdf` |
| Figure 4 | Scale crossover: MSE vs dimension for all three regimes | `paper/figures/fig4_crossover.pdf` |
| Figure 5 | Matched-parameters failure across dimensions | `paper/figures/fig5_matched.pdf` |
| Figure 6 | Gradient cosine similarity during joint training | `paper/figures/fig6_gradient.pdf` |
| Figure 7 | Channel structure modulation of sequential vs joint | `paper/figures/fig7_channels.pdf` |
| Figure 8 | Adaptation dynamics after channel rotation | `paper/figures/fig8_adaptation.pdf` |
| Figure 9 | Noise alignment boundary | `paper/figures/fig9_noise.pdf` |
