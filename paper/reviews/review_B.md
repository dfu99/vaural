# Review B: The Gradient Dynamics Theorist

**Overall Score**: 5/10

## Summary

The paper presents an empirical study of what a learned encoder converges to in a sequential training pipeline, identifying the full-pipeline inverse as the target rather than the channel inverse. The coordination quality metric C_i is well-defined and the measurement is clean. However, the paper lacks theoretical grounding for its central claims and entirely omits gradient dynamics analysis that would strengthen the optimization story.

## Strengths

- The C_i metric is well-motivated and precisely defined. The contrast between C_i^{chan} and C_i^{pipe} is a clean falsification of the naive hypothesis, and the magnitude ratio check (0.999) rules out trivial alignment artifacts.
- The variance decomposition for rotation invariance (80% SGD / 20% rotation for SiLU) is rigorous and demonstrates the authors can do careful empirical analysis.
- The fixed-receiver regime creates a clean experimental separation. Comparing trained vs. fixed linear vs. fixed MLP receivers is good experimental design that rules out confounds.
- The linearization approach for computing P^{-1} through a nonlinear receiver (mean Jacobian) is reasonable and clearly described.

## Weaknesses

- **No gradient analysis whatsoever.** The paper is about a two-phase training protocol but never analyzes the gradient dynamics that make it work (or fail). The authors reportedly have gradient cosine similarity data showing near-zero values (-0.004 to -0.007) throughout training -- this is direct evidence that emitter and receiver gradients are orthogonal, not conflicting, which would explain *why* sequential training works at small scale: there is no gradient conflict to resolve. This belongs in the paper.
- **No theoretical justification for convergence to P^{-1}.** The claim that f_theta converges to P^{-1} is stated as an empirical observation but never derived. For linear systems this follows from Widrow and Walach (1996), but the paper uses nonlinear MLPs. At minimum, a local convergence argument around the identity (for the trained receiver case) should be tractable: if P ~ I + epsilon, then gradient descent on ||P(f(s)) - s||^2 with f initialized near I should converge to I - epsilon to first order.
- **The crossover phenomenon is unexplained.** The authors have data showing sequential wins at dim=8 but joint wins at dim=16+. This is the most theoretically interesting finding -- it suggests a capacity-dependent phase transition in the optimization landscape -- and it is completely absent. Is the crossover related to the rank of the gradient interaction matrix? To the conditioning of P? To overfitting in the sequential regime?
- **Loss landscape analysis missing.** No Hessian analysis, no loss surface visualization, no analysis of the optimization trajectory. The paper claims to study training dynamics but only looks at inputs and outputs, not the optimization process itself.

## Questions for Authors

1. You have gradient cosine similarity measurements near zero (-0.004 to -0.007). Does this change across training phases? Does it differ at dim=8 vs dim=32? If gradients are orthogonal at dim=8 but conflicting at dim=32, that would explain the crossover.
2. Can you provide a local convergence proof for the linear case? Widrow and Walach prove convergence to M^{-1} for the single-block case; extending to the composite pipeline P = g o M should be straightforward.
3. Is the crossover at dim~16 an artifact of MLP capacity (hidden dim vs. input dim ratio) or a fundamental property of the sequential training objective? What happens if you scale the hidden dimension proportionally?
4. What is the Hessian spectrum at convergence? Is the trained-receiver solution (emitter ~ identity) a sharper or flatter minimum than the joint-training solution?
5. Have you measured the effective rank of the gradient covariance matrix during Phase 2? If gradients are confined to a low-rank subspace, that would explain both fast convergence and limited scalability.

## Suggestions for Improvement

- Add the gradient cosine similarity analysis as a core result. This is direct evidence for the optimization dynamics argument and belongs alongside the C_i measurement.
- Provide at least a sketch proof for convergence to P^{-1} in the linear case, and a local analysis for the nonlinear case near the identity fixed point.
- Include the crossover finding and attempt to explain it through gradient dynamics (conflict vs. orthogonality as a function of dimensionality).
- Add a loss landscape section: Hessian eigenspectrum at convergence, or at minimum, training loss trajectory comparisons between sequential and joint training across dimensions.
