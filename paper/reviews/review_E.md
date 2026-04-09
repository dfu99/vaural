# Review E: The Skeptical ML Theorist

**Overall Score**: 4/10

## Summary

The paper empirically demonstrates that a learned encoder in a sequential training pipeline converges to the full-pipeline inverse, not the channel inverse. The C_i metric is cleanly defined and the measurement is convincing at the tested scale. However, the paper overclaims generality from an extremely narrow experimental setup (dim=8 MLPs with linear channels), lacks theoretical analysis, and omits results the authors apparently have that would substantially change the paper's conclusions.

## Strengths

- The C_i metric is well-defined and the falsification of the channel-inverse hypothesis is clean. Measuring C_i^{chan} ~ 0 alongside C_i^{pipe} = 1.0 is a crisp empirical result that is easy to verify and reproduce.
- The experimental design separates confounds effectively: trained vs. fixed receiver, linear vs. nonlinear receiver, different noise levels, multiple rotations. This is good empirical methodology within the chosen scope.
- The variance decomposition showing SiLU's residual CV is 80% SGD noise is a rigorous way to characterize the rotation invariance claim.
- The honest limitations section (Section 5) acknowledges most of the weaknesses I would raise. The question is whether the paper's claims are calibrated to these limitations.

## Weaknesses

- **Claims exceed evidence.** The paper claims "formal justification for cross-embodiment architectures" based on dim=8 experiments with linear channels and 3-layer MLPs. CrossFormer operates on 900K trajectories across 30 embodiments with high-dimensional observations. The gap between the experimental setup and the claimed implications is enormous. The word "formal" implies mathematical proof, which is not provided.
- **The crossover finding invalidates the main narrative.** The authors have data showing that sequential training wins at dim=8, but joint training wins by 1.8x at dim=16 and 1.5x at dim=32. This means the paper's central prescription -- use sequential training -- is wrong for any practical dimensionality. Omitting this result is a serious concern. It transforms the paper from "sequential training is better" to "sequential training is better only in a narrow low-dimensional regime," which is a much weaker and more nuanced claim.
- **No theoretical analysis.** For linear P, the convergence f_theta -> P^{-1} follows from elementary linear algebra (the MSE-optimal solution for ||Pf(s) - s||^2 when f is unconstrained is f = P^{-1}). This is not a deep result; it is a consequence of the training objective. The paper does not prove anything beyond what the objective function already implies. For nonlinear P, no analysis is provided.
- **Is the pipeline inverse finding trivial?** If you train f to minimize ||P(f(s)) - s||^2 and f has sufficient capacity, then f -> P^{-1} by definition of the training objective when P is invertible. The "surprise" framing suggests the authors expected f -> M^{-1}, but this expectation is only justified if one ignores the decoder in the optimization objective. The real contribution is identifying *which P* the encoder targets (composite, not channel-only) -- but this follows directly from the loss function.
- **Matched-parameters control is absent.** The matched-params ablation (random frozen receiver, same parameter count) fails at MSE=0.39. This is the key control that rules out the "more parameters" confound for sequential training. Without it, a skeptic could argue that sequential training simply benefits from having two separately trained networks rather than one jointly trained network.
- **Architecture generality is unchecked.** The authors have transformer results showing the same ordering (sequential < joint < matched at dim=8) -- this would partially address the "is this MLP-specific?" concern and should be included.

## Questions for Authors

1. For invertible linear P with unconstrained f, argmin ||P(f(s)) - s||^2 = P^{-1} by construction. What is the nontrivial content of the full-pipeline inverse finding beyond this?
2. The crossover at dim~16 is arguably the most important result. Can you derive conditions (in terms of model capacity, channel conditioning, or gradient dynamics) under which sequential training loses its advantage?
3. You claim "formal justification" for cross-embodiment architectures. Can you state this as a theorem with assumptions and conclusion? If not, "formal" should be replaced with "intuitive" or "empirical."
4. The gradient cosine similarity is near-zero. Is this because gradients are in orthogonal subspaces (benign) or because they are individually near-zero (uninformative)? What are the gradient norms?
5. What is the sample complexity of Phase 1 (receiver pre-training) vs. Phase 2 (emitter training)? If Phase 1 requires disproportionate data, the "efficiency" of sequential training may be illusory.

## Suggestions for Improvement

- Include the crossover finding as a central result and reframe the contribution: "We characterize the regime where sequential training outperforms joint training, finding a dimension-dependent crossover around dim=16."
- Add the matched-parameters ablation and the transformer ablation. These are already completed and their absence weakens the paper unnecessarily.
- Replace "formal justification" with "empirical motivation." Reserve "formal" for results backed by proofs.
- Provide a theoretical analysis, even for the linear case. Characterize the convergence rate of sequential vs. joint training as a function of condition number and dimensionality. This would give the crossover a theoretical explanation.
