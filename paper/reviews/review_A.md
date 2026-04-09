# Review A: The Multimodal Training Expert

**Overall Score**: 5/10

## Summary

This paper studies sequential vs. joint training in a learned encoder-channel-decoder pipeline, showing that the encoder converges to the full-pipeline inverse (C_i = 1.0) rather than the channel inverse. The SiLU rotation invariance finding is clean and well-characterized. However, the paper is hobbled by its restriction to toy dimensionalities and the absence of ablations that the authors appear to have already run but not included.

## Strengths

- The C_i metric is a genuinely useful diagnostic. The contrast between C_i^{chan} ~ 0 and C_i^{pipe} = 1.0 is striking and clearly presented. This measurement technique could be adopted by other groups studying learned communication systems.
- The SiLU rotation invariance analysis is thorough: the curvature kink decomposition, variance decomposition (80% SGD / 20% rotation), and the sweep across four activations all support the conclusion mechanistically.
- The fixed-receiver regime is a creative experimental design that separates what the encoder can learn from what the decoder has already absorbed. The biological "fixed ear" analogy is apt.
- Writing quality is high. The paper reads well, the notation is consistent, and the experimental setups are clearly described.

## Weaknesses

- **Toy scale only.** All experiments use dim=8 or dim=16 with 3-layer MLPs. The paper acknowledges this in limitations but does not attempt any scaling analysis. The authors' own data (not in draft) shows a crossover: sequential wins at dim=8 but joint wins at dim=16 and dim=32. This is a critical finding that fundamentally changes the paper's narrative -- sequential training is not universally better; it is better *at small scale*. Omitting this makes the paper's claims about cross-embodiment robotics (where d >> 100) unsupported.
- **Missing ablation table.** The matched-parameters control (random frozen receiver with equal parameter count) reportedly fails catastrophically (MSE 0.39 at dim=8, worsening to 0.83 at dim=32). This is the most important ablation for establishing that sequential pre-training *matters* rather than just adding parameters, and it is entirely absent from the draft.
- **No practical impact quantification.** The cross-embodiment framing promises practical relevance but delivers none. There is no comparison to any robotics baseline, no real-world validation, and no analysis at the dimensionalities where real systems operate.
- **Architecture ablation missing.** The authors have transformer results (1-layer, 4-head) showing identical ordering to MLP. This would strengthen the "architecture-agnostic" claim considerably and should be included.

## Questions for Authors

1. You have a dim=8/16/32 scaling ablation showing a crossover where joint training overtakes sequential. Why is this not in the paper? Does this not undermine the claim that sequential training is preferable?
2. The matched-parameters control (random frozen receiver) fails at 0.39 MSE. Including this would show that pre-training the receiver is essential, not just having more parameters. Can you add this as a table?
3. At what dimensionality would you expect your findings to break down? Can you extrapolate from the dim=8/16/32 trend?
4. Have you measured gradient cosine similarity between emitter and receiver gradients during joint training? If so, what does it show about gradient conflict vs. orthogonality?

## Suggestions for Improvement

- Add a scaling ablation section (dim=8/16/32 at minimum) with the crossover finding prominently featured. Reframe the contribution as characterizing *when* sequential beats joint, not claiming it always does.
- Include the matched-parameters ablation as a table or figure -- it is the strongest evidence that pre-training matters.
- Add the transformer results to demonstrate architecture agnosticism, even as a short subsection or appendix.
- Tone down the cross-embodiment robotics framing unless you can provide at least one experiment at a relevant scale (dim >= 64).
