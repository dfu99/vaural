# Review D: The Motor Control Neuroscientist

**Overall Score**: 6/10

## Summary

The paper formalizes and measures a principle that is well-known qualitatively in motor control: the motor system inverts the entire sensorimotor loop, not just the plant. The quantitative measurement (C_i^{pipe} = 1.0 vs C_i^{chan} ~ 0) is novel and the fixed-receiver experiments provide a clean test of the "fixed ear" hypothesis. However, the biological framing overpromises relative to what the model can deliver, and several experimental results that would strengthen the neuroscience connection are missing from the draft.

## Strengths

- The core measurement is valuable. Wolpert (1998) and the internal models literature describe qualitatively that the inverse model covers the full sensorimotor loop, but nobody has measured the alignment quantitatively. C_i^{pipe} = 1.0000 vs C_i^{chan} ~ 0 is a clean empirical confirmation with a new metric.
- The fixed-receiver regime is the strongest contribution from a neuroscience perspective. Modeling the ear as a fixed linear transducer and showing the motor system can still learn the composite inverse -- while failing when the transducer is nonlinear -- is a testable prediction about biological sensors. The finding that fixed linear succeeds (MSE 0.0016) while fixed MLP fails (MSE 0.35) is consistent with the approximately linear response of cochlear mechanics.
- The two-phase training protocol is appropriately connected to developmental asymmetry (perception before production). The DIVA model comparison is accurate and not overclaimed.
- The "accent effect" (2.3x MSE penalty for channel rotation with frozen receiver) is a nice emergent property with a natural biological interpretation.

## Weaknesses

- **Wolpert already describes this.** The paper acknowledges that the internal models literature implies the full-pipeline inverse, but the framing still suggests this is a "surprise" (line 19: "The surprise."). For the motor control community, the surprise would be if the encoder did NOT converge to the pipeline inverse. The contribution is the quantitative measurement, not the qualitative finding -- the paper should be honest about this.
- **No connection to neural data.** The model produces predictions (e.g., motor cortex activity should align with P^{-1}, not M^{-1}; the alignment should degrade with sensory noise following the curve in Figure 4), but none are tested against neural recordings. Even citing existing neurophysiology data that is consistent (or inconsistent) with the predictions would strengthen the paper.
- **The matched-parameters ablation is essential for the neuroscience story.** The reported result that a random frozen receiver with matched parameter count fails catastrophically (MSE 0.39) is the strongest evidence that *pre-training perception matters*, not just having a decoder in the loop. This directly supports the developmental asymmetry argument: perception must be trained first, not just present. Its absence from the draft weakens the core argument.
- **Scaling matters for biological relevance.** The auditory system processes signals with thousands of frequency channels, not 8. The crossover finding (sequential wins at dim=8, joint wins at dim=16+) raises a concerning question: does the two-phase developmental advantage disappear at biologically relevant scales? If so, the biological analogy may be misleading.
- **No multi-speaker or multi-listener setting.** Real speech involves multiple speakers adapting to multiple listeners in shared acoustic environments. The single emitter-receiver pair misses the combinatorial challenge of biological communication.

## Questions for Authors

1. You frame the full-pipeline inverse as a "surprise" but Wolpert and Kawato (1998) explicitly define the controlled object as the full sensorimotor cascade. Can you clarify what is genuinely new beyond the measurement?
2. The scaling crossover (sequential wins at dim=8, joint at dim=16+) -- does this undermine the biological analogy? Human speech perception develops before production, which maps to your sequential protocol, but your data suggests this only helps at low dimensionality.
3. The matched-parameters ablation (random frozen receiver, MSE=0.39) would be the strongest evidence for the developmental asymmetry claim. Why is it not in the draft?
4. Are there any neurophysiology datasets (e.g., motor cortex recordings during speech production) where you could test whether neural activity aligns with P^{-1} predictions?
5. Does the gradient orthogonality finding (-0.004 to -0.007) have a biological interpretation? Could it relate to the independence of sensory and motor learning pathways in the brain?

## Suggestions for Improvement

- Reframe the contribution honestly: the qualitative result is known from Wolpert; the quantitative measurement (C_i) and the matched-params/fixed-receiver experiments are the novel contributions.
- Add the matched-parameters ablation. It is the single most impactful missing result for the neuroscience narrative.
- Include the scaling crossover and discuss its implications for biological plausibility. If the developmental advantage (sequential training) disappears at higher dimensions, that is an important caveat for the biological framing.
- Generate testable predictions for neurophysiology: specific measurements that could be made in motor cortex, auditory cortex, or cerebellar recordings that would confirm or falsify the model's predictions.
