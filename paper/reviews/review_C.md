# Review C: The Communications Engineer

**Overall Score**: 4/10

## Summary

The paper studies a learned encoder-decoder system over a linear channel and finds that sequential training causes the encoder to converge to the full-pipeline inverse rather than the channel inverse. While the C_i metric is novel, the experimental setup is a stripped-down version of the autoencoder-over-channel paradigm established by O'Shea and Hoydis (2017), with important elements removed (constellation design, discrete messages, realistic channel models) rather than added. The paper needs to either engage seriously with the communications literature or reposition away from it.

## Strengths

- The C_i metric decomposition (channel vs. pipeline) is a genuinely new diagnostic that could be useful for analyzing learned communication systems. I am not aware of prior work that explicitly measures whether the encoder targets the channel inverse or the composite inverse.
- The rotation invariance analysis and SiLU recommendation are practical contributions. Activation function sensitivity to channel orientation is a real concern for learned modulation schemes, and the curvature-kink mechanism is a clean explanation.
- The noise alignment boundary experiment (Section 4.4) connects to capacity-like behavior, with the trained receiver providing implicit Wiener filtering. This is the most relevant result for communications and deserves more development.
- The SVD-based channel parameterization (Section 3.2) is sound and enables controlled experiments. Separating spectrum from orientation is standard but well-applied here.

## Weaknesses

- **Linear channels only.** This is the most severe limitation. The entire deep-learning-for-communications field exists because linear channels are already solved by classical methods (ZF, MMSE equalization, Viterbi decoding). The value of learned approaches is for nonlinear, time-varying, or analytically intractable channels. Testing only on random linear matrices M means the paper's results may not transfer to any setting where learned comms are actually useful. The structured channel results the authors reportedly have (sequential 20x faster on random, 6x on ill-conditioned, joint wins on noisy) would significantly address this concern if included.
- **No comparison to communications baselines.** There is no comparison to zero-forcing equalization, MMSE, or even a basic autoencoder trained end-to-end a la O'Shea and Hoydis. The reader cannot assess whether the sequential training regime offers any advantage over established methods.
- **No discrete messages or constellation design.** Real communication systems transmit discrete symbols. The continuous Gaussian sound tokens used here sidestep constellation design, peak power constraints, and the bit-error-rate metrics that communications engineers care about.
- **No information-theoretic analysis.** The paper mentions channel capacity in the discussion but never computes it. For a linear Gaussian channel, capacity is known exactly: C = 0.5 * sum(log(1 + sigma_k^2/N_0)). Comparing the system's achieved rate to capacity would ground the noise results in theory.
- **Missing channel diversity results.** The authors have data on how sequential vs. joint training performs across channel types (random, ill-conditioned, noisy). These structured channel experiments are essential for a communications audience and are inexplicably absent.

## Questions for Authors

1. For a linear Gaussian channel with known M, classical MMSE equalization is optimal. What does your system achieve relative to the MMSE bound? If it matches, the result is expected. If it does not, why not?
2. You report structured channel results (sequential 20x faster on random, 6x on ill-conditioned, joint wins on noisy). Why are these not in the paper? The finding that joint training wins on noisy channels is particularly important -- it suggests sequential training sacrifices noise robustness for speed.
3. Have you tested with OFDM-like channel models or frequency-selective fading? The linear random matrix model has no frequency structure, which removes a key challenge of real channels.
4. What happens with discrete inputs and a cross-entropy loss? Continuous reconstruction MSE does not tell us about symbol error rate or achievable rate.

## Suggestions for Improvement

- Include the structured channel ablation (random vs. ill-conditioned vs. noisy) as a core experiment. The finding that different channel types favor different training regimes is the most interesting result for communications.
- Add at least one nonlinear channel experiment (even a simple one like clipping or polynomial nonlinearity) to demonstrate that findings transfer beyond the linear case.
- Compute and compare to the MMSE bound for the linear Gaussian channel. This is a one-line computation and would ground the results information-theoretically.
- Either engage fully with communications (add baselines, discrete messages, BER curves) or reposition the paper away from communications and toward the control/neuroscience framing where the linear setting is less problematic.
