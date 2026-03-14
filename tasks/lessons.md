# Lessons — vaural

_Hard-won lessons, gotchas, and things that broke before._
_This file is append-mostly. Only remove entries proven wrong._

## Training Performance
- CPU training is very slow: full test suite takes ~25 min, dim=32 full training takes >10 min
- Replacing DataLoader with direct index slicing + pre-computing fixed transforms is faster
- Pre-computing `action_to_signal(sounds)` and `environment(signals)` in receiver pre-training avoids redundant forward passes every epoch

## Architecture Constraints
- **sound_dim must equal action_dim** — receiver pre-training uses identity mapping (sound=action). Asymmetric dims cause `RuntimeError: mat1 and mat2 shapes cannot be multiplied`. An assertion guards this in `pretrain_receiver()`.
- dim=8 converges easily even with very few epochs (300+500); dim≥16 needs 1000+ epochs

## Sweep Execution
- Parameter sweeps with many configs timeout on CPU. Run experiments individually or use very small epoch counts for quick screening, then validate winners with full training.

## Research Direction
- The two-phase training (pre-train Receiver, freeze, then train Emitter) is a deliberate design choice that mirrors developmental asymmetry (infants perceive speech before producing it). The key ablation is joint training vs. sequential — this directly tests whether sensorimotor coupling produces better or more sample-efficient solutions.
- The Machine Speech Chain (Hori et al., 2017) is the closest computational precedent — jointly trains ASR+TTS in a closed loop. Vaural's novelty is in continuous representations through unknown physical channels rather than discrete text.
- A VQ bottleneck on Emitter output is the cleanest way to test emergent discrete units (phoneme-like codes) without pre-specifying a phoneme library.

## Joint Training Results
- Joint training (Emitter + Receiver trained simultaneously) beats sequential training by 2.7x at dim=8 (MSE 0.000710 vs 0.001909 with 600 total epochs, 2k samples).
- At dim=16, both methods are underfitted with 600 epochs and show near-equal MSE (~0.235). The joint advantage may require sufficient compute to manifest at higher dims.
- Joint training starts slower (higher initial loss) but converges to a lower final loss — consistent with the mutual regularization hypothesis from Machine Speech Chain literature.

## VQ Bottleneck
- EMA codebook updates (not gradient-based) are essential — gradient-based codebook updates cause encoder-codebook divergence where VQ loss dominates training.
- Initialize codebook from encoder outputs, not random — otherwise the codebook starts far from the encoder distribution and never recovers.
- Random continuous inputs (Gaussian N(0,1)) don't benefit from discrete VQ codes. With dim=8, even 256 codes (MSE=0.44) are far from continuous (MSE=0.002). Discretization only helps when inputs have exploitable structure (clusters, categories).
- Codebook utilization is excellent in this setup — all codes active, near-max entropy at every size. The VQ mechanism works; the bottleneck is fundamental capacity vs. continuous data.

## Channel Geometry & Rotational Invariance
- **The system is NOT rotationally invariant.** Channel condition number (κ) is the dominant factor in reconstruction quality. Orthogonal channels (κ=1) are 17x easier than random (κ=62) and 471x easier than ill-conditioned (κ=1000).
- **The Receiver does all channel inversion, not the Emitter.** Receiver Jacobian ≈ M⁻¹ (Frobenius distance 0.63). Emitter Jacobian ≈ I (Frobenius distance 0.50 from identity). The Emitter has no incentive to pre-compensate because the Receiver already handles the inversion during pre-training.
- **Error concentrates on weak singular directions.** Per-direction MSE is roughly proportional to 1/σ². The direction with σ=0.28 has 18.8x more error than σ=17.5. This means the MLP Receiver can't perfectly invert directions that are severely attenuated by the channel.
- **Running many duplicate experiment processes can happen** if the experiment takes a long time and multiple attempts are launched. Always check for existing processes before re-running.
- **Rotation matters, not just spectrum.** Even at κ=1 (identical singular values), different rotation matrices cause 1.43× MSE variation. This is inherent MLP orientation bias from ReLU axis-alignment — ReLU creates preferred directions along coordinate axes. At κ=10, CV reaches 23% (2.3× max/min). Spectrum dominates (4× effect from κ=1→100) but rotation is a meaningful secondary factor worth controlling for in experiments.
