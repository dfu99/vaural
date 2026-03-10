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
