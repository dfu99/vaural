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
