# Research — Sensorimotor Feedback Loops and Joint Embedding Spaces

## Hypothesis

A vocal communication model should be grounded in closely linked sensorimotor feedback loops — analogous to how infants learn to speak by relying on auditory feedback to adjust vocalizations. Rather than treating audio generation and audio processing as independent models, they should share a joint embedding space where generated and processed audio can be compared and aligned.

## Verdict: Well-grounded, but not entirely novel

The core intuition is strongly supported by neuroscience and has several computational precedents. The novelty lies in Vaural's specific framing (continuous representations through unknown physical channels) rather than the general principle of sensorimotor coupling.

---

## 1. Neuroscience Support

### DIVA Model (Guenther, Boston University)
The strongest validation. DIVA (Directions Into Velocities of Articulators) is a neurocomputational model with a feedforward motor controller plus auditory and somatosensory feedback controllers. During babbling, the system depends primarily on auditory feedback to update motor targets. The Auditory Error Map detects deviation from a target and sends corrective commands back into articulatory corrections — structurally identical to Vaural's gradient flow.

**Ref**: Guenther, "The DIVA model: A neural theory of speech acquisition and production" (PMC3650855)

### Sensorimotor Coupling Models
A 2004 model (PubMed 15068923) learns articulatory-auditory coupling during a babbling phase, with both perceptual and motor representations developing concurrently and bidirectionally — the "joint embedding" idea in biological form.

### INFERNO (2021)
Combines reinforcement learning and spiking neural networks to construct a vocal repertoire via computational babbling (PMC7891699).

### Caveat: Motor Theory of Speech Perception is Contested
Liberman and Mattingly's "Motor Theory of Speech Perception" (1985) and the mirror neuron-speech link are heavily disputed. Lesion studies show patients with motor cortex damage perform at ceiling on receptive speech tasks (PMC3681806). **Do not lean on motor theory as a foundation** — the DIVA feedback-loop framing is more defensible.

---

## 2. Existing Computational Models

### Machine Speech Chain (Hori et al., 2017)
**Most direct precedent.** Jointly trains ASR and TTS in a closed loop: `speech → ASR → text → TTS → speech`. The two models improve each other through the feedback loop. This is exactly the "joint sensorimotor loop" hypothesis, implemented in 2017.

**Ref**: arXiv:1707.04879

### AudioLM (Google, 2022)
Two-level tokenization: semantic tokens from HuBERT + acoustic tokens from SoundStream. Uses perceptual model latents as the generative prior — the perception model's representation space becomes the generative space.

**Ref**: arXiv:2209.03143

### HuBERT / wav2vec 2.0 (Meta)
Learn discrete speech units from self-supervised masked prediction, with no phoneme labels. HuBERT alternates between k-means clustering (perceptual grouping) and masked prediction, producing discrete units that empirically correlate with phonemes without explicit phoneme supervision.

**Refs**: arXiv:2106.07447, arXiv:2006.11477

### VQ-VAE for Speech
Provides a discrete bottleneck between encoder (perception) and decoder (generation), trained end-to-end. The codebook is a learned "phoneme library" with emergent entries.

**Ref**: arXiv:1711.00937

### Joint ASR/TTS (2025-2026)
Recent work (arXiv:2601.10770, arXiv:2410.18607) achieves simultaneous ASR and TTS with shared parameters and discrete latent spaces, outperforming independently trained models.

### CLAP (Contrastive Language-Audio Pretraining)
Learns shared text-audio embedding via contrastive loss. Provides a joint space for understanding and generation conditioning, though it doesn't close the feedback loop.

**Ref**: arXiv:2206.04769

---

## 3. Phonemes: Emergent vs. Symbolic

The hypothesis correctly identifies that phonemes are one specific, symbolic solution to a more general problem. Key findings:

- **Emergent phonemes are real**: HuBERT and wav2vec 2.0 discover units that correlate with phonemes without being given them
- **Generative Adversarial Phonology** (PMC7861218) models unsupervised phonological learning, showing category structures emerge from distributional learning
- **Data efficiency gap**: Neural models need 2-4 orders of magnitude more data than human infants for comparable representations
- **Vaural's sound tokens are already phoneme-like**: The fixed-size representation functions like a pre-specified inventory. A VQ bottleneck would make this emergent rather than assumed

### Giving the model a phoneme library
This is a valid approach but changes the problem from "emergence" to "grounding." With a fixed library, the model must learn which combinations map to meaningful units — a compositional problem. Without one, the model discovers its own discrete units — an emergence problem. The latter is more scientifically interesting and better aligns with the infant learning analogy.

---

## 4. What Vaural Already Does Right

Vaural's architecture is already a structural analog of the DIVA model:
- **Emitter** = motor controller (articulatory commands)
- **ActionToSignal + Environment** = vocal tract + acoustic propagation
- **Receiver** = auditory cortex (perception system)
- **Gradient flow** = sensorimotor feedback loop

The two-phase training (pre-train Receiver, then train Emitter) mirrors the developmental asymmetry: infants can perceive speech months before they can produce it.

---

## 5. What's Novel vs. Reinvented

### Already well-covered in literature
- Closed-loop ASR/TTS joint training (Machine Speech Chain, 2017)
- Shared latent spaces for audio (AudioLM, CLAP, VQ-VAE)
- Computational babbling models (DIVA, INFERNO)
- Emergent phoneme discovery (HuBERT, wav2vec 2.0)

### Potentially novel angles
1. **Communication through unknown random channels** — most Lewis signaling game literature uses discrete symbols; continuous representations through physical transforms is less explored
2. **Two-phase training ablation** — comparing sequential (freeze Receiver, then train Emitter) vs. joint training in this specific setting has not been done
3. **Motor command as separate representation** — the "action" intermediate space, distinct from both sound and signal, adds a biologically motivated extra layer most joint ASR/TTS models collapse

### Key weakness
The hypothesis implies joint embedding is necessary for good performance. Vaural already achieves MSE ~0.000254 with fully sequential training. The hypothesis needs an experiment showing that coupling produces better or more sample-efficient solutions.

---

## 6. Recommended Experiments

1. **Joint training**: Unfreeze Receiver during Emitter training (or train both from scratch). Compare convergence speed and final MSE to sequential approach.
2. **VQ bottleneck**: Add vector quantization to Emitter output to produce discrete action codes — test whether emergent "phonemes" arise.
3. **Asymmetric dimensions**: Alternative pre-training (random pairs instead of identity mapping) to remove sound_dim == action_dim constraint, enabling bottleneck experiments.
4. **Internal feedback signal**: Add a DIVA-inspired loss term penalizing distance between Emitter output and what the Receiver "expects" in its input space.
5. **Noise robustness**: Add Gaussian noise to the channel and measure how encoding strategies change.

---

## Key References

| Paper | Year | Relevance |
|-------|------|-----------|
| DIVA model (Guenther) | 2006+ | Strongest neuroscience validation |
| Machine Speech Chain (Hori et al.) | 2017 | Most direct computational precedent |
| wav2vec 2.0 | 2020 | Self-supervised discrete speech units |
| HuBERT | 2021 | Emergent phoneme discovery |
| VQ-VAE (van den Oord et al.) | 2017 | Discrete bottleneck for audio |
| AudioLM (Google) | 2022 | Perception latents as generative prior |
| CLAP | 2022 | Joint audio-text embedding |
| INFERNO | 2021 | Computational babbling with RL |
| Lewis signaling games | Various | Emergent communication theory |
| Joint ASR/TTS transformers | 2025-26 | Shared parameter generation + perception |

---

## Cross-Project: Sensory-Motor Alignment as Projection Learning

*Cross-posted from WorldNN (2026-03-18). Full formalization in
`/home2/Documents/code/WorldNN/tasks/research.md`.*

### Connection to vaural

The Emitter learning to map sound tokens → motor actions through a fixed
random channel (ActionToSignal + Environment) IS learning a sensory-motor
alignment operator $R$: finding the rotation/scaling that aligns the sound
embedding with the unknown motor-to-acoustic transform.

- The Receiver is a frozen downstream projection $W_m$
- The Emitter searches for $R$ such that $W_m R(\mathbf{e}_\text{sound})$
  reconstructs the input
- **Coordination quality** $\mathcal{C}_i = \text{cos}(\text{emitter\_output},
  \text{optimal\_action})$ could predict MSE across channel conditions
- The rotation invariance findings (obj-013 through obj-024) are directly
  about how robust $R$ is to rotations of the channel — SiLU reducing
  sensitivity to ~2% means the learned $R$ generalizes across orientations

### Unified framework (WorldNN ↔ vaural ↔ CorticalNN)

All three projects instantiate the same question: *how does an agent learn
to act effectively through a lossy, unknown channel?*

| Project | What is R? | What is the channel? | What varies? |
|---------|-----------|---------------------|-------------|
| WorldNN | Organism embedding layer | VAE + noise | Capacity (embed_dim), perception quality |
| vaural | Emitter MLP | ActionToSignal + Environment | Channel rotation, dimension |
| CorticalNN | 3D growth topology | Sparse connectivity | Growth parameters, depth |

### Prior art

See WorldNN research.md for full assessment. Key overlaps: Friston's FEP,
CCA/CLIP, Churchland rotational dynamics, BDNF critical periods. The
synthesis across all three projects may be novel — no single paper combines
developmental neuroscience, multimodal alignment, and controlled simulation.
