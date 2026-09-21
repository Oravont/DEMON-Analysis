# Validation — Methodology and Status

How DEMON detection accuracy is measured in this repository, what the failure modes are, and
where the published numbers stand.

## What is measured

`src/demon_batch_validator_v5.py` runs the DEMON chain over a labelled clip set and compares
three recovered quantities against the ground-truth manifest:

| quantity | tolerance | notes |
|---|---|---|
| Blade rate (BR) | ±1.0 Hz | the primary observable — comb spacing |
| Shaft rate (SR) | ±0.3 Hz | inferred, harder (see below) |
| Blade count *N* | exact | derived as BR / SR, so it inherits both errors |

Per-clip results are written to CSV, with summary statistics by vessel class and diagnostic plots
for misdetections.

## Two operating modes

**Optimal mode** uses class-specific bandpass and display settings. The vessel class is assumed
known or hypothesised. This measures ceiling performance — what DEMON can recover when the band
is chosen correctly.

**Blind mode** treats the band itself as a search problem, using particle-swarm optimisation to
find the band that maximises comb structure. No class knowledge is used. This is the realistic
deployment case, at the cost of compute and occasional convergence to the wrong region for
atypical vessels.

The gap between the two modes is itself the informative quantity. It measures how much a
lightweight classifier front-end would buy you.

## Why SR is harder than BR

BR and SR are not equally observable, and the difference is physical rather than algorithmic.

Cavitation is modulated at blade passage, so the blade rate and its harmonics are what the
envelope spectrum actually contains. The shaft rate is recovered indirectly, by finding a weaker
line at BR/*N*.

The difficulty: BR = *N* × SR, so the blade rate is *itself* a shaft-rate harmonic. With a low
blade count — three especially — BR = 3×SR sits in a family where 4×SR and 5×SR score equally
well under comb reasoning. A detector searching for the strongest regular comb has no principled
way to prefer the correct member. This is a property of the observable, not a defect in the
detector, and it bounds SR accuracy for low-blade-count vessels regardless of SNR or cavitation
strength.

Consequence: BR accuracy and SR accuracy should be read as separate results. A high BR recovery
rate alongside a low SR rate is the expected signature of this ambiguity, not evidence of a
broken chain.

## Two known traps

**The display window.** The conventional 0–50 Hz DEMON plot encodes merchant-vessel assumptions.
A fast small craft with several blades routinely has BR above 100 Hz. In validation runs on the
synthetic dataset, the standard window missed the blade rate for the overwhelming majority of the
small-craft class. The comb was present; the window was not. Always raise `--max-mod` before
concluding a fast craft has no recoverable comb.

**Fixed-frequency artefacts.** A periodic modulation at a fixed frequency does not stay politely
at its fundamental. An earlier version of the synthetic generator applied sea-swell amplitude
modulation at a fixed 0.5 Hz, printing harmonics at 1.0, 1.5 and 2.0 Hz — straight into the band
where shaft rates live, and 2.0 Hz is a perfectly plausible shaft rate for a vessel at 120 RPM.
An artefact that lands *on top of* a credible propulsion line is far more dangerous than an
obviously spurious one. See [SWELL_MODULATION_FIX.md](SWELL_MODULATION_FIX.md). The validator
carries a residual low-frequency comb check for this reason.

## Status of published results

No validation results are currently published in this repository.

Earlier result sets covered a single vessel class using a superseded detector, and are not
representative of the current chain. Publishing them would misstate performance in both
directions, so they have been withdrawn rather than reframed.

Results will be published when a run is available that meets all of the following:

- all vessel classes, not a single class
- the current detector and the corrected cavitation model
- both optimal and blind modes, reported separately
- per-class breakdown, with BR and SR reported independently
- blade-count distribution stated, since SR accuracy is conditioned on it

The validator that produces these results is in the repository, and the sample clips and
ground-truth metadata in `examples/` are sufficient to run it on a small scale.
