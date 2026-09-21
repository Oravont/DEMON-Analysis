# DEMON Analysis Project Notes

**DEMON** (Detection of Envelope Modulation on Noise) is a classical underwater-acoustics method for extracting
propulsion signatures from **broadband cavitation noise** by analysing low-frequency **amplitude modulation**.

This repository is intentionally **signal-processing only** (no ML). It is designed to consume corrected clips
from **SKANN-SSL** and provide physics-grounded features and report-quality plots.

## Core relationships

- **Shaft rate (SR)** = RPM / 60  (Hz)
- **Blade rate (BR)** = SR × *N* blades  (Hz)

## Canonical DEMON pipeline

1. **Bandpass** in a cavitation-dominated band (typical starting point: **500–2000 Hz**, vessel-dependent)
2. **Hilbert envelope** of the bandpassed signal
3. **AC coupling**: remove DC + slow drift on the envelope
4. **FFT of the envelope** → low-frequency modulation spectrum (default **0–50 Hz**)

## Visual standard (single source of truth)

The plot style is centralised in `src/demon_plotting.py`:

- Dark navy background (briefing/presentation friendly)
- Cyan spectrum trace
- Orange harmonic markers for BR multiples
- Normalised magnitude axis: **dB re max** (0 dB at spectrum peak)

## Why this exists alongside SKANN-SSL

DEMON provides a transparent physics check that complements self-supervised embeddings:
- quick validation of synthetic/real clips
- interpretable propulsion parameters (BR/SR)
- reproducible figures for reports and briefings

## The “2 Hz” artefact case study (SKANN-SSL)

A prior SKANN-SSL synthetic generator version introduced a dominant low-frequency artefact that
was not obvious in typical spectrograms but was immediately visible in DEMON.

Root cause: unrealistic, fixed swell modulation frequency created strong harmonics in the DEMON band.
Fix: randomise swell frequency to realistic ocean swell ranges (**0.05–0.15 Hz**) and reduce modulation depth.

See: `docs/SWELL_MODULATION_FIX.md`.
