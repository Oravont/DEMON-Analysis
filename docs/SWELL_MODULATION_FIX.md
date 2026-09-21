# SWELL_MODULATION_FIX

This note documents a synthetic-data artefact that can contaminate DEMON analysis.

## Issue (historical)

A fixed swell modulation frequency produced strong low-frequency harmonics that leaked into the DEMON band
and could mask true propulsion signatures.

Example of problematic pattern (fixed frequency):
- swell frequency fixed at **0.5 Hz**
- harmonics can appear at **1.0, 1.5, 2.0 Hz**, etc.

## Fix applied in `ship_noise_corrected.py`

- Randomise swell frequency to a realistic ocean swell range:
  - **0.05–0.15 Hz** (period ~7–20 s)
- Use a modest modulation depth (e.g. ~0.2) so swell remains a gentle amplitude variation

Result: swell-related content stays well below typical shaft-rate search bands (≈1–10 Hz) and does not dominate DEMON.
