# DEMON-Analysis

**DEMON** (Detection of Envelope Modulation on Noise) analysis for underwater acoustics, focused on extracting
physically meaningful propulsion signatures: **shaft rate (SR)** and **blade rate (BR)**.

This repository is a **signal-processing companion** to **SKANN-SSL**: it consumes corrected `.npy` clips and
produces interpretable DEMON modulation spectra suitable for reports and briefings.

## What DEMON does

1. Bandpass filtering in a cavitation-dominated band (carrier)
2. Hilbert envelope extraction (amplitude modulation)
3. AC coupling (remove DC + slow drift)
4. FFT of the envelope → low-frequency modulation spectrum (default **0–50 Hz**)

Key relationships:
- **SR (Hz)** = RPM / 60
- **BR (Hz)** = SR × *N* blades

## Quick start

### 1) Install dependencies
```bash
pip install -r requirements.txt
```

### 2) Generate a DEMON plot from the included sample clips

Cargo sample:
```bash
python src/demon_cli.py examples/test_30s_cargo_v2.npy --output outputs/test_30s_cargo_v2_demon.png
```

Fishing sample:
```bash
python src/demon_cli.py examples/test_30s_fishing_v2.npy --output outputs/test_30s_fishing_v2_demon.png
```

Notes:
- `outputs/` is **ignored by git** (local artefacts only).
- The plot uses a standard dark theme and marks detected BR harmonics when a BR is available.

### 3) If you already know BR (and blade count)

```bash
python src/demon_cli.py examples/test_30s_cargo_v2.npy --br 22.0 --blades 4 --harmonics 4 --output outputs/cargo_known_br.png
```

## Repository layout

- `src/demon_core.py` — DEMON computation + BR/SR detection (signal processing)
- `src/demon_plotting.py` — **single source of truth** for plot styling
- `src/demon_cli.py` — CLI tool to generate plots from `.npy` clips
- `examples/` — two small sample `.npy` clips + metadata CSV
- `assets/` — curated figures for the repo (tracked)
- `docs/` — project notes and the SKANN-SSL swell-modulation artefact fix

## Documentation

- Project notes: `docs/DEMON_ANALYSIS_PROJECT_NOTES.md`
- Swell modulation artefact fix: `docs/SWELL_MODULATION_FIX.md`

## Licence

Add a licence if/when you want to publish beyond internal use.
