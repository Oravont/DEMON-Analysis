# DEMON Visualizer Tool - Handover Brief

**Date:** January 27, 2026  
**Purpose:** Continue DEMON tool development in a fresh chat

---

## What is DEMON?

**DEMON** (Detection of Envelope Modulation on Noise) extracts blade rate and shaft rate from underwater acoustic recordings by:

1. Bandpass filtering to isolate cavitation noise (e.g., 500-1500 Hz)
2. Hilbert transform → amplitude envelope
3. FFT of envelope → reveals modulation frequencies
4. Peaks at blade rate (BR) and shaft rate (SR)

**Key relationship:** `BR = n × SR` where n = number of propeller blades

---

## Current Tool: `demon_visualizer.py`

### Features Implemented ✅

| Feature | Description |
|---------|-------------|
| **PSO optimization** | Finds optimal cavitation band by maximizing spectral quality |
| **Auto-detection** | Detects BR, SR, and infers blade count |
| **Interactive slider** | Real-time smoothing adjustment |
| **DC drift removal** | 0.3 Hz high-pass on envelope |
| **Vessel presets** | fishing, cargo, tanker, small_craft |
| **Dark theme** | Professional publication-ready plots |

### CLI Usage

```bash
# PSO optimization (best quality)
python demon_visualizer.py clip.npy --pso

# Vessel preset
python demon_visualizer.py clip.npy --vessel fishing

# Manual bandpass
python demon_visualizer.py clip.npy --bandpass 1400 2200

# With smoothing
python demon_visualizer.py clip.npy --pso --smooth 5

# Save without display
python demon_visualizer.py clip.npy --pso -o plot.png --no-show
```

### Key Parameters

```python
# Style
STYLE = {
    'bg_color': '#0a1628',
    'accent_color': '#00b4d8',
    'marker_color': '#ff6b35',  # BR markers (orange)
    'spectrum_lw': 0.8,          # Thin line
}

# Processing
hp_cutoff = 0.3        # High-pass to remove DC drift
zero_pad_factor = 4    # FFT resolution
```

---

## Fixes Applied in This Session

### 1. DC Drift / Swell Leakage
**Problem:** 0.10 Hz peak appearing in spectrum  
**Fix:** Added 0.3 Hz high-pass filter after envelope extraction

```python
if hp_cutoff > 0:
    b_hp, a_hp = scipy_signal.butter(2, hp_cutoff / nyq, btype='high')
    envelope_ac = scipy_signal.filtfilt(b_hp, a_hp, envelope_ac)
```

### 2. Smoothing Reducing Peak Amplitude
**Problem:** Moving average pulled peaks down  
**Fix:** Re-normalize after smoothing

```python
def get_smoothed_db(smooth_val):
    if smooth_val > 1:
        smoothed = uniform_filter1d(spectrum_db, size=int(smooth_val))
        smoothed = smoothed - np.max(smoothed)  # Re-normalize!
        return smoothed
    return spectrum_db.copy()
```

### 3. Line Too Thick
**Problem:** Spectrum trace was 2.5 px  
**Fix:** Reduced to 0.8 px

### 4. Interactive Slider Added
- Smoothing range: 1-25
- Updates spectrum, fill, and label positions in real-time
- Disabled when saving to file (batch mode)

---

## Related: 2 Hz Swell Bug in Ship Noise Generator

**Problem in `ship_noise.py`:**
```python
# BUGGY: 0.5 Hz swell creates 2 Hz artifact in DEMON range
swell_mod = 1.0 + 0.3 * np.sin(2 * np.pi * 0.5 * t_center + swell_phase)
```

**Fixed in `ship_noise_corrected.py`:**
```python
# CORRECT: Real ocean swell 0.05-0.15 Hz (7-20 sec period)
swell_freq = rng.uniform(0.05, 0.15)
swell_mod = 1.0 + 0.2 * np.sin(2 * np.pi * swell_freq * t_center + swell_phase)
```

This fix is in the synthetic data generator, not the DEMON tool itself.

---

## Pending Improvements (Priority List)

### Priority 1: Artifact Checker
Build `demon_artifact_checker.py` to:
- Flag 2 Hz peaks (swell artifact)
- Flag 50/60 Hz peaks (generator noise)
- Report artifact/signal ratio
- Auto-detect non-matching harmonic series

### Priority 2: Batch Processing Mode
- Process folder of .npy files
- Generate summary CSV with detected BR/SR/n
- Flag clips with low confidence or artifacts

### Priority 3: Additional Visualizations
- Spectrogram + DEMON side-by-side
- Time-frequency envelope plot
- Harmonic series overlay

---

## File Locations

| File | Description |
|------|-------------|
| `demon_visualizer.py` | Main tool (outputs folder) |
| `ship_noise_corrected.py` | Fixed ship noise generator |
| `SWELL_MODULATION_FIX.md` | Documentation of 2 Hz bug |

---

## Test Data

### Validated Synthetic Clips
| Clip | True SR | True BR | n | Cavitation Band |
|------|---------|---------|---|-----------------|
| test_30s_fishing_v2.npy | 5.50 Hz | 22.00 Hz | 4 | 1400-2200 Hz |
| test_30s_cargo_v2.npy | 2.00 Hz | 10.00 Hz | 5 | 400-1200 Hz |

### Vessel Presets
```python
VESSEL_PRESETS = {
    'fishing':  (1400, 2200),  # Small fast props
    'cargo':    (400, 1200),   # Large slow props
    'tanker':   (300, 1000),   # Very large slow
    'small':    (1500, 3000),  # Outboards
    'default':  (500, 1500),
}
```

---

## Architecture Overview

```
demon_visualizer.py
├── compute_demon_spectrum()    # Core DEMON processing
├── find_peaks_in_demon()       # Peak detection
├── detect_blade_rate()         # Auto BR/SR/n detection
├── auto_detect_cavitation_band()  # PSD-based band detection
├── pso_optimize_bandpass()     # PSO optimization
├── demon_quality_score()       # Objective function for PSO
├── plot_demon_spectrum()       # Visualization with slider
└── main()                      # CLI entry point
```

---

## Quick Test

```bash
# Generate test signal and analyze
python -c "
import numpy as np
from demon_visualizer import compute_demon_spectrum, detect_blade_rate

signal = np.load('test_clip.npy')
freqs, spectrum_db = compute_demon_spectrum(signal, fs=16000)
result = detect_blade_rate(freqs, spectrum_db)
print(f'BR: {result[\"blade_rate\"]} Hz')
print(f'SR: {result[\"shaft_rate\"]} Hz')
print(f'Blades: {result[\"n_blades\"]}')
"
```

---

## Next Steps

1. Download `demon_visualizer.py` from this chat
2. Start fresh chat with this brief
3. Continue with artifact checker or other improvements

---

## Contact Files

All in `/mnt/user-data/outputs/`:
- `demon_visualizer.py`
- `SKANN_SSL_HANDOVER_BRIEF.md` (for training chat)
- `DEMON_TOOL_HANDOVER_BRIEF.md` (this file)
