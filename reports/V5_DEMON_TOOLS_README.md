# SKANN-SSL V5 Dataset - DEMON Tool Updates

## V5 Physics Changes Summary

The V5 dataset has significantly different acoustic parameters compared to the previous version:

### Shaft Rate / Blade Rate Changes

| Vessel Class | Old SR | V5 SR | Old BR | V5 BR | Impact |
|--------------|--------|-------|--------|-------|--------|
| **small_craft** | 5-12 Hz | **15-30 Hz** | 15-60 Hz | **45-150 Hz** | ⚠️ BR exceeds 50 Hz display |
| fishing_vessel | 3-8 Hz | 4-8 Hz | 9-40 Hz | 12-40 Hz | Minor change |
| cargo_ship | 1-3 Hz | 1.5-2.5 Hz | 3-15 Hz | 4.5-12.5 Hz | Narrower range |
| tanker | 0.8-2.5 Hz | 1-1.5 Hz | 2.4-12.5 Hz | 3-7.5 Hz | Narrower range |

### Cavitation Peak Frequencies (Now Fixed per Class)

| Vessel Class | Old Range | V5 Fixed Value |
|--------------|-----------|----------------|
| small_craft | 2324-2522 Hz | **5000 Hz** |
| fishing_vessel | 2020-2334 Hz | **1500 Hz** |
| cargo_ship | 885-1089 Hz | **600 Hz** |
| tanker | 628-812 Hz | **400 Hz** |

## Critical: 96.2% of small_craft have BR > 50 Hz

The standard DEMON display (0-50 Hz) will NOT show blade rates for most small_craft!
The updated tools use class-specific frequency ranges.

---

## Detection Modes

### OPTIMAL MODE (default)
Uses class-specific bandpass and frequency ranges. This represents the **ceiling performance** with perfect prior knowledge of vessel type.

```bash
python demon_batch_validator_v5.py --manifest manifest.csv --waveforms waveforms/
```

### BLIND MODE (`--blind`)
Uses PSO optimization to find the best bandpass **without knowing vessel class**. This simulates **real-world deployment** where vessel type is unknown.

```bash
python demon_batch_validator_v5.py --manifest manifest.csv --waveforms waveforms/ --blind
```

### COMPARISON MODE (`--compare`)
Runs BOTH modes and generates a comparison report showing how much accuracy degrades without class knowledge.

```bash
python demon_batch_validator_v5.py --manifest manifest.csv --waveforms waveforms/ --compare
```

---

## Updated Tool Settings

### OPTIMAL MODE: Class-Specific Settings

```python
# Bandpass (centered on fixed cav_peak)
VESSEL_BANDS = {
    'small_craft':    (4000, 6000),  # cav_peak = 5000 Hz
    'fishing_vessel': (1000, 2000),  # cav_peak = 1500 Hz
    'cargo_ship':     (400, 800),    # cav_peak = 600 Hz
    'tanker':         (250, 550),    # cav_peak = 400 Hz
}

# Max frequency for DEMON display
VESSEL_MAX_FREQ = {
    'small_craft':    200,  # BR up to 150 Hz
    'fishing_vessel':  60,  # BR up to 40 Hz
    'cargo_ship':      30,  # BR up to 12.5 Hz
    'tanker':          20,  # BR up to 7.5 Hz
}
```

### BLIND MODE: PSO Search Settings

```python
# Wide bandpass search range
BLIND_BANDPASS_SEARCH = {
    'min_center': 400,
    'max_center': 5000,
    'min_bandwidth': 400,
    'max_bandwidth': 2000,
}

# Wide frequency range for BR/SR detection
BLIND_MAX_FREQ = 200   # Covers all vessel types
BLIND_MIN_BR = 1.5
BLIND_MAX_BR = 180     # small_craft can hit 150 Hz
```

---

## Files Provided

| File | Description |
|------|-------------|
| `demon_batch_validator_v5.py` | Batch validation with optimal/blind/compare modes |
| `demon_visualizer_v5.py` | Interactive DEMON visualizer with V5 presets |
| `V5_DEMON_TOOLS_README.md` | This documentation |

---

## Usage Examples

### Quick Test (100 clips)
```bash
# Optimal mode
python demon_batch_validator_v5.py -m manifest.csv -w waveforms/ --max-clips 100

# Blind mode
python demon_batch_validator_v5.py -m manifest.csv -w waveforms/ --max-clips 100 --blind

# Compare both
python demon_batch_validator_v5.py -m manifest.csv -w waveforms/ --max-clips 100 --compare
```

### Full Validation (12,000 clips)
```bash
# Full comparison (will take ~30-40 min due to PSO in blind mode)
python demon_batch_validator_v5.py -m manifest.csv -w waveforms/ --compare

# Optimal only (~15 min)
python demon_batch_validator_v5.py -m manifest.csv -w waveforms/

# With sample plots
python demon_batch_validator_v5.py -m manifest.csv -w waveforms/ --plots 180
```

---

## Output Structure

### Single Mode (optimal or blind)
```
demon_validation/
├── demon_validation_results_optimal.csv  # or _blind.csv
├── validation_summary_optimal.md         # or _blind.md
└── plots/
    ├── stratified/
    └── failures/
```

### Comparison Mode
```
demon_validation/
├── results_optimal.csv
├── results_blind.csv
├── comparison_report.md      # ← Key output!
└── plots/
    ├── stratified/
    └── failures/
```

---

## Validation Metrics

| Metric | Tolerance | Description |
|--------|-----------|-------------|
| SR Accuracy | ±0.3 Hz | Shaft rate must be within tolerance |
| BR Accuracy | ±1.0 Hz | Blade rate must be within tolerance |
| n_blades | Exact | Inferred blade count must match |
| 2 Hz Artifact | > -15 dB | Flags residual swell bug |

---

## Expected Results

### Optimal Mode
- High accuracy for cavitating clips (ceiling performance)
- Accuracy depends on cavitation intensity

### Blind Mode
- Some degradation expected (PSO may not find optimal band)
- Worst for small_craft (very different cav_peak from others)
- Best for cargo_ship/tanker (similar frequency ranges)

### Comparison Insights
- If blind ≈ optimal: Tool is robust, class knowledge not critical
- If blind << optimal: Consider multi-band analysis or classifier pre-step

---

## Troubleshooting

### "No peaks detected" for small_craft
- In optimal mode: Should work with max_freq=200
- In blind mode: PSO may not search high enough frequencies

### Blind mode very slow
- PSO runs 12 particles × 20 iterations per clip
- Use `--max-clips 100` for quick tests
- Full blind validation of 12K clips takes ~30-40 min

### pyswarm not installed
- `pip install pyswarm`
- Falls back to grid search if not available (slower, less accurate)
