#!/usr/bin/env python3
"""
DEMON Batch Validator - Dataset-Wide Detection Accuracy Testing
================================================================

Tests DEMON detection accuracy against ground truth manifest for
the SKANN-SSL synthetic dataset.

Outputs:
- demon_validation_results.csv: Per-clip detection results
- validation_summary.md: Accuracy statistics by class
- plots/stratified/: Representative sample plots
- plots/failures/: Misdetection plots for debugging

Usage:
    python demon_batch_validator.py --manifest path/to/manifest.csv --waveforms path/to/waveforms/
    python demon_batch_validator.py --manifest manifest.csv --waveforms ./waveforms/ --plots 200

Author: SKANN-SSL Project
Date: January 2026
"""

import argparse
import sys
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from scipy.fft import fft, fftfreq
from tqdm import tqdm

# Optional plotting imports
try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# =============================================================================
# CONFIGURATION
# =============================================================================

# Detection tolerances
SR_TOLERANCE_HZ = 0.3      # Shaft rate must be within ±0.3 Hz
BR_TOLERANCE_HZ = 1.0      # Blade rate must be within ±1.0 Hz
ARTIFACT_2HZ_THRESHOLD_DB = -15  # 2 Hz peak above this = artifact

# Vessel-specific cavitation bands - UPDATED FOR V5 DATASET
# V5 has fixed cavitation peak frequencies per class
VESSEL_BANDS = {
    'small_craft': (4000, 6000),     # V5: cav_peak = 5000 Hz
    'fishing_vessel': (1000, 2000),  # V5: cav_peak = 1500 Hz
    'cargo_ship': (400, 800),        # V5: cav_peak = 600 Hz
    'tanker': (250, 550),            # V5: cav_peak = 400 Hz
    'no_vessel': (500, 1500),        # Default, won't be used
}

# V5 has different BR ranges per class - need class-specific max_freq for DEMON
VESSEL_MAX_FREQ = {
    'small_craft': 200,      # BR up to 150 Hz, need to see harmonics
    'fishing_vessel': 60,    # BR up to 40 Hz
    'cargo_ship': 30,        # BR up to 12.5 Hz
    'tanker': 20,            # BR up to 7.5 Hz
    'no_vessel': 50,         # Default
}

# =============================================================================
# BLIND MODE CONFIGURATION (no class knowledge)
# =============================================================================

# Wide bandpass search ranges for blind PSO optimization
BLIND_BANDPASS_SEARCH = {
    'min_center': 400,       # Min center frequency to search
    'max_center': 5000,      # Max center frequency to search
    'min_bandwidth': 400,    # Min bandwidth
    'max_bandwidth': 2000,   # Max bandwidth
}

# Wide frequency range for blind BR/SR detection
BLIND_MAX_FREQ = 200         # Search up to 200 Hz (covers all vessel types)
BLIND_MIN_BR = 1.5           # Min BR to consider
BLIND_MAX_BR = 180           # Max BR to consider (small_craft can hit 150 Hz)

# Plot styling
STYLE = {
    'bg_color': '#0a1628',
    'text_color': '#caf0f8',
    'accent_color': '#00b4d8',
    'marker_color': '#ff6b35',
    'sr_color': '#2ecc71',
    'grid_color': '#234567',
}


# =============================================================================
# DEMON ANALYSIS CORE
# =============================================================================

def compute_demon_spectrum(signal, fs=16000, bandpass_low=500, bandpass_high=1500,
                           max_freq=50, zero_pad_factor=4, hp_cutoff=0.3):
    """Compute DEMON spectrum from audio signal."""
    nyq = fs / 2
    if bandpass_high >= nyq:
        bandpass_high = int(nyq * 0.95)
    
    # Bandpass filter
    b, a = scipy_signal.butter(6, [bandpass_low/nyq, bandpass_high/nyq], btype='band')
    filtered = scipy_signal.filtfilt(b, a, signal)
    
    # Hilbert transform and envelope
    analytic_signal = scipy_signal.hilbert(filtered)
    envelope = np.abs(analytic_signal)
    
    # AC coupling + high-pass to remove DC drift
    envelope_ac = envelope - np.mean(envelope)
    if hp_cutoff > 0:
        b_hp, a_hp = scipy_signal.butter(2, hp_cutoff / nyq, btype='high')
        envelope_ac = scipy_signal.filtfilt(b_hp, a_hp, envelope_ac)
    
    # Window and FFT
    window = np.hanning(len(envelope_ac))
    envelope_windowed = envelope_ac * window
    
    n_fft = len(envelope_windowed) * zero_pad_factor
    freqs = fftfreq(n_fft, 1/fs)
    spectrum = np.abs(fft(envelope_windowed, n=n_fft)) / len(envelope_windowed)
    
    # Extract positive frequencies
    mask = (freqs >= 0) & (freqs <= max_freq)
    demon_freqs = freqs[mask]
    demon_spectrum = spectrum[mask]
    
    # Convert to dB (normalized)
    max_val = np.max(demon_spectrum)
    if max_val > 0:
        demon_db = 20 * np.log10(demon_spectrum / max_val + 1e-10)
    else:
        demon_db = np.zeros_like(demon_spectrum)
    
    return demon_freqs, demon_db


def find_peaks_in_demon(freqs, spectrum_db, threshold_db=-25, min_distance_hz=0.8):
    """Find spectral peaks in DEMON spectrum."""
    freq_resolution = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    min_distance_samples = max(1, int(min_distance_hz / freq_resolution))
    
    peak_indices, _ = scipy_signal.find_peaks(
        spectrum_db,
        height=threshold_db,
        distance=min_distance_samples
    )
    
    return freqs[peak_indices], spectrum_db[peak_indices]


def detect_blade_rate(freqs, spectrum_db, min_br=2.0, max_br=45.0):
    """Detect BR, SR, and blade count from DEMON spectrum."""
    peak_freqs, peak_amps = find_peaks_in_demon(freqs, spectrum_db)
    
    if len(peak_freqs) == 0:
        return None, None, None, 0.0
    
    # Sort by amplitude
    sorted_idx = np.argsort(peak_amps)[::-1]
    peak_freqs = peak_freqs[sorted_idx]
    peak_amps = peak_amps[sorted_idx]
    
    best_br, best_sr, best_n = None, None, None
    best_score = 0
    
    for freq, amp in zip(peak_freqs[:10], peak_amps[:10]):
        if not (min_br <= freq <= max_br):
            continue
        
        # Check for harmonics
        harmonic_score = 0
        for h in range(2, 5):
            expected = freq * h
            for pf in peak_freqs[:15]:
                if abs(pf - expected) < 1.0:
                    harmonic_score += 1.0 / h
                    break
        
        # Check for sub-harmonics (SR)
        for n in range(3, 7):
            expected_sr = freq / n
            if expected_sr < 0.5:
                continue
            for pf in peak_freqs[:15]:
                if abs(pf - expected_sr) < 0.5:
                    score = harmonic_score + 0.5
                    if score > best_score:
                        best_score = score
                        best_br = freq
                        best_sr = pf
                        best_n = n
                    break
        
        if harmonic_score > best_score and best_br is None:
            best_score = harmonic_score
            best_br = freq
    
    if best_br is None:
        for freq in peak_freqs:
            if min_br <= freq <= max_br:
                best_br = freq
                break
    
    confidence = min(1.0, best_score / 2.0) if best_score > 0 else 0.3
    return best_br, best_sr, best_n, confidence


# =============================================================================
# BLIND MODE: PSO OPTIMIZATION (no class knowledge)
# =============================================================================

def demon_quality_score(signal, fs, bandpass_low, bandpass_high, max_freq=BLIND_MAX_FREQ):
    """
    Compute quality score for DEMON spectrum with given bandpass.
    Higher score = clearer harmonics, better SNR.
    Used by blind mode PSO to find optimal bandpass without class knowledge.
    """
    try:
        freqs, spectrum_db = compute_demon_spectrum(
            signal, fs, 
            bandpass_low=int(bandpass_low), 
            bandpass_high=int(bandpass_high),
            max_freq=max_freq
        )
        
        peak_freqs, peak_amps = find_peaks_in_demon(freqs, spectrum_db, threshold_db=-30)
        
        if len(peak_freqs) < 2:
            return -100
        
        # Sort by amplitude
        sorted_idx = np.argsort(peak_amps)[::-1]
        peak_freqs = peak_freqs[sorted_idx]
        peak_amps = peak_amps[sorted_idx]
        
        # Find strongest peak as BR candidate
        br_candidate = None
        br_amp = -100
        for freq, amp in zip(peak_freqs[:10], peak_amps[:10]):
            if BLIND_MIN_BR <= freq <= BLIND_MAX_BR:
                br_candidate = freq
                br_amp = amp
                break
        
        if br_candidate is None:
            return -100
        
        score = 0.0
        
        # 1. Peak prominence
        noise_floor = np.median(spectrum_db)
        prominence = br_amp - noise_floor
        score += prominence * 2
        
        # 2. Harmonic consistency
        for h in range(2, 5):
            expected = br_candidate * h
            if expected > max_freq:
                break
            for pf in peak_freqs[:10]:
                if abs(pf - expected) < 1.5:
                    score += (5.0 / h)
                    break
        
        # 3. Sub-harmonic (SR) presence
        for n in range(3, 7):
            expected_sr = br_candidate / n
            if expected_sr < 1.0:
                continue
            for pf in peak_freqs[:10]:
                if abs(pf - expected_sr) < 0.5:
                    score += 3.0
                    break
        
        # 4. Noise penalty
        noise_std = np.std(spectrum_db[spectrum_db < noise_floor + 5])
        score -= noise_std * 0.5
        
        return score
        
    except Exception:
        return -100


def blind_pso_optimize(signal, fs, n_particles=12, max_iter=20):
    """
    PSO optimization to find optimal bandpass without knowing vessel class.
    Searches wide frequency range to find best cavitation band.
    
    Returns: (bandpass_low, bandpass_high, score)
    """
    try:
        from pyswarm import pso
        use_pso = True
    except ImportError:
        use_pso = False
    
    nyquist = fs / 2
    
    def objective(params):
        center_freq, bandwidth = params
        bandpass_low = center_freq - bandwidth / 2
        bandpass_high = center_freq + bandwidth / 2
        
        if bandpass_low < 100 or bandpass_high > nyquist - 100:
            return 1000
        if bandpass_high <= bandpass_low + 200:
            return 1000
        
        score = demon_quality_score(signal, fs, bandpass_low, bandpass_high)
        return -score  # Negate for minimization
    
    if use_pso:
        # PSO bounds: [center_freq, bandwidth]
        lb = [BLIND_BANDPASS_SEARCH['min_center'], BLIND_BANDPASS_SEARCH['min_bandwidth']]
        ub = [min(BLIND_BANDPASS_SEARCH['max_center'], nyquist - 500), 
              BLIND_BANDPASS_SEARCH['max_bandwidth']]
        
        best_params, best_score = pso(
            objective, lb, ub,
            swarmsize=n_particles,
            maxiter=max_iter,
            debug=False
        )
        
        center_freq, bandwidth = best_params
    else:
        # Fallback: Grid search
        best_score = 1000
        best_params = (1000, 800)
        
        for center in np.linspace(500, 4000, 8):
            for bw in [600, 1000, 1500]:
                bandpass_low = center - bw/2
                bandpass_high = center + bw/2
                if bandpass_low < 100 or bandpass_high > nyquist - 100:
                    continue
                score = objective([center, bw])
                if score < best_score:
                    best_score = score
                    best_params = (center, bw)
        
        center_freq, bandwidth = best_params
    
    bandpass_low = int(center_freq - bandwidth / 2)
    bandpass_high = int(center_freq + bandwidth / 2)
    
    return bandpass_low, bandpass_high, -best_score


def check_2hz_artifact(freqs, spectrum_db, threshold_db=-15):
    """Check for 2 Hz swell artifact from buggy ship noise generator."""
    # Look for peak near 2 Hz
    mask = (freqs >= 1.7) & (freqs <= 2.3)
    if not np.any(mask):
        return False, -100.0
    
    local_db = spectrum_db[mask]
    peak_amp = np.max(local_db)
    
    return peak_amp > threshold_db, peak_amp


# =============================================================================
# VALIDATION RESULT STRUCTURE
# =============================================================================

@dataclass
class ValidationResult:
    """Result of DEMON validation for a single clip."""
    clip_id: int
    filename: str
    vessel_class: str
    
    # Ground truth
    true_sr: float
    true_br: float
    true_n_blades: int
    has_cavitation: bool
    cavitation_intensity: float
    sea_state: int
    
    # Detection results
    detected_sr: Optional[float]
    detected_br: Optional[float]
    detected_n_blades: Optional[int]
    confidence: float
    
    # Accuracy flags
    sr_correct: bool
    br_correct: bool
    n_blades_correct: bool
    
    # Errors
    sr_error_hz: float
    br_error_hz: float
    
    # Artifacts
    has_2hz_artifact: bool
    artifact_2hz_db: float
    
    # Metadata
    bandpass_low: int
    bandpass_high: int
    max_freq: int  # V5: class-specific max frequency
    blind_mode: bool = False  # Was this detected without class knowledge?


# =============================================================================
# BATCH VALIDATION
# =============================================================================

def validate_clip(row: pd.Series, waveform_dir: Path, fs: int = 16000, 
                  blind_mode: bool = False) -> ValidationResult:
    """
    Validate DEMON detection for a single clip against ground truth.
    
    Parameters
    ----------
    row : pd.Series
        Row from manifest with ground truth
    waveform_dir : Path
        Directory containing .npy waveforms
    fs : int
        Sample rate
    blind_mode : bool
        If True, use PSO optimization without class knowledge (real-world scenario)
        If False, use class-specific optimal settings (ceiling performance)
    """
    
    # Load waveform
    filename = row['filename']
    filepath = waveform_dir / filename
    
    if not filepath.exists():
        # Return empty result with error flag
        return ValidationResult(
            clip_id=row['clip_id'],
            filename=filename,
            vessel_class=row['vessel_class'],
            true_sr=row['shaft_rate'],
            true_br=row['blade_pass_freq'],
            true_n_blades=row['n_blades'],
            has_cavitation=row['has_cavitation'],
            cavitation_intensity=row['cavitation_intensity'],
            sea_state=row['sea_state'],
            detected_sr=None,
            detected_br=None,
            detected_n_blades=None,
            confidence=0.0,
            sr_correct=False,
            br_correct=False,
            n_blades_correct=False,
            sr_error_hz=999.0,
            br_error_hz=999.0,
            has_2hz_artifact=False,
            artifact_2hz_db=-100.0,
            bandpass_low=0,
            bandpass_high=0,
            max_freq=0,
            blind_mode=blind_mode,
        )
    
    signal = np.load(filepath)
    vessel_class = row['vessel_class']
    
    if blind_mode:
        # BLIND MODE: Use PSO to find optimal bandpass without class knowledge
        bandpass_low, bandpass_high, pso_score = blind_pso_optimize(signal, fs)
        max_freq = BLIND_MAX_FREQ  # Wide range to catch all vessel types
    else:
        # OPTIMAL MODE: Use class-specific settings (ceiling performance)
        bandpass_low, bandpass_high = VESSEL_BANDS.get(vessel_class, (500, 1500))
        max_freq = VESSEL_MAX_FREQ.get(vessel_class, 50)
    
    # Compute DEMON spectrum
    freqs, spectrum_db = compute_demon_spectrum(
        signal, fs, bandpass_low, bandpass_high, max_freq=max_freq
    )
    
    # Detect BR/SR
    if blind_mode:
        # Wide search range for blind detection
        detected_br, detected_sr, detected_n, confidence = detect_blade_rate(
            freqs, spectrum_db, min_br=BLIND_MIN_BR, max_br=BLIND_MAX_BR
        )
    else:
        # Class-specific BR range
        max_br = max_freq * 0.9
        detected_br, detected_sr, detected_n, confidence = detect_blade_rate(
            freqs, spectrum_db, min_br=2.0, max_br=max_br
        )
    
    # Check for 2 Hz artifact
    has_artifact, artifact_db = check_2hz_artifact(freqs, spectrum_db)
    
    # Ground truth
    true_sr = row['shaft_rate']
    true_br = row['blade_pass_freq']
    true_n = row['n_blades']
    
    # Calculate errors
    if detected_sr is not None and true_sr > 0:
        sr_error = abs(detected_sr - true_sr)
        sr_correct = sr_error <= SR_TOLERANCE_HZ
    else:
        sr_error = 999.0 if true_sr > 0 else 0.0
        sr_correct = (true_sr == 0)  # Correct if no_vessel
    
    if detected_br is not None and true_br > 0:
        br_error = abs(detected_br - true_br)
        br_correct = br_error <= BR_TOLERANCE_HZ
    else:
        br_error = 999.0 if true_br > 0 else 0.0
        br_correct = (true_br == 0)
    
    n_correct = (detected_n == true_n) if detected_n is not None else (true_n == 0)
    
    return ValidationResult(
        clip_id=row['clip_id'],
        filename=filename,
        vessel_class=vessel_class,
        true_sr=true_sr,
        true_br=true_br,
        true_n_blades=true_n,
        has_cavitation=row['has_cavitation'],
        cavitation_intensity=row['cavitation_intensity'],
        sea_state=row['sea_state'],
        detected_sr=detected_sr,
        detected_br=detected_br,
        detected_n_blades=detected_n,
        confidence=confidence,
        sr_correct=sr_correct,
        br_correct=br_correct,
        n_blades_correct=n_correct,
        sr_error_hz=sr_error,
        br_error_hz=br_error,
        has_2hz_artifact=has_artifact,
        artifact_2hz_db=artifact_db,
        bandpass_low=bandpass_low,
        bandpass_high=bandpass_high,
        max_freq=max_freq,
        blind_mode=blind_mode,
    )


def run_batch_validation(manifest_path: Path, waveform_dir: Path, 
                         fs: int = 16000, max_clips: Optional[int] = None,
                         blind_mode: bool = False,
                         clip_range: Optional[Tuple[int, Optional[int]]] = None) -> pd.DataFrame:
    """
    Run validation on all clips in manifest.
    
    Parameters
    ----------
    manifest_path : Path
        Path to CSV manifest with ground truth
    waveform_dir : Path
        Directory containing .npy waveforms
    fs : int
        Sample rate (default: 16000)
    max_clips : int, optional
        Limit number of clips (for testing)
    blind_mode : bool
        If True, use PSO optimization without class knowledge
        If False, use class-specific optimal settings
    clip_range : tuple, optional
        (start_clip_id, end_clip_id) - filter by clip_id range
        end_clip_id can be None to mean "to end"
    """
    
    # Load manifest
    df = pd.read_csv(manifest_path)
    
    # Apply clip_range filter first (by clip_id)
    if clip_range is not None:
        start_id, end_id = clip_range
        if end_id is not None:
            df = df[(df['clip_id'] >= start_id) & (df['clip_id'] <= end_id)]
        else:
            df = df[df['clip_id'] >= start_id]
        print(f"Filtered to clip_id range: {start_id} - {end_id if end_id else 'end'}")
    
    # Then apply max_clips limit
    if max_clips is not None:
        df = df.head(max_clips)
    
    mode_str = "BLIND MODE (PSO)" if blind_mode else "OPTIMAL MODE (class-specific)"
    print(f"Validating {len(df)} clips in {mode_str}...")
    
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="DEMON Validation"):
        result = validate_clip(row, waveform_dir, fs, blind_mode=blind_mode)
        results.append(asdict(result))
    
    return pd.DataFrame(results)


# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

def generate_summary(results_df: pd.DataFrame) -> str:
    """Generate markdown summary of validation results."""
    
    # Check if this is blind mode
    is_blind = results_df['blind_mode'].iloc[0] if 'blind_mode' in results_df.columns else False
    mode_str = "BLIND MODE (PSO, no class knowledge)" if is_blind else "OPTIMAL MODE (class-specific settings)"
    
    lines = [
        "# DEMON Validation Summary",
        "",
        f"**Total clips validated:** {len(results_df)}",
        f"**Mode:** {mode_str}",
        f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "---",
        "",
        "## Overall Accuracy",
        "",
    ]
    
    # Filter to clips with cavitation (DEMON needs cavitation to work)
    cav_df = results_df[results_df['has_cavitation'] == True]
    no_cav_df = results_df[results_df['has_cavitation'] == False]
    vessels_df = results_df[results_df['vessel_class'] != 'no_vessel']
    
    lines.append(f"### Clips WITH Cavitation (n={len(cav_df)})")
    lines.append("")
    
    if len(cav_df) > 0:
        sr_acc = cav_df['sr_correct'].mean() * 100
        br_acc = cav_df['br_correct'].mean() * 100
        n_acc = cav_df['n_blades_correct'].mean() * 100
        
        lines.append(f"| Metric | Accuracy |")
        lines.append(f"|--------|----------|")
        lines.append(f"| Shaft Rate (±{SR_TOLERANCE_HZ} Hz) | {sr_acc:.1f}% |")
        lines.append(f"| Blade Rate (±{BR_TOLERANCE_HZ} Hz) | {br_acc:.1f}% |")
        lines.append(f"| Blade Count (exact) | {n_acc:.1f}% |")
        lines.append("")
        
        # Mean errors
        sr_err = cav_df[cav_df['sr_error_hz'] < 100]['sr_error_hz'].mean()
        br_err = cav_df[cav_df['br_error_hz'] < 100]['br_error_hz'].mean()
        lines.append(f"**Mean SR error:** {sr_err:.3f} Hz")
        lines.append(f"**Mean BR error:** {br_err:.3f} Hz")
        lines.append("")
    
    lines.append(f"### Clips WITHOUT Cavitation (n={len(no_cav_df)})")
    lines.append("")
    lines.append("*Note: DEMON relies on cavitation modulation. Non-cavitating clips expected to have lower detection rates.*")
    lines.append("")
    
    # Breakdown by vessel class
    lines.append("---")
    lines.append("")
    lines.append("## Accuracy by Vessel Class")
    lines.append("")
    lines.append("| Vessel Class | n (cav) | SR Acc | BR Acc | n_blades Acc | Mean SR Err |")
    lines.append("|--------------|---------|--------|--------|--------------|-------------|")
    
    for vc in ['small_craft', 'fishing_vessel', 'cargo_ship', 'tanker']:
        vc_cav = cav_df[cav_df['vessel_class'] == vc]
        if len(vc_cav) > 0:
            sr_acc = vc_cav['sr_correct'].mean() * 100
            br_acc = vc_cav['br_correct'].mean() * 100
            n_acc = vc_cav['n_blades_correct'].mean() * 100
            sr_err = vc_cav[vc_cav['sr_error_hz'] < 100]['sr_error_hz'].mean()
            lines.append(f"| {vc} | {len(vc_cav)} | {sr_acc:.1f}% | {br_acc:.1f}% | {n_acc:.1f}% | {sr_err:.3f} Hz |")
    
    lines.append("")
    
    # Breakdown by blade count
    lines.append("---")
    lines.append("")
    lines.append("## Accuracy by Blade Count")
    lines.append("")
    lines.append("| n_blades | n (cav) | SR Acc | BR Acc | n_blades Acc |")
    lines.append("|----------|---------|--------|--------|--------------|")
    
    for n in [3, 4, 5]:
        n_cav = cav_df[cav_df['true_n_blades'] == n]
        if len(n_cav) > 0:
            sr_acc = n_cav['sr_correct'].mean() * 100
            br_acc = n_cav['br_correct'].mean() * 100
            n_acc = n_cav['n_blades_correct'].mean() * 100
            lines.append(f"| {n} | {len(n_cav)} | {sr_acc:.1f}% | {br_acc:.1f}% | {n_acc:.1f}% |")
    
    lines.append("")
    
    # Breakdown by sea state
    lines.append("---")
    lines.append("")
    lines.append("## Accuracy by Sea State")
    lines.append("")
    lines.append("| Sea State | n (cav) | SR Acc | BR Acc | Mean Confidence |")
    lines.append("|-----------|---------|--------|--------|-----------------|")
    
    for ss in sorted(cav_df['sea_state'].unique()):
        ss_cav = cav_df[cav_df['sea_state'] == ss]
        if len(ss_cav) > 0:
            sr_acc = ss_cav['sr_correct'].mean() * 100
            br_acc = ss_cav['br_correct'].mean() * 100
            conf = ss_cav['confidence'].mean()
            lines.append(f"| {ss} | {len(ss_cav)} | {sr_acc:.1f}% | {br_acc:.1f}% | {conf:.2f} |")
    
    lines.append("")
    
    # Breakdown by cavitation intensity
    lines.append("---")
    lines.append("")
    lines.append("## Accuracy by Cavitation Intensity")
    lines.append("")
    
    cav_df_copy = cav_df.copy()
    cav_df_copy['cav_bin'] = pd.cut(cav_df_copy['cavitation_intensity'], 
                                     bins=[0, 0.4, 0.7, 1.0],
                                     labels=['Low (0.33-0.4)', 'Med (0.4-0.7)', 'High (0.7-1.0)'])
    
    lines.append("| Intensity | n | SR Acc | BR Acc |")
    lines.append("|-----------|---|--------|--------|")
    
    for label in ['Low (0.33-0.4)', 'Med (0.4-0.7)', 'High (0.7-1.0)']:
        subset = cav_df_copy[cav_df_copy['cav_bin'] == label]
        if len(subset) > 0:
            sr_acc = subset['sr_correct'].mean() * 100
            br_acc = subset['br_correct'].mean() * 100
            lines.append(f"| {label} | {len(subset)} | {sr_acc:.1f}% | {br_acc:.1f}% |")
    
    lines.append("")
    
    # Artifact detection
    lines.append("---")
    lines.append("")
    lines.append("## Artifact Detection")
    lines.append("")
    
    n_artifacts = results_df['has_2hz_artifact'].sum()
    pct_artifacts = n_artifacts / len(results_df) * 100
    lines.append(f"**Clips with 2 Hz swell artifact:** {n_artifacts} ({pct_artifacts:.1f}%)")
    lines.append("")
    
    if n_artifacts > 0:
        artifact_clips = results_df[results_df['has_2hz_artifact']]
        lines.append("Affected vessel classes:")
        for vc in artifact_clips['vessel_class'].unique():
            n = len(artifact_clips[artifact_clips['vessel_class'] == vc])
            lines.append(f"  - {vc}: {n}")
    
    lines.append("")
    
    # Failure summary
    lines.append("---")
    lines.append("")
    lines.append("## Failures Summary")
    lines.append("")
    
    failures = cav_df[(cav_df['sr_correct'] == False) | (cav_df['br_correct'] == False)]
    lines.append(f"**Total misdetections (cavitating clips):** {len(failures)} ({len(failures)/len(cav_df)*100:.1f}%)")
    lines.append("")
    
    if len(failures) > 0:
        lines.append("Top failure modes:")
        
        # SR only wrong
        sr_only_wrong = failures[(failures['sr_correct'] == False) & (failures['br_correct'] == True)]
        lines.append(f"  - SR wrong, BR correct: {len(sr_only_wrong)}")
        
        # BR only wrong
        br_only_wrong = failures[(failures['sr_correct'] == True) & (failures['br_correct'] == False)]
        lines.append(f"  - SR correct, BR wrong: {len(br_only_wrong)}")
        
        # Both wrong
        both_wrong = failures[(failures['sr_correct'] == False) & (failures['br_correct'] == False)]
        lines.append(f"  - Both wrong: {len(both_wrong)}")
    
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Generated by demon_batch_validator.py*")
    
    return "\n".join(lines)


# =============================================================================
# PLOTTING
# =============================================================================

def plot_demon_result(result: dict, waveform_dir: Path, output_path: Path = None, 
                      fs: int = 16000, smooth_display: int = 1, 
                      interactive: bool = False, show: bool = False, **kwargs):
    """
    Generate DEMON plot for a single validation result.
    
    Parameters
    ----------
    result : dict
        Validation result dictionary
    waveform_dir : Path
        Directory containing waveforms
    output_path : Path, optional
        Path to save plot (if None, don't save)
    fs : int
        Sample rate
    smooth_display : int
        Smoothing window size (1 = no smoothing)
    interactive : bool
        If True and show=True, add interactive smoothing slider
    show : bool
        If True, display plot interactively
    """
    
    if not HAS_MATPLOTLIB:
        return
    
    # Load signal
    filepath = waveform_dir / result['filename']
    if not filepath.exists():
        return
    
    signal = np.load(filepath)
    
    # Get class-specific max_freq (V5 dataset)
    max_freq = result.get('max_freq', VESSEL_MAX_FREQ.get(result['vessel_class'], 50))
    
    # Compute spectrum with correct max_freq
    freqs, spectrum_db = compute_demon_spectrum(
        signal, fs, result['bandpass_low'], result['bandpass_high'], max_freq=max_freq
    )
    
    # Setup plot
    plt.rcParams.update({
        'figure.facecolor': STYLE['bg_color'],
        'axes.facecolor': STYLE['bg_color'],
        'axes.edgecolor': STYLE['text_color'],
        'axes.labelcolor': STYLE['text_color'],
        'text.color': STYLE['text_color'],
        'xtick.color': STYLE['text_color'],
        'ytick.color': STYLE['text_color'],
    })
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Apply initial display smoothing
    def get_smoothed_db(smooth_val):
        if smooth_val > 1:
            from scipy.ndimage import uniform_filter1d
            smoothed = uniform_filter1d(spectrum_db, size=int(smooth_val))
            # Re-normalize to preserve 0 dB max (smoothing reduces peak heights)
            smoothed = smoothed - np.max(smoothed)
            return smoothed
        return spectrum_db.copy()
    
    # Helper to get dB value at a specific frequency
    def get_db_at_freq(freq, db_array):
        idx = np.argmin(np.abs(freqs - freq))
        return db_array[idx]
    
    display_db = get_smoothed_db(smooth_display)
    
    # Plot spectrum with clean gradient fill
    line, = ax.plot(freqs, display_db, color=STYLE['accent_color'], linewidth=0.8, zorder=3)
    fill = ax.fill_between(freqs, display_db, -40, color='#0a4d6e', alpha=0.85, zorder=1)
    
    # Mark true BR harmonics - store text objects for slider updates
    br_texts = []
    true_br = result['true_br']
    if true_br > 0:
        for h in range(1, 5):
            hf = true_br * h
            if hf <= freqs[-1]:
                ax.axvline(hf, color=STYLE['marker_color'], linestyle='--', 
                          linewidth=1.0, alpha=0.7)
                txt = ax.text(hf, 3, f'{h}xBR\n({hf:.1f})', color=STYLE['marker_color'],
                       fontsize=8, ha='center', va='bottom', fontweight='bold')
                br_texts.append((txt, hf, h))
    
    # Mark true SR harmonics - store text objects for slider updates
    sr_texts = []
    true_sr = result['true_sr']
    if true_sr > 0:
        for h in range(1, 4):
            hf = true_sr * h
            if hf <= freqs[-1] and abs(hf - true_br) > 0.5:  # Don't overlap with BR
                ax.axvline(hf, color=STYLE['sr_color'], linestyle=':', 
                          linewidth=1.0, alpha=0.7)
                txt = ax.text(hf, 2, f'{h}xSR\n({hf:.1f})', color=STYLE['sr_color'],
                       fontsize=7, ha='center', va='bottom')
                sr_texts.append((txt, hf, h))
    
    # Mark detected values
    if result['detected_br'] is not None:
        ax.axvline(result['detected_br'], color='yellow', linestyle='-', 
                  linewidth=1.5, alpha=0.5, label=f"Det BR: {result['detected_br']:.2f}")
    
    if result['detected_sr'] is not None:
        ax.axvline(result['detected_sr'], color='lime', linestyle='-', 
                  linewidth=1.5, alpha=0.5, label=f"Det SR: {result['detected_sr']:.2f}")
    
    # Mark 2 Hz artifact if present
    if result['has_2hz_artifact']:
        ax.axvline(2.0, color='red', linestyle='-', linewidth=2, alpha=0.8)
        ax.text(2.0, -5, '2Hz\nARTIFACT', color='red', fontsize=9, 
               ha='center', fontweight='bold')
    
    # Configure axes
    ax.set_xlabel('Modulation Frequency (Hz)')
    ax.set_ylabel('Amplitude (dB re max)')
    ax.set_xlim(0, max_freq)
    ax.set_ylim(-40, 8)
    ax.grid(True, alpha=0.3, color=STYLE['grid_color'])
    # Adaptive x-axis ticks based on frequency range
    if max_freq <= 30:
        ax.xaxis.set_major_locator(MultipleLocator(5))
    elif max_freq <= 60:
        ax.xaxis.set_major_locator(MultipleLocator(10))
    else:
        ax.xaxis.set_major_locator(MultipleLocator(20))
    
    # Title with validation status
    status = "PASS" if result['sr_correct'] and result['br_correct'] else "FAIL"
    title = (f"{status} {result['filename']} | {result['vessel_class']} | "
             f"n={result['true_n_blades']} | SS={result['sea_state']} | "
             f"Cav={result['cavitation_intensity']:.2f}")
    ax.set_title(title, fontsize=11, fontweight='bold')
    
    # Info box
    det_sr_str = f"{result['detected_sr']:.2f}" if result['detected_sr'] is not None else 'N/A'
    det_br_str = f"{result['detected_br']:.2f}" if result['detected_br'] is not None else 'N/A'
    info_lines = [
        f"True: SR={result['true_sr']:.2f}, BR={result['true_br']:.2f}",
        f"Det:  SR={det_sr_str}, BR={det_br_str}",
        f"Err:  SR={result['sr_error_hz']:.2f}Hz, BR={result['br_error_hz']:.2f}Hz",
        f"Conf: {result['confidence']:.2f}"
    ]
    ax.text(0.98, 0.97, '\n'.join(info_lines), transform=ax.transAxes,
           fontsize=9, fontfamily='monospace', color='white',
           ha='right', va='top',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='#1a3a5c', 
                    edgecolor=STYLE['accent_color'], alpha=0.9))
    
    # Add interactive smoothing slider if enabled
    slider_ax = None
    smooth_slider = None
    if interactive and show:
        from matplotlib.widgets import Slider
        # Create slider axis
        slider_ax = plt.axes([0.15, 0.05, 0.70, 0.03], facecolor='#1a3a5c')
        smooth_slider = Slider(
            slider_ax, 'Smoothing', 
            valmin=1, valmax=25, valinit=max(1, smooth_display),
            valstep=1,
            color=STYLE['accent_color']
        )
        slider_ax.tick_params(colors=STYLE['text_color'])
        for txt in slider_ax.texts:
            txt.set_color(STYLE['text_color'])
        
        # Update function for slider
        def update_smoothing(val):
            nonlocal fill
            smooth_val = int(smooth_slider.val)
            new_db = get_smoothed_db(smooth_val)
            
            # Update line data
            line.set_ydata(new_db)
            
            # Update fill (need to remove and recreate)
            fill.remove()
            fill = ax.fill_between(freqs, new_db, -40, 
                                   color='#0a4d6e', alpha=0.85, zorder=1)
            
            # Update BR label positions
            for txt, freq, i in br_texts:
                new_y = min(get_db_at_freq(freq, new_db) + 4, 3)
                txt.set_position((freq, new_y))
            
            # Update SR label positions  
            for txt, freq, i in sr_texts:
                new_y = min(get_db_at_freq(freq, new_db) + 4, 3)
                txt.set_position((freq, new_y))
            
            fig.canvas.draw_idle()
        
        smooth_slider.on_changed(update_smoothing)
    
    plt.tight_layout()
    if interactive and show and slider_ax is not None:
        plt.subplots_adjust(bottom=0.15)  # Re-adjust after tight_layout
    
    # Save if output path specified
    if output_path is not None:
        fig.savefig(output_path, dpi=120, facecolor=STYLE['bg_color'], bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig, ax


def generate_stratified_plots(results_df: pd.DataFrame, waveform_dir: Path, 
                              output_dir: Path, n_per_stratum: int = 3, 
                              smooth_display: int = 1):
    """Generate stratified sample of plots."""
    
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not available, skipping plots")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Only plot cavitating clips
    cav_df = results_df[results_df['has_cavitation'] == True].copy()
    
    # Create strata: vessel_class × n_blades × cavitation_intensity_bin
    cav_df['cav_bin'] = pd.cut(cav_df['cavitation_intensity'], 
                                bins=[0, 0.5, 0.75, 1.0],
                                labels=['low', 'med', 'high'])
    
    plots_generated = 0
    
    for vc in ['small_craft', 'fishing_vessel', 'cargo_ship', 'tanker']:
        for n in [3, 4, 5]:
            for cav_bin in ['low', 'med', 'high']:
                stratum = cav_df[(cav_df['vessel_class'] == vc) & 
                                (cav_df['true_n_blades'] == n) &
                                (cav_df['cav_bin'] == cav_bin)]
                
                if len(stratum) == 0:
                    continue
                
                # Sample up to n_per_stratum
                sample = stratum.sample(min(n_per_stratum, len(stratum)), random_state=42)
                
                for _, row in sample.iterrows():
                    out_path = output_dir / f"{vc}_n{n}_cav{cav_bin}_{row['clip_id']:06d}.png"
                    plot_demon_result(row.to_dict(), waveform_dir, out_path, 
                                     smooth_display=smooth_display)
                    plots_generated += 1
    
    print(f"Generated {plots_generated} stratified sample plots")


def generate_failure_plots(results_df: pd.DataFrame, waveform_dir: Path, 
                           output_dir: Path, max_plots: int = 100,
                           smooth_display: int = 1):
    """Generate plots for failed detections."""
    
    if not HAS_MATPLOTLIB:
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get failures (cavitating clips only)
    cav_df = results_df[results_df['has_cavitation'] == True]
    failures = cav_df[(cav_df['sr_correct'] == False) | (cav_df['br_correct'] == False)]
    
    # Sort by error magnitude
    failures = failures.copy()
    failures['total_error'] = failures['sr_error_hz'] + failures['br_error_hz']
    failures = failures.sort_values('total_error', ascending=False)
    
    # Limit
    failures = failures.head(max_plots)
    
    plots_generated = 0
    for _, row in tqdm(failures.iterrows(), total=len(failures), desc="Failure plots"):
        out_path = output_dir / f"fail_{row['vessel_class']}_{row['clip_id']:06d}.png"
        plot_demon_result(row.to_dict(), waveform_dir, out_path, 
                         smooth_display=smooth_display)
        plots_generated += 1
    
    print(f"Generated {plots_generated} failure analysis plots")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='DEMON Batch Validator - Test detection accuracy against ground truth',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # OPTIMAL MODE - class-specific settings (ceiling performance)
  python demon_batch_validator.py --manifest manifest.csv --waveforms ./waveforms/
  
  # BLIND MODE - PSO optimization, no class knowledge (real-world)
  python demon_batch_validator.py --manifest manifest.csv --waveforms ./waveforms/ --blind
  
  # COMPARISON - run both modes and compare
  python demon_batch_validator.py --manifest manifest.csv --waveforms ./waveforms/ --compare
  
  # Process specific clip range (e.g., clips 891-1101)
  python demon_batch_validator.py --manifest manifest.csv --waveforms ./waveforms/ --clip-range 891-1101
  
  # Process from clip 500 onwards
  python demon_batch_validator.py --manifest manifest.csv --waveforms ./waveforms/ --clip-range 500-
  
  # With smoothed plots
  python demon_batch_validator.py --manifest manifest.csv --waveforms ./waveforms/ --plots 200 --smooth 5
  
  # View single clip interactively with smoothing slider
  python demon_batch_validator.py --manifest manifest.csv --waveforms ./waveforms/ --view-clip 1234

Detection Modes:
  (default)   OPTIMAL - Uses class-specific bandpass and frequency ranges
              This is the "ceiling" performance with perfect prior knowledge
              
  --blind     BLIND - Uses PSO to find optimal bandpass without class hints
              This simulates real-world deployment where vessel type is unknown
              
  --compare   Runs BOTH modes and generates comparison report

Interactive Mode:
  --view-clip CLIP_ID   Open interactive plot for a specific clip with smoothing slider
        """
    )
    
    parser.add_argument('--manifest', '-m', type=str, required=True,
                        help='Path to master_dataset_manifest.csv')
    parser.add_argument('--waveforms', '-w', type=str, required=True,
                        help='Path to waveforms directory')
    parser.add_argument('--output', '-o', type=str, default='demon_validation',
                        help='Output directory name (default: demon_validation)')
    parser.add_argument('--max-clips', type=int, default=None,
                        help='Limit number of clips to process (for testing)')
    parser.add_argument('--clip-range', type=str, default=None,
                        help='Process clip_id range, e.g. "891-1101" or "500-" for 500 onwards')
    parser.add_argument('--plots', type=int, default=0,
                        help='Number of stratified sample plots to generate (0=none)')
    parser.add_argument('--failure-plots', type=int, default=50,
                        help='Max failure plots to generate (default: 50)')
    parser.add_argument('--fs', type=int, default=16000,
                        help='Sample rate in Hz (default: 16000)')
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--blind', action='store_true',
                           help='Blind mode: PSO optimization without class knowledge (real-world scenario)')
    mode_group.add_argument('--compare', action='store_true',
                           help='Run both optimal and blind modes, generate comparison report')
    
    # Plot options
    parser.add_argument('--smooth', type=int, default=1,
                        help='Smoothing window size for plots (1=none, default: 1)')
    parser.add_argument('--view-clip', type=int, default=None,
                        help='View single clip interactively by clip_id (with smoothing slider)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    manifest_path = Path(args.manifest)
    waveform_dir = Path(args.waveforms)
    output_dir = Path(args.output)
    
    # Validate paths
    if not manifest_path.exists():
        print(f"Error: Manifest not found: {manifest_path}")
        return 1
    
    if not waveform_dir.exists():
        print(f"Error: Waveform directory not found: {waveform_dir}")
        return 1
    
    # Parse clip_range if provided
    clip_range = None
    if args.clip_range:
        try:
            if '-' in args.clip_range:
                parts = args.clip_range.split('-')
                start_id = int(parts[0])
                end_id = int(parts[1]) if parts[1] else None
                clip_range = (start_id, end_id)
            else:
                # Single clip id
                clip_id = int(args.clip_range)
                clip_range = (clip_id, clip_id)
        except ValueError:
            print(f"Error: Invalid clip-range format '{args.clip_range}'. Use '891-1101' or '500-'")
            return 1
    
    # Handle single clip interactive view
    if args.view_clip is not None:
        print(f"Interactive view for clip_id={args.view_clip}")
        df = pd.read_csv(manifest_path)
        clip_row = df[df['clip_id'] == args.view_clip]
        if len(clip_row) == 0:
            print(f"Error: clip_id {args.view_clip} not found in manifest")
            return 1
        
        row = clip_row.iloc[0]
        print(f"Vessel: {row['vessel_class']} | SR: {row['shaft_rate']:.2f} Hz | BR: {row['blade_pass_freq']:.2f} Hz")
        print(f"Cavitation: {row['has_cavitation']} (intensity: {row['cavitation_intensity']:.2f})")
        
        # Run validation on this clip
        result = validate_clip(row, waveform_dir, args.fs, blind_mode=args.blind)
        result_dict = asdict(result)
        
        print(f"\nDetection results:")
        print(f"  Detected SR: {result.detected_sr:.2f if result.detected_sr else 'N/A'} Hz")
        print(f"  Detected BR: {result.detected_br:.2f if result.detected_br else 'N/A'} Hz")
        print(f"  SR correct: {result.sr_correct} | BR correct: {result.br_correct}")
        
        # Show interactive plot
        plot_demon_result(result_dict, waveform_dir, output_path=None, fs=args.fs,
                         smooth_display=args.smooth, interactive=True, show=True)
        return 0
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("DEMON BATCH VALIDATOR")
    print("="*60)
    print(f"Manifest: {manifest_path}")
    print(f"Waveforms: {waveform_dir}")
    print(f"Output: {output_dir}")
    
    if clip_range:
        start_id, end_id = clip_range
        range_str = f"{start_id}-{end_id}" if end_id else f"{start_id}-end"
        print(f"Clip Range: {range_str}")
    
    if args.compare:
        print(f"Mode: COMPARISON (optimal vs blind)")
    elif args.blind:
        print(f"Mode: BLIND (PSO, no class knowledge)")
    else:
        print(f"Mode: OPTIMAL (class-specific settings)")
    print()
    
    if args.compare:
        # Run BOTH modes and compare
        print("\n" + "="*60)
        print("PHASE 1: OPTIMAL MODE (ceiling performance)")
        print("="*60)
        results_optimal = run_batch_validation(
            manifest_path, waveform_dir, args.fs, args.max_clips, 
            blind_mode=False, clip_range=clip_range
        )
        
        print("\n" + "="*60)
        print("PHASE 2: BLIND MODE (real-world performance)")
        print("="*60)
        results_blind = run_batch_validation(
            manifest_path, waveform_dir, args.fs, args.max_clips, 
            blind_mode=True, clip_range=clip_range
        )
        
        # Save both results
        results_optimal.to_csv(output_dir / 'results_optimal.csv', index=False)
        results_blind.to_csv(output_dir / 'results_blind.csv', index=False)
        
        # Generate comparison report
        comparison = generate_comparison_report(results_optimal, results_blind)
        comparison_path = output_dir / 'comparison_report.md'
        with open(comparison_path, 'w', encoding='utf-8') as f:
            f.write(comparison)
        print(f"\nSaved comparison report to: {comparison_path}")
        
        # Print comparison summary
        cav_opt = results_optimal[results_optimal['has_cavitation'] == True]
        cav_blind = results_blind[results_blind['has_cavitation'] == True]
        
        print("\n" + "="*60)
        print("COMPARISON SUMMARY (cavitating clips)")
        print("="*60)
        print(f"{'Metric':<25} {'Optimal':>12} {'Blind':>12} {'Diff':>10}")
        print("-"*60)
        
        opt_sr = cav_opt['sr_correct'].mean()*100
        blind_sr = cav_blind['sr_correct'].mean()*100
        print(f"{'SR Accuracy':<25} {opt_sr:>11.1f}% {blind_sr:>11.1f}% {blind_sr-opt_sr:>+9.1f}%")
        
        opt_br = cav_opt['br_correct'].mean()*100
        blind_br = cav_blind['br_correct'].mean()*100
        print(f"{'BR Accuracy':<25} {opt_br:>11.1f}% {blind_br:>11.1f}% {blind_br-opt_br:>+9.1f}%")
        
        opt_n = cav_opt['n_blades_correct'].mean()*100
        blind_n = cav_blind['n_blades_correct'].mean()*100
        print(f"{'Blade Count Accuracy':<25} {opt_n:>11.1f}% {blind_n:>11.1f}% {blind_n-opt_n:>+9.1f}%")
        
        results_df = results_optimal  # Use optimal for plots
        
    else:
        # Single mode
        blind_mode = args.blind
        results_df = run_batch_validation(
            manifest_path, waveform_dir, args.fs, args.max_clips, 
            blind_mode=blind_mode, clip_range=clip_range
        )
        
        # Save results CSV
        mode_suffix = '_blind' if blind_mode else '_optimal'
        results_csv = output_dir / f'demon_validation_results{mode_suffix}.csv'
        results_df.to_csv(results_csv, index=False)
        print(f"\nSaved results to: {results_csv}")
        
        # Generate summary
        summary = generate_summary(results_df)
        summary_path = output_dir / f'validation_summary{mode_suffix}.md'
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        print(f"Saved summary to: {summary_path}")
        
        # Print quick summary
        cav_df = results_df[results_df['has_cavitation'] == True]
        if len(cav_df) > 0:
            mode_label = "BLIND" if blind_mode else "OPTIMAL"
            print("\n" + "="*60)
            print(f"QUICK SUMMARY - {mode_label} MODE (cavitating clips only)")
            print("="*60)
            print(f"SR Accuracy: {cav_df['sr_correct'].mean()*100:.1f}%")
            print(f"BR Accuracy: {cav_df['br_correct'].mean()*100:.1f}%")
            print(f"Blade Count Accuracy: {cav_df['n_blades_correct'].mean()*100:.1f}%")
            print(f"2 Hz Artifacts: {results_df['has_2hz_artifact'].sum()} clips")
    
    # Generate plots if requested
    if args.plots > 0:
        print(f"\nGenerating stratified sample plots...")
        plots_dir = output_dir / 'plots' / 'stratified'
        n_per_stratum = max(1, args.plots // (4 * 3 * 3))  # 4 vessels × 3 blades × 3 cav levels
        generate_stratified_plots(results_df, waveform_dir, plots_dir, n_per_stratum,
                                 smooth_display=args.smooth)
    
    # Generate failure plots
    if args.failure_plots > 0:
        failures = results_df[(results_df['has_cavitation']) & 
                             ((results_df['sr_correct'] == False) | 
                              (results_df['br_correct'] == False))]
        if len(failures) > 0:
            print(f"\nGenerating failure analysis plots...")
            fail_dir = output_dir / 'plots' / 'failures'
            generate_failure_plots(results_df, waveform_dir, fail_dir, args.failure_plots,
                                  smooth_display=args.smooth)
    
    print("\nDone!")
    return 0


def generate_comparison_report(results_optimal: pd.DataFrame, results_blind: pd.DataFrame) -> str:
    """Generate markdown comparison report between optimal and blind modes."""
    
    lines = [
        "# DEMON Validation: Optimal vs Blind Mode Comparison",
        "",
        f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Total clips:** {len(results_optimal)}",
        "",
        "## Mode Descriptions",
        "",
        "| Mode | Description |",
        "|------|-------------|",
        "| **Optimal** | Class-specific bandpass and frequency ranges (ceiling performance) |",
        "| **Blind** | PSO optimization without class knowledge (real-world scenario) |",
        "",
        "---",
        "",
        "## Overall Accuracy Comparison",
        "",
    ]
    
    # Filter to cavitating clips
    cav_opt = results_optimal[results_optimal['has_cavitation'] == True]
    cav_blind = results_blind[results_blind['has_cavitation'] == True]
    
    lines.append(f"*Cavitating clips only (n={len(cav_opt)})*")
    lines.append("")
    lines.append("| Metric | Optimal | Blind | Difference |")
    lines.append("|--------|---------|-------|------------|")
    
    # SR accuracy
    opt_sr = cav_opt['sr_correct'].mean()*100
    blind_sr = cav_blind['sr_correct'].mean()*100
    lines.append(f"| SR Accuracy (±{SR_TOLERANCE_HZ} Hz) | {opt_sr:.1f}% | {blind_sr:.1f}% | {blind_sr-opt_sr:+.1f}% |")
    
    # BR accuracy
    opt_br = cav_opt['br_correct'].mean()*100
    blind_br = cav_blind['br_correct'].mean()*100
    lines.append(f"| BR Accuracy (±{BR_TOLERANCE_HZ} Hz) | {opt_br:.1f}% | {blind_br:.1f}% | {blind_br-opt_br:+.1f}% |")
    
    # Blade count
    opt_n = cav_opt['n_blades_correct'].mean()*100
    blind_n = cav_blind['n_blades_correct'].mean()*100
    lines.append(f"| Blade Count (exact) | {opt_n:.1f}% | {blind_n:.1f}% | {blind_n-opt_n:+.1f}% |")
    
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Accuracy by Vessel Class")
    lines.append("")
    lines.append("| Vessel Class | Optimal SR | Blind SR | Optimal BR | Blind BR |")
    lines.append("|--------------|------------|----------|------------|----------|")
    
    for vc in ['small_craft', 'fishing_vessel', 'cargo_ship', 'tanker']:
        opt_vc = cav_opt[cav_opt['vessel_class'] == vc]
        blind_vc = cav_blind[cav_blind['vessel_class'] == vc]
        
        if len(opt_vc) > 0:
            opt_sr_vc = opt_vc['sr_correct'].mean()*100
            blind_sr_vc = blind_vc['sr_correct'].mean()*100
            opt_br_vc = opt_vc['br_correct'].mean()*100
            blind_br_vc = blind_vc['br_correct'].mean()*100
            lines.append(f"| {vc} | {opt_sr_vc:.1f}% | {blind_sr_vc:.1f}% | {opt_br_vc:.1f}% | {blind_br_vc:.1f}% |")
    
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Key Insights")
    lines.append("")
    
    # Calculate degradation
    sr_degradation = opt_sr - blind_sr
    br_degradation = opt_br - blind_br
    
    if sr_degradation > 5:
        lines.append(f"- ⚠️ **SR detection degrades {sr_degradation:.1f}% in blind mode** - PSO may not find optimal cavitation band")
    elif sr_degradation < -5:
        lines.append(f"- ✅ **SR detection improves {-sr_degradation:.1f}% in blind mode** - PSO finds better bands than presets")
    else:
        lines.append(f"- SR detection is similar between modes (diff = {sr_degradation:.1f}%)")
    
    if br_degradation > 5:
        lines.append(f"- ⚠️ **BR detection degrades {br_degradation:.1f}% in blind mode**")
    elif br_degradation < -5:
        lines.append(f"- ✅ **BR detection improves {-br_degradation:.1f}% in blind mode**")
    else:
        lines.append(f"- BR detection is similar between modes (diff = {br_degradation:.1f}%)")
    
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Generated by demon_batch_validator.py with --compare flag*")
    
    return "\n".join(lines)


if __name__ == '__main__':
    sys.exit(main())
