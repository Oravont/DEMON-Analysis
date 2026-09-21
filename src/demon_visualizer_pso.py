#!/usr/bin/env python3
"""
DEMON Visualizer - Professional Blade Rate Modulation Spectrum Plots

Detection of Envelope Modulation On Noise (DEMON) analysis tool for
underwater acoustic vessel classification.

Generates publication-quality DEMON spectrum plots with:
- Dark theme optimized for presentations
- Blade rate harmonic markers
- Configurable cavitation band parameters

Usage:
    python demon_visualizer.py clip.npy --br 22.0 --harmonics 4
    python demon_visualizer.py clip.npy --bandpass 500 1500 --br 8.0 --output plot.png

Author: DEMON Analysis Project
Date: January 2026
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import signal as scipy_signal
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


# =============================================================================
# PROFESSIONAL PLOT STYLING
# =============================================================================

# Color scheme - dark theme for presentations
STYLE = {
    'bg_color': '#0a1628',       # Dark navy background
    'text_color': '#caf0f8',     # Light cyan for labels
    'accent_color': '#00b4d8',   # Cyan for spectrum line
    'marker_color': '#ff6b35',   # Orange for BR harmonic markers
    'grid_color': '#234567',     # Subtle grid lines
    'spectrum_lw': 2.5,          # Spectrum line width
    'marker_lw': 1.5,            # Harmonic marker line width
    'font_size': 12,
    'title_size': 14,
    'label_size': 11,
}


def apply_dark_style():
    """Apply dark theme to matplotlib."""
    plt.rcParams.update({
        'figure.facecolor': STYLE['bg_color'],
        'axes.facecolor': STYLE['bg_color'],
        'axes.edgecolor': STYLE['text_color'],
        'axes.labelcolor': STYLE['text_color'],
        'text.color': STYLE['text_color'],
        'xtick.color': STYLE['text_color'],
        'ytick.color': STYLE['text_color'],
        'grid.color': STYLE['grid_color'],
        'font.size': STYLE['font_size'],
        'axes.titlesize': STYLE['title_size'],
        'axes.labelsize': STYLE['label_size'],
    })


# =============================================================================
# DEMON ANALYSIS FUNCTIONS
# =============================================================================

def compute_demon_spectrum(signal, fs, bandpass_low=500, bandpass_high=1500, 
                           max_freq=50, zero_pad_factor=1):
    """
    Compute DEMON spectrum from audio signal.
    
    DEMON Analysis Pipeline:
    1. Bandpass filter to isolate cavitation band
    2. Hilbert transform for analytic signal
    3. Extract amplitude envelope
    4. AC coupling (remove DC)
    5. FFT of envelope to reveal modulation frequencies
    6. Convert to dB scale (normalized to peak)
    
    Parameters
    ----------
    signal : np.ndarray
        Audio waveform (mono)
    fs : int
        Sample rate in Hz
    bandpass_low : int
        Lower cavitation band limit in Hz (default: 500)
    bandpass_high : int
        Upper cavitation band limit in Hz (default: 1500)
    max_freq : float
        Maximum modulation frequency to return in Hz (default: 50)
    zero_pad_factor : int
        Zero-padding multiplier for frequency resolution (default: 1)
    
    Returns
    -------
    freqs : np.ndarray
        Modulation frequencies in Hz
    spectrum_db : np.ndarray
        Amplitude in dB re max (normalized so peak = 0 dB)
    """
    # Validate inputs
    nyq = fs / 2
    if bandpass_high >= nyq:
        bandpass_high = int(nyq * 0.95)
        print(f"Warning: Adjusted bandpass_high to {bandpass_high} Hz (below Nyquist)")
    
    # Step 1: Bandpass filter (6th order Butterworth)
    b, a = scipy_signal.butter(6, [bandpass_low/nyq, bandpass_high/nyq], btype='band')
    filtered = scipy_signal.filtfilt(b, a, signal)
    
    # Step 2-3: Hilbert transform and envelope extraction
    analytic_signal = scipy_signal.hilbert(filtered)
    envelope = np.abs(analytic_signal)
    
    # Step 4: AC coupling (remove mean)
    envelope_ac = envelope - np.mean(envelope)
    
    # Apply Hann window for natural peak broadening (reduces spectral leakage artifacts)
    window = np.hanning(len(envelope_ac))
    envelope_windowed = envelope_ac * window
    
    # Step 5: FFT with optional zero-padding for frequency resolution
    n_fft = len(envelope_windowed) * zero_pad_factor
    freqs = fftfreq(n_fft, 1/fs)
    spectrum = np.abs(fft(envelope_windowed, n=n_fft)) / len(envelope_windowed)
    
    # Extract positive frequencies up to max_freq
    mask = (freqs >= 0) & (freqs <= max_freq)
    demon_freqs = freqs[mask]
    demon_spectrum = spectrum[mask]
    
    # Step 6: Convert to dB re max
    max_val = np.max(demon_spectrum)
    if max_val > 0:
        demon_db = 20 * np.log10(demon_spectrum / max_val + 1e-10)
    else:
        demon_db = np.zeros_like(demon_spectrum)
    
    return demon_freqs, demon_db


def auto_detect_cavitation_band(signal, fs, min_freq=300, max_freq=6000, bandwidth=800):
    """
    Automatically detect the cavitation band from signal power spectrum.
    
    Looks for elevated broadband energy (excluding narrowband tonals).
    
    Parameters
    ----------
    signal : np.ndarray
        Audio waveform
    fs : int
        Sample rate in Hz
    min_freq : float
        Minimum frequency to consider (Hz) - excludes machinery tonals
    max_freq : float
        Maximum frequency to consider (Hz)
    bandwidth : float
        Width of bandpass filter to use (Hz)
    
    Returns
    -------
    results : dict
        {
            'center_freq': float,
            'bandpass_low': int,
            'bandpass_high': int,
            'confidence': float
        }
    """
    from scipy.ndimage import uniform_filter1d
    
    # Compute power spectrum
    n_fft = min(8192, len(signal))
    freqs_psd = np.fft.rfftfreq(n_fft, 1/fs)
    
    # Use Welch's method for smoother estimate
    freqs_psd, psd = scipy_signal.welch(signal, fs, nperseg=n_fft, noverlap=n_fft//2)
    psd_db = 10 * np.log10(psd + 1e-10)
    
    # Focus on the search range
    mask = (freqs_psd >= min_freq) & (freqs_psd <= max_freq)
    freqs_masked = freqs_psd[mask]
    psd_masked = psd_db[mask]
    
    if len(freqs_masked) == 0:
        return {'center_freq': 1000, 'bandpass_low': 600, 'bandpass_high': 1400, 'confidence': 0.0}
    
    # Smooth heavily to find broadband envelope (ignore narrowband tonals)
    # Window size proportional to bandwidth we want to detect
    freq_resolution = freqs_masked[1] - freqs_masked[0] if len(freqs_masked) > 1 else 10
    smooth_window = max(1, int(bandwidth / 2 / freq_resolution))
    psd_smooth = uniform_filter1d(psd_masked, size=smooth_window)
    
    # Find peak of smoothed spectrum (broadband energy center)
    peak_idx = np.argmax(psd_smooth)
    center_freq = freqs_masked[peak_idx]
    
    # Calculate confidence based on how much the peak stands out
    psd_range = np.max(psd_smooth) - np.min(psd_smooth)
    peak_prominence = psd_smooth[peak_idx] - np.median(psd_smooth)
    confidence = min(1.0, peak_prominence / (psd_range + 1e-10))
    
    # Define bandpass limits
    half_bw = bandwidth / 2
    bandpass_low = max(50, int(center_freq - half_bw))
    bandpass_high = min(int(fs / 2 - 100), int(center_freq + half_bw))
    
    return {
        'center_freq': center_freq,
        'bandpass_low': bandpass_low,
        'bandpass_high': bandpass_high,
        'confidence': confidence
    }


def demon_quality_score(signal, fs, bandpass_low, bandpass_high, min_br=2.0, max_br=45.0):
    """
    Compute a quality score for DEMON analysis with given bandpass parameters.
    
    Higher score = better DEMON spectrum (clearer harmonics, higher SNR).
    
    Parameters
    ----------
    signal : np.ndarray
        Audio waveform
    fs : int
        Sample rate
    bandpass_low : float
        Lower bandpass limit (Hz)
    bandpass_high : float
        Upper bandpass limit (Hz)
    min_br : float
        Minimum blade rate to consider (Hz)
    max_br : float
        Maximum blade rate to consider (Hz)
    
    Returns
    -------
    score : float
        Quality score (higher is better, negative for minimization)
    """
    try:
        # Compute DEMON spectrum
        freqs, spectrum_db = compute_demon_spectrum(
            signal, fs, 
            bandpass_low=int(bandpass_low), 
            bandpass_high=int(bandpass_high)
        )
        
        # Find peaks
        peak_freqs, peak_amps = find_peaks_in_demon(freqs, spectrum_db, threshold_db=-30)
        
        if len(peak_freqs) < 2:
            return -100  # No useful peaks
        
        # Sort by amplitude
        sorted_idx = np.argsort(peak_amps)[::-1]
        peak_freqs = peak_freqs[sorted_idx]
        peak_amps = peak_amps[sorted_idx]
        
        # Find strongest peak in BR range
        br_candidate = None
        br_amp = -100
        for freq, amp in zip(peak_freqs, peak_amps):
            if min_br <= freq <= max_br:
                br_candidate = freq
                br_amp = amp
                break
        
        if br_candidate is None:
            return -100  # No BR candidate
        
        # Score components:
        score = 0.0
        
        # 1. Peak prominence (how much BR stands above noise floor)
        noise_floor = np.median(spectrum_db)
        prominence = br_amp - noise_floor
        score += prominence * 2  # Weight prominence heavily
        
        # 2. Harmonic consistency (are there peaks at 2×, 3×, 4× BR?)
        harmonic_bonus = 0
        for h in range(2, 5):
            expected = br_candidate * h
            if expected > 50:  # Beyond display range
                break
            # Check if there's a peak near expected harmonic
            for pf, pa in zip(peak_freqs[:10], peak_amps[:10]):
                if abs(pf - expected) < 1.5:  # Within 1.5 Hz
                    harmonic_bonus += (5.0 / h)  # Reward, weighted by harmonic number
                    break
        score += harmonic_bonus
        
        # 3. Sub-harmonic check (SR = BR/n for n=3,4,5,6)
        subharmonic_bonus = 0
        for n in range(3, 7):
            expected_sr = br_candidate / n
            if expected_sr < 1.0:
                continue
            for pf, pa in zip(peak_freqs[:10], peak_amps[:10]):
                if abs(pf - expected_sr) < 0.5:
                    subharmonic_bonus += 3.0  # Reward finding SR
                    break
        score += subharmonic_bonus
        
        # 4. Noise floor penalty (cleaner spectrum is better)
        noise_std = np.std(spectrum_db[spectrum_db < noise_floor + 5])
        score -= noise_std * 0.5  # Penalize noisy spectra
        
        return score
        
    except Exception:
        return -100  # Invalid parameters


def pso_optimize_bandpass(signal, fs, min_freq=200, max_freq=6000, 
                          min_bandwidth=400, max_bandwidth=1500,
                          n_particles=15, max_iter=30, verbose=True):
    """
    Use Particle Swarm Optimization to find optimal DEMON bandpass parameters.
    
    Optimizes center frequency and bandwidth to maximize DEMON spectral quality.
    
    Parameters
    ----------
    signal : np.ndarray
        Audio waveform
    fs : int
        Sample rate
    min_freq : float
        Minimum center frequency to search (Hz)
    max_freq : float
        Maximum center frequency to search (Hz)
    min_bandwidth : float
        Minimum bandwidth to try (Hz)
    max_bandwidth : float
        Maximum bandwidth to try (Hz)
    n_particles : int
        Number of PSO particles (default: 15)
    max_iter : int
        Maximum PSO iterations (default: 30)
    verbose : bool
        Print progress
    
    Returns
    -------
    results : dict
        {
            'center_freq': float,
            'bandwidth': float,
            'bandpass_low': int,
            'bandpass_high': int,
            'score': float,
            'iterations': int
        }
    """
    try:
        from pyswarm import pso
    except ImportError:
        if verbose:
            print("Warning: pyswarm not installed, falling back to grid search")
        return _grid_search_bandpass(signal, fs, min_freq, max_freq, 
                                     min_bandwidth, max_bandwidth, verbose)
    
    nyquist = fs / 2
    
    def objective(params):
        """Objective function to minimize (negative of quality score)."""
        center_freq, bandwidth = params
        
        # Calculate bandpass limits
        bandpass_low = center_freq - bandwidth / 2
        bandpass_high = center_freq + bandwidth / 2
        
        # Enforce constraints
        if bandpass_low < 50 or bandpass_high > nyquist - 50:
            return 1000  # Invalid
        if bandpass_high <= bandpass_low + 100:
            return 1000  # Too narrow
        
        # Get quality score (negate for minimization)
        score = demon_quality_score(signal, fs, bandpass_low, bandpass_high)
        return -score
    
    # Define search bounds: [center_freq, bandwidth]
    lb = [min_freq + min_bandwidth/2, min_bandwidth]
    ub = [min(max_freq, nyquist - max_bandwidth/2), max_bandwidth]
    
    if verbose:
        print(f"PSO optimization: searching for optimal cavitation band...")
        print(f"  Center freq range: {lb[0]:.0f} - {ub[0]:.0f} Hz")
        print(f"  Bandwidth range: {lb[1]:.0f} - {ub[1]:.0f} Hz")
    
    # Run PSO
    best_params, best_score = pso(
        objective, lb, ub,
        swarmsize=n_particles,
        maxiter=max_iter,
        debug=False
    )
    
    center_freq, bandwidth = best_params
    bandpass_low = int(center_freq - bandwidth / 2)
    bandpass_high = int(center_freq + bandwidth / 2)
    
    if verbose:
        print(f"  Optimal center: {center_freq:.0f} Hz")
        print(f"  Optimal bandwidth: {bandwidth:.0f} Hz")
        print(f"  Bandpass: {bandpass_low} - {bandpass_high} Hz")
        print(f"  Quality score: {-best_score:.1f}")
    
    return {
        'center_freq': center_freq,
        'bandwidth': bandwidth,
        'bandpass_low': bandpass_low,
        'bandpass_high': bandpass_high,
        'score': -best_score,
        'method': 'PSO'
    }


def _grid_search_bandpass(signal, fs, min_freq, max_freq, min_bandwidth, max_bandwidth, verbose):
    """Fallback grid search if pyswarm not available."""
    best_score = -1000
    best_params = (1000, 800)
    
    for center in np.linspace(min_freq + min_bandwidth/2, max_freq, 10):
        for bw in np.linspace(min_bandwidth, max_bandwidth, 5):
            bandpass_low = center - bw/2
            bandpass_high = center + bw/2
            if bandpass_low < 50 or bandpass_high > fs/2 - 50:
                continue
            score = demon_quality_score(signal, fs, bandpass_low, bandpass_high)
            if score > best_score:
                best_score = score
                best_params = (center, bw)
    
    center_freq, bandwidth = best_params
    return {
        'center_freq': center_freq,
        'bandwidth': bandwidth,
        'bandpass_low': int(center_freq - bandwidth/2),
        'bandpass_high': int(center_freq + bandwidth/2),
        'score': best_score,
        'method': 'grid_search'
    }


def find_peaks_in_demon(freqs, spectrum_db, threshold_db=-20, min_distance_hz=1.0):
    """
    Find spectral peaks in DEMON spectrum.
    
    Parameters
    ----------
    freqs : np.ndarray
        Frequency array in Hz
    spectrum_db : np.ndarray
        Spectrum in dB
    threshold_db : float
        Minimum peak height relative to max (default: -20 dB)
    min_distance_hz : float
        Minimum distance between peaks in Hz (default: 1.0)
    
    Returns
    -------
    peak_freqs : np.ndarray
        Frequencies of detected peaks
    peak_amplitudes : np.ndarray
        Amplitudes of detected peaks in dB
    """
    # Convert min_distance to samples
    freq_resolution = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    min_distance_samples = max(1, int(min_distance_hz / freq_resolution))
    
    # Find peaks
    peak_indices, properties = scipy_signal.find_peaks(
        spectrum_db, 
        height=threshold_db,
        distance=min_distance_samples
    )
    
    peak_freqs = freqs[peak_indices]
    peak_amplitudes = spectrum_db[peak_indices]
    
    return peak_freqs, peak_amplitudes


def detect_blade_rate(freqs, spectrum_db, min_br=2.0, max_br=40.0, n_blades_range=(3, 6)):
    """
    Automatically detect blade rate and shaft rate from DEMON spectrum.
    
    Uses harmonic pattern matching to find the most likely BR.
    
    Parameters
    ----------
    freqs : np.ndarray
        Frequency array in Hz
    spectrum_db : np.ndarray
        Spectrum in dB
    min_br : float
        Minimum expected blade rate (Hz)
    max_br : float
        Maximum expected blade rate (Hz)
    n_blades_range : tuple
        Range of blade counts to consider (min, max)
    
    Returns
    -------
    results : dict
        {
            'blade_rate': float or None,
            'shaft_rate': float or None,
            'n_blades': int or None,
            'confidence': float,
            'peaks': list of (freq, amplitude) tuples
        }
    """
    # Find significant peaks
    peak_freqs, peak_amps = find_peaks_in_demon(freqs, spectrum_db, threshold_db=-25)
    
    if len(peak_freqs) == 0:
        return {'blade_rate': None, 'shaft_rate': None, 'n_blades': None, 
                'confidence': 0.0, 'peaks': []}
    
    # Sort by amplitude (strongest first)
    sorted_indices = np.argsort(peak_amps)[::-1]
    peak_freqs = peak_freqs[sorted_indices]
    peak_amps = peak_amps[sorted_indices]
    
    # Get top peaks for analysis
    top_peaks = list(zip(peak_freqs[:10], peak_amps[:10]))
    
    best_br = None
    best_sr = None
    best_n = None
    best_score = 0
    
    # Strategy 1: Assume strongest peak in BR range is the blade rate
    for freq, amp in top_peaks:
        if min_br <= freq <= max_br:
            # Check if this could be BR by looking for harmonics
            harmonic_score = 0
            for h in range(2, 5):  # Check 2nd, 3rd, 4th harmonics
                expected_harmonic = freq * h
                # Look for peak near expected harmonic
                for pf, pa in top_peaks:
                    if abs(pf - expected_harmonic) < 1.0:  # Within 1 Hz
                        harmonic_score += (1.0 / h)  # Weight lower harmonics more
                        break
            
            # Check for sub-harmonics (potential SR)
            for n in range(n_blades_range[0], n_blades_range[1] + 1):
                expected_sr = freq / n
                for pf, pa in top_peaks:
                    if abs(pf - expected_sr) < 0.5:  # Within 0.5 Hz
                        # Found potential SR
                        sr_score = harmonic_score + 0.5
                        if sr_score > best_score:
                            best_score = sr_score
                            best_br = freq
                            best_sr = pf
                            best_n = n
                        break
            
            # Even without SR match, strong harmonic pattern suggests BR
            if harmonic_score > best_score and best_br is None:
                best_score = harmonic_score
                best_br = freq
    
    # If no BR found, use strongest peak in range as fallback
    if best_br is None:
        for freq, amp in top_peaks:
            if min_br <= freq <= max_br:
                best_br = freq
                break
    
    # Calculate confidence (0-1)
    confidence = min(1.0, best_score / 2.0) if best_score > 0 else 0.3
    
    return {
        'blade_rate': best_br,
        'shaft_rate': best_sr,
        'n_blades': best_n,
        'confidence': confidence,
        'peaks': top_peaks
    }


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_demon_spectrum(freqs, spectrum_db, blade_rate=None, shaft_rate=None, n_harmonics=4,
                        bandpass_low=500, bandpass_high=1500, detected_n_blades=None,
                        title=None, output_path=None, show=True,
                        figsize=(14, 6), dpi=150):
    """
    Create professional DEMON spectrum plot.
    
    Parameters
    ----------
    freqs : np.ndarray
        Modulation frequencies in Hz
    spectrum_db : np.ndarray
        Amplitude in dB re max
    blade_rate : float, optional
        Expected blade rate in Hz (for BR harmonic markers, orange)
    shaft_rate : float, optional
        Expected shaft rate in Hz (for SR harmonic markers, green)
    n_harmonics : int
        Number of harmonics to mark for BR and SR (default: 4)
    bandpass_low : int
        Lower cavitation band limit (for title)
    bandpass_high : int
        Upper cavitation band limit (for title)
    detected_n_blades : int, optional
        Number of blades (for display in legend)
    title : str, optional
        Custom plot title
    output_path : str, optional
        Path to save figure (if None, not saved)
    show : bool
        Whether to display the plot (default: True)
    figsize : tuple
        Figure size in inches (default: (14, 6))
    dpi : int
        Resolution for saved figure (default: 150)
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    ax : matplotlib.axes.Axes
        The axes object
    """
    apply_dark_style()
    
    fig, ax = plt.subplots(figsize=figsize, facecolor=STYLE['bg_color'])
    ax.set_facecolor(STYLE['bg_color'])
    
    # Plot spectrum with clean gradient fill
    ax.plot(freqs, spectrum_db, color=STYLE['accent_color'], 
            linewidth=STYLE['spectrum_lw'], zorder=3)
    
    # Simple solid fill - reliable across all platforms
    ax.fill_between(freqs, spectrum_db, -40, 
                    color='#0a4d6e', alpha=0.85, zorder=1)
    
    # Helper to get spectrum value at a frequency
    def get_db_at_freq(f):
        idx = np.argmin(np.abs(freqs - f))
        return spectrum_db[idx]
    
    # SR color (green/teal to contrast with orange BR)
    sr_color = '#2ecc71'  # Green for shaft rate
    
    # Add shaft rate harmonic markers if specified
    if shaft_rate is not None and shaft_rate > 0:
        for i in range(1, n_harmonics + 1):
            harmonic_freq = shaft_rate * i
            if harmonic_freq <= freqs[-1]:
                # Skip if this SR harmonic coincides with a BR harmonic
                is_br_harmonic = False
                if blade_rate is not None:
                    for j in range(1, n_harmonics + 1):
                        if abs(harmonic_freq - blade_rate * j) < 0.5:
                            is_br_harmonic = True
                            break
                
                if not is_br_harmonic:
                    ax.axvline(x=harmonic_freq, color=sr_color,
                              linestyle=':', linewidth=STYLE['marker_lw'],
                              alpha=0.7, zorder=4)
                    
                    # Label ABOVE the peak
                    peak_db = get_db_at_freq(harmonic_freq)
                    label_y = min(peak_db + 4, 3)  # Above peak but not above plot
                    ax.text(harmonic_freq, label_y, f'{i}×SR\n({harmonic_freq:.1f}Hz)',
                           color=sr_color, fontsize=8,
                           ha='center', va='bottom', fontweight='bold')
    
    # Add blade rate harmonic markers if specified
    if blade_rate is not None and blade_rate > 0:
        for i in range(1, n_harmonics + 1):
            harmonic_freq = blade_rate * i
            if harmonic_freq <= freqs[-1]:
                ax.axvline(x=harmonic_freq, color=STYLE['marker_color'],
                          linestyle='--', linewidth=STYLE['marker_lw'],
                          alpha=0.8, zorder=4)
                
                # Label ABOVE the peak
                peak_db = get_db_at_freq(harmonic_freq)
                label_y = min(peak_db + 4, 3)  # Above peak but not above plot
                ax.text(harmonic_freq, label_y, f'{i}×BR\n({harmonic_freq:.0f}Hz)',
                       color=STYLE['marker_color'], fontsize=9,
                       ha='center', va='bottom', fontweight='bold')
    
    # Configure axes
    ax.set_xlabel('Modulation Frequency (Hz)')
    ax.set_ylabel('Amplitude (dB re max)')
    ax.set_xlim(0, freqs[-1])
    ax.set_ylim(-40, 8)  # Slightly higher to accommodate labels above peaks
    
    # Grid styling
    ax.grid(True, alpha=0.3, color=STYLE['grid_color'])
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(10))
    
    # Title
    if title is None:
        title = f'DEMON (Cavitation Band {bandpass_low}-{bandpass_high} Hz) → Blade Rate Modulation'
    ax.set_title(title, fontsize=STYLE['title_size'], fontweight='bold', 
                 color=STYLE['text_color'])
    
    # Add detection summary box if we have detected values
    if blade_rate is not None or shaft_rate is not None:
        summary_lines = []
        if blade_rate is not None:
            summary_lines.append(f'BR = {blade_rate:.2f} Hz')
        if shaft_rate is not None:
            summary_lines.append(f'SR = {shaft_rate:.2f} Hz')
        if detected_n_blades is not None:
            summary_lines.append(f'n = {detected_n_blades} blades')
        
        if summary_lines:
            summary_text = '\n'.join(summary_lines)
            ax.text(0.98, 0.97, summary_text, transform=ax.transAxes,
                   fontsize=10, fontweight='bold', color='#ffffff',
                   ha='right', va='top',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='#1a3a5c', 
                            edgecolor=STYLE['accent_color'], alpha=0.9))
    
    plt.tight_layout()
    
    # Save if output path specified
    if output_path is not None:
        fig.savefig(output_path, dpi=dpi, facecolor=STYLE['bg_color'],
                    edgecolor='none', bbox_inches='tight')
        print(f"Saved plot to: {output_path}")
    
    if show:
        plt.show()
    
    return fig, ax


def analyze_and_plot(audio_path, fs=16000, bandpass_low=500, bandpass_high=1500,
                     blade_rate=None, shaft_rate=None, n_harmonics=4, output_path=None, 
                     show=True, verbose=True, auto_detect=True):
    """
    Complete DEMON analysis pipeline: load, analyze, and visualize.
    
    Parameters
    ----------
    audio_path : str
        Path to .npy audio file
    fs : int
        Sample rate in Hz (default: 16000)
    bandpass_low : int
        Lower cavitation band limit in Hz (default: 500)
    bandpass_high : int
        Upper cavitation band limit in Hz (default: 1500)
    blade_rate : float, optional
        Expected blade rate in Hz (if None and auto_detect=True, will detect)
    shaft_rate : float, optional
        Expected shaft rate in Hz (if None and auto_detect=True, will detect)
    n_harmonics : int
        Number of harmonics to mark for BR and SR (default: 4)
    output_path : str, optional
        Path to save figure
    show : bool
        Whether to display the plot (default: True)
    verbose : bool
        Print analysis details (default: True)
    auto_detect : bool
        Automatically detect BR/SR if not provided (default: True)
    
    Returns
    -------
    results : dict
        Dictionary containing:
        - freqs: Modulation frequencies
        - spectrum_db: DEMON spectrum in dB
        - detected_peaks: (frequencies, amplitudes) of peaks
        - blade_rate: Detected or provided BR
        - shaft_rate: Detected or provided SR
        - n_blades: Inferred blade count
        - fig: matplotlib figure
        - ax: matplotlib axes
    """
    # Load audio
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    signal = np.load(audio_path)
    
    if verbose:
        duration = len(signal) / fs
        print(f"Loaded: {audio_path.name}")
        print(f"  Duration: {duration:.2f} s | Sample rate: {fs} Hz")
        print(f"  Cavitation band: {bandpass_low}-{bandpass_high} Hz")
    
    # Compute DEMON spectrum
    freqs, spectrum_db = compute_demon_spectrum(
        signal, fs, 
        bandpass_low=bandpass_low,
        bandpass_high=bandpass_high
    )
    
    # Find peaks
    peak_freqs, peak_amps = find_peaks_in_demon(freqs, spectrum_db)
    
    # Auto-detect BR/SR if not provided
    detected_n_blades = None
    if auto_detect and (blade_rate is None or shaft_rate is None):
        detection = detect_blade_rate(freqs, spectrum_db)
        
        if blade_rate is None and detection['blade_rate'] is not None:
            blade_rate = detection['blade_rate']
        if shaft_rate is None and detection['shaft_rate'] is not None:
            shaft_rate = detection['shaft_rate']
        if detection['n_blades'] is not None:
            detected_n_blades = detection['n_blades']
        
        if verbose:
            print(f"\n--- Auto-Detection Results ---")
            if detection['blade_rate'] is not None:
                print(f"  Blade Rate (BR): {detection['blade_rate']:.2f} Hz")
            if detection['shaft_rate'] is not None:
                print(f"  Shaft Rate (SR): {detection['shaft_rate']:.2f} Hz")
            if detection['n_blades'] is not None:
                print(f"  Inferred blades: {detection['n_blades']}")
            print(f"  Confidence: {detection['confidence']:.0%}")
    
    if verbose and len(peak_freqs) > 0:
        print(f"\nDetected peaks (top 5):")
        sorted_indices = np.argsort(peak_amps)[::-1][:5]
        for idx in sorted_indices:
            print(f"  {peak_freqs[idx]:.2f} Hz: {peak_amps[idx]:.1f} dB")
    
    # Generate plot
    fig, ax = plot_demon_spectrum(
        freqs, spectrum_db,
        blade_rate=blade_rate,
        shaft_rate=shaft_rate,
        n_harmonics=n_harmonics,
        bandpass_low=bandpass_low,
        bandpass_high=bandpass_high,
        detected_n_blades=detected_n_blades,
        output_path=output_path,
        show=show
    )
    
    return {
        'freqs': freqs,
        'spectrum_db': spectrum_db,
        'detected_peaks': (peak_freqs, peak_amps),
        'blade_rate': blade_rate,
        'shaft_rate': shaft_rate,
        'n_blades': detected_n_blades,
        'fig': fig,
        'ax': ax
    }


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

# Vessel type presets for cavitation bands
VESSEL_PRESETS = {
    'fishing': {'bandpass': (1400, 2200), 'desc': 'Fishing vessel (1400-2200 Hz)'},
    'cargo': {'bandpass': (400, 1200), 'desc': 'Cargo ship (400-1200 Hz)'},
    'tanker': {'bandpass': (300, 1000), 'desc': 'Tanker (300-1000 Hz)'},
    'small': {'bandpass': (1500, 3000), 'desc': 'Small craft (1500-3000 Hz)'},
    'default': {'bandpass': (500, 1500), 'desc': 'General purpose (500-1500 Hz)'},
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='DEMON Visualizer - Professional Blade Rate Modulation Spectrum Plots',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # PSO OPTIMIZATION (best quality, recommended)
  python demon_visualizer.py clip.npy --pso
  
  # Optimizes bandpass to maximize DEMON spectral quality!

  # Fast auto-detection (good for quick analysis)
  python demon_visualizer.py clip.npy --auto-band

  # With vessel type hint (uses preset cavitation band)
  python demon_visualizer.py clip.npy --vessel fishing

  # Manual override (for validation against known parameters)
  python demon_visualizer.py clip.npy --bandpass 1400 2200 --br 22.0 --sr 5.5

Cavitation Band Selection (in priority order):
  --bandpass    Manual override (most specific)
  --pso         PSO optimization (best quality, ~5-10 sec)
  --auto-band   Auto-detect from signal PSD (fast, ~1 sec)
  --vessel      Use vessel type preset
  (default)     500-1500 Hz if nothing specified

PSO Optimization:
  Uses Particle Swarm Optimization to find the bandpass filter parameters
  that maximize DEMON spectral quality. The objective function rewards:
  - Strong peak prominence (BR stands out from noise)
  - Clear harmonic series (peaks at 2×, 3×, 4× BR)
  - Sub-harmonic presence (SR = BR/n confirms blade count)

Vessel Type Presets (--vessel):
  fishing   1400-2200 Hz   Fishing vessels, high-speed props
  cargo     400-1200 Hz    Cargo ships, large slow props  
  tanker    300-1000 Hz    Tankers, very large slow props
  small     1500-3000 Hz   Small craft, outboards, fast props
  default   500-1500 Hz    General purpose
        """
    )
    
    parser.add_argument('audio_file', type=str,
                        help='Path to .npy audio file')
    
    parser.add_argument('--fs', type=int, default=16000,
                        help='Sample rate in Hz (default: 16000)')
    
    parser.add_argument('--vessel', type=str, default=None,
                        choices=['fishing', 'cargo', 'tanker', 'small', 'default'],
                        help='Vessel type preset (sets cavitation band automatically)')
    
    parser.add_argument('--bandpass', type=int, nargs=2, default=None,
                        metavar=('LOW', 'HIGH'),
                        help='Cavitation band limits in Hz (overrides --vessel and --auto-band)')
    
    parser.add_argument('--auto-band', action='store_true',
                        help='Automatically detect cavitation band from signal spectrum')
    
    parser.add_argument('--pso', action='store_true',
                        help='Use PSO optimization to find optimal cavitation band (best quality)')
    
    parser.add_argument('--br', type=float, default=None,
                        help='Manual blade rate override (Hz) - disables BR auto-detection')
    
    parser.add_argument('--sr', type=float, default=None,
                        help='Manual shaft rate override (Hz) - disables SR auto-detection')
    
    parser.add_argument('--harmonics', type=int, default=4,
                        help='Number of harmonics to mark for BR and SR (default: 4)')
    
    parser.add_argument('--no-auto', action='store_true',
                        help='Disable auto-detection (show raw spectrum only)')
    
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output path for saved plot (e.g., plot.png)')
    
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display plot (useful for batch processing)')
    
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress analysis output')
    
    return parser.parse_args()


def main():
    """Main entry point for CLI."""
    args = parse_args()
    
    # Load signal first if we need auto-band detection or PSO
    signal = None
    if (args.auto_band or args.pso) and args.bandpass is None:
        try:
            signal = np.load(args.audio_file)
        except Exception as e:
            print(f"Error loading file for band detection: {e}", file=sys.stderr)
            return 1
    
    # Resolve bandpass from various sources (in priority order)
    if args.bandpass is not None:
        # Explicit bandpass overrides everything
        bandpass_low, bandpass_high = args.bandpass
        if not args.quiet:
            print(f"Using manual bandpass: {bandpass_low}-{bandpass_high} Hz")
    elif args.pso:
        # PSO optimization (best quality, slower)
        pso_result = pso_optimize_bandpass(signal, args.fs, verbose=not args.quiet)
        bandpass_low = pso_result['bandpass_low']
        bandpass_high = pso_result['bandpass_high']
    elif args.auto_band:
        # Simple auto-detect from signal PSD (fast)
        band_result = auto_detect_cavitation_band(signal, args.fs)
        bandpass_low = band_result['bandpass_low']
        bandpass_high = band_result['bandpass_high']
        if not args.quiet:
            print(f"Auto-detected cavitation band:")
            print(f"  Center frequency: {band_result['center_freq']:.0f} Hz")
            print(f"  Bandpass: {bandpass_low}-{bandpass_high} Hz")
            print(f"  Confidence: {band_result['confidence']:.0%}")
    elif args.vessel is not None:
        # Use vessel preset
        preset = VESSEL_PRESETS[args.vessel]
        bandpass_low, bandpass_high = preset['bandpass']
        if not args.quiet:
            print(f"Using vessel preset: {preset['desc']}")
    else:
        # Default fallback
        bandpass_low, bandpass_high = 500, 1500
        if not args.quiet:
            print("Using default bandpass: 500-1500 Hz")
            print("Tip: Use --pso for optimal band detection (slower but best)")
            print("     Or --auto-band for fast detection")
            print("     Or --vessel <type> for presets")
    
    # Determine auto-detection mode for BR/SR
    auto_detect = not args.no_auto
    
    try:
        results = analyze_and_plot(
            audio_path=args.audio_file,
            fs=args.fs,
            bandpass_low=bandpass_low,
            bandpass_high=bandpass_high,
            blade_rate=args.br,
            shaft_rate=args.sr,
            n_harmonics=args.harmonics,
            output_path=args.output,
            show=not args.no_show,
            verbose=not args.quiet,
            auto_detect=auto_detect
        )
        
        if not args.quiet:
            print("\nDEMON analysis complete.")
        
        return 0
        
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error during analysis: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
