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
        Zero-padding multiplier for frequency resolution (default: 4)
    
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


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_demon_spectrum(freqs, spectrum_db, blade_rate=None, n_harmonics=4,
                        bandpass_low=500, bandpass_high=1500,
                        title=None, output_path=None, show=True,
                        figsize=(12, 6), dpi=150):
    """
    Create professional DEMON spectrum plot.
    
    Parameters
    ----------
    freqs : np.ndarray
        Modulation frequencies in Hz
    spectrum_db : np.ndarray
        Amplitude in dB re max
    blade_rate : float, optional
        Expected blade rate in Hz (for harmonic markers)
    n_harmonics : int
        Number of BR harmonics to mark (default: 4)
    bandpass_low : int
        Lower cavitation band limit (for title)
    bandpass_high : int
        Upper cavitation band limit (for title)
    title : str, optional
        Custom plot title
    output_path : str, optional
        Path to save figure (if None, not saved)
    show : bool
        Whether to display the plot (default: True)
    figsize : tuple
        Figure size in inches (default: (12, 6))
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
    
    # Plot spectrum with smooth gradient fill
    ax.plot(freqs, spectrum_db, color=STYLE['accent_color'], 
            linewidth=STYLE['spectrum_lw'], zorder=3)
    
    # Create smooth vertical gradient using imshow
    gradient = np.linspace(0, 1, 256).reshape(-1, 1)
    gradient = np.hstack([gradient] * 10)
    
    # Dark blue at bottom (#012a4a) to medium blue at top (#0096c7)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('demon_gradient', 
                                              ['#012a4a', '#014f86', '#0077b6'])
    
    # Draw gradient background in the plot area
    extent = [freqs[0], freqs[-1], -40, 5]
    ax.imshow(gradient, aspect='auto', extent=extent, origin='lower',
              cmap=cmap, alpha=0.7, zorder=0)
    
    # Fill above the curve with background color to mask gradient
    ax.fill_between(freqs, spectrum_db, 5, color=STYLE['bg_color'], zorder=2)
    
    # Add blade rate harmonic markers if specified
    if blade_rate is not None and blade_rate > 0:
        for i in range(1, n_harmonics + 1):
            harmonic_freq = blade_rate * i
            if harmonic_freq <= freqs[-1]:
                ax.axvline(x=harmonic_freq, color=STYLE['marker_color'],
                          linestyle='--', linewidth=STYLE['marker_lw'],
                          alpha=0.8, zorder=4)
                
                # Add label positioned within the plot area (not at top edge)
                label_y = 3  # Just below the top of y-axis (which is 5)
                ax.text(harmonic_freq, label_y, f'{i}×BR\n({harmonic_freq:.0f}Hz)',
                       color=STYLE['marker_color'], fontsize=9,
                       ha='center', va='top', fontweight='bold')
    
    # Configure axes
    ax.set_xlabel('Modulation Frequency (Hz)')
    ax.set_ylabel('Amplitude (dB re max)')
    ax.set_xlim(0, freqs[-1])
    ax.set_ylim(-40, 5)
    
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
                     blade_rate=None, n_harmonics=4, output_path=None, 
                     show=True, verbose=True):
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
        Expected blade rate in Hz (for harmonic markers)
    n_harmonics : int
        Number of BR harmonics to mark (default: 4)
    output_path : str, optional
        Path to save figure
    show : bool
        Whether to display the plot (default: True)
    verbose : bool
        Print analysis details (default: True)
    
    Returns
    -------
    results : dict
        Dictionary containing:
        - freqs: Modulation frequencies
        - spectrum_db: DEMON spectrum in dB
        - detected_peaks: (frequencies, amplitudes) of peaks
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
    
    if verbose and len(peak_freqs) > 0:
        print(f"\nDetected peaks (top 5):")
        sorted_indices = np.argsort(peak_amps)[::-1][:5]
        for idx in sorted_indices:
            print(f"  {peak_freqs[idx]:.2f} Hz: {peak_amps[idx]:.1f} dB")
    
    # Generate plot
    fig, ax = plot_demon_spectrum(
        freqs, spectrum_db,
        blade_rate=blade_rate,
        n_harmonics=n_harmonics,
        bandpass_low=bandpass_low,
        bandpass_high=bandpass_high,
        output_path=output_path,
        show=show
    )
    
    return {
        'freqs': freqs,
        'spectrum_db': spectrum_db,
        'detected_peaks': (peak_freqs, peak_amps),
        'fig': fig,
        'ax': ax
    }


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='DEMON Visualizer - Professional Blade Rate Modulation Spectrum Plots',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with blade rate markers
  python demon_visualizer.py test_30s_fishing_v2.npy --br 22.0

  # Custom cavitation band for cargo vessel
  python demon_visualizer.py test_30s_cargo_v2.npy --bandpass 400 1200 --br 10.0

  # Save plot without displaying
  python demon_visualizer.py clip.npy --br 8.0 --output demon_plot.png --no-show

  # Full analysis with all parameters
  python demon_visualizer.py clip.npy --fs 16000 --bandpass 500 1500 --br 22.0 --harmonics 6

Typical Cavitation Bands by Vessel Type:
  Cargo Ship:     400-1200 Hz  (BR: 4-12 Hz)
  Tanker:         300-1000 Hz  (BR: 3-8 Hz)
  Fishing Vessel: 1400-2200 Hz (BR: 15-30 Hz)
  Small Craft:    1500-3000 Hz (BR: 20-50 Hz)
        """
    )
    
    parser.add_argument('audio_file', type=str,
                        help='Path to .npy audio file')
    
    parser.add_argument('--fs', type=int, default=16000,
                        help='Sample rate in Hz (default: 16000)')
    
    parser.add_argument('--bandpass', type=int, nargs=2, default=[500, 1500],
                        metavar=('LOW', 'HIGH'),
                        help='Cavitation band limits in Hz (default: 500 1500)')
    
    parser.add_argument('--br', type=float, default=None,
                        help='Expected blade rate in Hz (enables harmonic markers)')
    
    parser.add_argument('--harmonics', type=int, default=4,
                        help='Number of BR harmonics to mark (default: 4)')
    
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
    
    try:
        results = analyze_and_plot(
            audio_path=args.audio_file,
            fs=args.fs,
            bandpass_low=args.bandpass[0],
            bandpass_high=args.bandpass[1],
            blade_rate=args.br,
            n_harmonics=args.harmonics,
            output_path=args.output,
            show=not args.no_show,
            verbose=not args.quiet
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
