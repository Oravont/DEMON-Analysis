#!/usr/bin/env python3
"""
DEMON plotting utilities (visual standard)

- dark navy background
- cyan spectrum trace
- orange harmonic markers for BR multiples
- normalised dB scale (0 dB re max)
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


STYLE = {
    "bg": "#0a1628",
    "fg": "#caf0f8",
    "grid": "#234567",
    "trace": "#00b4d8",
    "marker": "#ff6b35",
    "trace_lw": 2.5,
    "marker_lw": 1.6,
    "font": 12,
    "title": 14,
    "label": 11,
}


def apply_dark_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": STYLE["bg"],
        "axes.facecolor": STYLE["bg"],
        "axes.edgecolor": STYLE["fg"],
        "axes.labelcolor": STYLE["fg"],
        "text.color": STYLE["fg"],
        "xtick.color": STYLE["fg"],
        "ytick.color": STYLE["fg"],
        "grid.color": STYLE["grid"],
        "font.size": STYLE["font"],
        "axes.titlesize": STYLE["title"],
        "axes.labelsize": STYLE["label"],
    })


def plot_demon_spectrum(
    freqs_hz: np.ndarray,
    mag_db: np.ndarray,
    *,
    title: str = "DEMON Modulation Spectrum",
    bandpass_hz: Optional[Tuple[float, float]] = None,
    blade_rate_hz: Optional[float] = None,
    shaft_rate_hz: Optional[float] = None,
    n_harmonics: int = 4,
    xlim_hz: Tuple[float, float] = (0.0, 50.0),
    outfile: Optional[str] = None,
    show: bool = False,
):
    apply_dark_style()
    fig, ax = plt.subplots(figsize=(10, 6), dpi=160)
    ax.plot(freqs_hz, mag_db, linewidth=STYLE["trace_lw"])
    ax.set_xlim(*xlim_hz)
    ax.set_ylim(-80, 5)
    ax.grid(True, alpha=0.35, linestyle="--", linewidth=0.8)
    ax.set_xlabel("Modulation frequency (Hz)")
    ax.set_ylabel("Magnitude (dB re max)")
    ax.set_title(title)

    if bandpass_hz is not None:
        lo, hi = bandpass_hz
        ax.text(0.99, 0.02, f"Cavitation band: {lo:.0f}–{hi:.0f} Hz",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=10, alpha=0.95)

    if blade_rate_hz is not None and blade_rate_hz > 0:
        for k in range(1, max(1, n_harmonics) + 1):
            f = blade_rate_hz * k
            if f > xlim_hz[1]:
                break
            ax.axvline(f, color=STYLE["marker"], linewidth=STYLE["marker_lw"], alpha=0.9)
        ax.text(0.02, 0.96, f"BR ≈ {blade_rate_hz:.2f} Hz",
                transform=ax.transAxes, ha="left", va="top", fontsize=11)

    if shaft_rate_hz is not None and shaft_rate_hz > 0:
        ax.axvline(shaft_rate_hz, color=STYLE["fg"], linewidth=1.2, alpha=0.8, linestyle=":")
        ax.text(0.02, 0.90, f"SR ≈ {shaft_rate_hz:.2f} Hz",
                transform=ax.transAxes, ha="left", va="top", fontsize=11)

    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    fig.tight_layout()

    if outfile:
        fig.savefig(outfile, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return fig
