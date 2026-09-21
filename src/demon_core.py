#!/usr/bin/env python3
"""
DEMON core utilities (signal-processing only)

Implements classical DEMON (Detection of Envelope Modulation on Noise):
bandpass → Hilbert envelope → AC coupling → FFT of envelope.

Designed for 16 kHz clips from SKANN-SSL, but configurable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import signal as sp_signal
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks


@dataclass(frozen=True)
class DemonConfig:
    fs: int = 16000
    bandpass_hz: Tuple[float, float] = (500.0, 2000.0)  # cavitation band (typical)
    max_mod_hz: float = 50.0
    envelope_hp_hz: float = 0.5  # remove slow drift after envelope extraction
    decimate_to_hz: float = 400.0  # envelope sampling rate after decimation
    butter_order: int = 4
    peak_min_hz: float = 0.5
    peak_prom_db: float = 6.0  # peak prominence threshold in dB
    n_harmonics: int = 4


@dataclass
class DemonSpectrum:
    freqs_hz: np.ndarray
    mag_db: np.ndarray  # normalised so max == 0 dB
    bandpass_hz: Tuple[float, float]
    fs_env: float


@dataclass
class DemonDetection:
    blade_rate_hz: Optional[float]
    shaft_rate_hz: Optional[float]
    n_blades: Optional[int]
    peaks_hz: np.ndarray
    peaks_db: np.ndarray
    notes: str = ""


def _butter_bandpass(x: np.ndarray, fs: int, band: Tuple[float, float], order: int) -> np.ndarray:
    lo, hi = band
    if not (0 < lo < hi < fs / 2):
        raise ValueError(f"Invalid bandpass {band} for fs={fs}. Must satisfy 0 < lo < hi < fs/2.")
    b, a = sp_signal.butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return sp_signal.filtfilt(b, a, x).astype(np.float32)


def _butter_highpass(x: np.ndarray, fs: float, hp_hz: float, order: int) -> np.ndarray:
    if hp_hz <= 0:
        return x
    if hp_hz >= fs / 2:
        raise ValueError(f"Invalid highpass {hp_hz} for fs={fs}. Must be < fs/2.")
    b, a = sp_signal.butter(order, hp_hz / (fs / 2), btype="high")
    return sp_signal.filtfilt(b, a, x).astype(np.float32)


def _hilbert_envelope(x: np.ndarray) -> np.ndarray:
    return np.abs(sp_signal.hilbert(x)).astype(np.float32)


def _decimate_envelope(env: np.ndarray, fs: int, target_fs: float) -> Tuple[np.ndarray, float]:
    if target_fs >= fs:
        return env, float(fs)
    q = int(np.floor(fs / target_fs))
    q = max(q, 1)
    fs_new = fs / q
    env_dec = sp_signal.decimate(env, q, ftype="fir", zero_phase=True).astype(np.float32)
    return env_dec, float(fs_new)


def compute_demon_spectrum(x: np.ndarray, cfg: DemonConfig = DemonConfig()) -> DemonSpectrum:
    x = np.asarray(x, dtype=np.float32).flatten()
    if x.size < int(cfg.fs * 0.5):
        raise ValueError("Clip is too short for DEMON. Provide at least ~0.5 s, preferably 10–30 s.")

    x_bp = _butter_bandpass(x, cfg.fs, cfg.bandpass_hz, cfg.butter_order)

    env = _hilbert_envelope(x_bp)

    env = env - float(np.mean(env))
    env = _butter_highpass(env, cfg.fs, cfg.envelope_hp_hz, order=2)

    env_dec, fs_env = _decimate_envelope(env, cfg.fs, cfg.decimate_to_hz)

    n = env_dec.size
    win = np.hanning(n).astype(np.float32)
    y = env_dec * win
    mag = np.abs(rfft(y)) + 1e-12
    freqs = rfftfreq(n, d=1.0 / fs_env)

    mask = freqs <= cfg.max_mod_hz
    freqs = freqs[mask]
    mag = mag[mask]

    mag_db = 20.0 * np.log10(mag)
    mag_db -= float(np.max(mag_db))

    return DemonSpectrum(freqs_hz=freqs, mag_db=mag_db, bandpass_hz=cfg.bandpass_hz, fs_env=fs_env)


def _find_peaks_db(freqs: np.ndarray, mag_db: np.ndarray, cfg: DemonConfig):
    m = freqs >= cfg.peak_min_hz
    f = freqs[m]
    d = mag_db[m]
    peaks, props = find_peaks(d, prominence=cfg.peak_prom_db)
    if peaks.size == 0:
        return np.array([]), np.array([])
    pf = f[peaks]
    pd = d[peaks]
    order = np.argsort(pd)[::-1]
    return pf[order], pd[order]


def detect_br_sr(
    spec: DemonSpectrum,
    cfg: DemonConfig = DemonConfig(),
    expected_br_hz: Optional[float] = None,
    n_blades: Optional[int] = None,
    blade_candidates: Tuple[int, ...] = (3, 4, 5),
) -> DemonDetection:
    freqs = spec.freqs_hz
    db = spec.mag_db
    pk_f, pk_db = _find_peaks_db(freqs, db, cfg)

    if pk_f.size == 0:
        return DemonDetection(None, None, None, pk_f, pk_db, notes="No peaks found above prominence threshold.")

    if expected_br_hz is not None:
        br = float(expected_br_hz)
        sr = float(br / n_blades) if n_blades else None
        nb = int(n_blades) if n_blades else None
        return DemonDetection(br, sr, nb, pk_f, pk_db, notes="Used expected BR (and blades if provided).")

    plausible = (pk_f >= 3.0) & (pk_f <= min(45.0, cfg.max_mod_hz))
    if np.any(plausible):
        br0 = float(pk_f[plausible][0])
        # pick highest (closest to 0 dB) among plausible
        br0 = float(pk_f[plausible][np.argmax(pk_db[plausible])])
    else:
        br0 = float(pk_f[0])

    if n_blades is not None:
        sr = br0 / float(n_blades)
        return DemonDetection(br0, sr, int(n_blades), pk_f, pk_db, notes="BR picked; SR inferred from provided blade count.")

    best_score = -1e9
    best_sr = None
    best_n = None
    best_note = "BR picked as strongest peak."

    for n in blade_candidates:
        sr = br0 / float(n)
        tol = max(0.15, 0.03 * sr)
        near = np.abs(pk_f - sr) <= tol
        if not np.any(near):
            continue
        sr_peak_db = float(pk_db[near][0])
        score = sr_peak_db - 0.5 * abs((pk_f[near][0] - sr) / tol)
        if score > best_score:
            best_score = score
            best_sr = sr
            best_n = n
            best_note = f"BR strongest; SR validated near {sr:.2f} Hz for n={n}."

    return DemonDetection(br0, best_sr, best_n, pk_f, pk_db, notes=best_note)
