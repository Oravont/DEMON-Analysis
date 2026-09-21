#!/usr/bin/env python3
"""
Command-line DEMON plot generator.

Examples (from repo root):
    python src/demon_cli.py examples/test_30s_cargo_v2.npy --output outputs/cargo_demon.png
    python src/demon_cli.py examples/test_30s_fishing_v2.npy --bandpass 500 2000 --max-mod 50 --output outputs/fishing_demon.png

Optional metadata:
    python src/demon_cli.py examples/test_30s_cargo_v2.npy --br 22.0 --blades 4 --harmonics 4 --output outputs/cargo_known_br.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from demon_core import DemonConfig, compute_demon_spectrum, detect_br_sr
from demon_plotting import plot_demon_spectrum


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a DEMON modulation spectrum plot from a .npy waveform.")
    p.add_argument("input", type=str, help="Path to input .npy waveform (1-D array).")
    p.add_argument("--fs", type=int, default=16000, help="Sampling rate (Hz). Default: 16000")
    p.add_argument("--bandpass", nargs=2, type=float, default=(500.0, 2000.0),
                   metavar=("LOW", "HIGH"), help="Cavitation bandpass in Hz. Default: 500 2000")
    p.add_argument("--max-mod", type=float, default=50.0, help="Max modulation frequency (Hz). Default: 50")
    p.add_argument("--env-hp", type=float, default=0.5, help="Envelope high-pass (Hz). Default: 0.5")
    p.add_argument("--decimate-to", type=float, default=400.0, help="Envelope sample rate after decimation (Hz). Default: 400")
    p.add_argument("--br", type=float, default=None, help="Expected blade rate (Hz). If provided, harmonics are marked accordingly.")
    p.add_argument("--blades", type=int, default=None, help="Number of blades (for SR inference if --br provided).")
    p.add_argument("--harmonics", type=int, default=4, help="Number of BR harmonics to mark. Default: 4")
    p.add_argument("--output", type=str, default=None, help="Output PNG path. Default: outputs/<stem>_demon.png")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")

    x = np.load(in_path)
    if x.ndim != 1:
        x = x.reshape(-1)

    cfg = DemonConfig(
        fs=args.fs,
        bandpass_hz=(float(args.bandpass[0]), float(args.bandpass[1])),
        max_mod_hz=float(args.max_mod),
        envelope_hp_hz=float(args.env_hp),
        decimate_to_hz=float(args.decimate_to),
        n_harmonics=int(args.harmonics),
    )

    spec = compute_demon_spectrum(x, cfg)
    det = detect_br_sr(spec, cfg, expected_br_hz=args.br, n_blades=args.blades)

    out_path = Path(args.output) if args.output else Path("outputs") / f"{in_path.stem}_demon.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    title = f"DEMON Spectrum — {in_path.stem}"
    plot_demon_spectrum(
        spec.freqs_hz,
        spec.mag_db,
        title=title,
        bandpass_hz=spec.bandpass_hz,
        blade_rate_hz=det.blade_rate_hz,
        shaft_rate_hz=det.shaft_rate_hz,
        n_harmonics=args.harmonics,
        xlim_hz=(0.0, cfg.max_mod_hz),
        outfile=str(out_path),
    )

    print("DEMON:", in_path.name)
    print(f"  bandpass: {spec.bandpass_hz[0]:.0f}–{spec.bandpass_hz[1]:.0f} Hz")
    if det.blade_rate_hz is not None:
        print(f"  detected BR: {det.blade_rate_hz:.3f} Hz")
    if det.shaft_rate_hz is not None:
        extra = f" (n={det.n_blades})" if det.n_blades is not None else ""
        print(f"  detected SR: {det.shaft_rate_hz:.3f} Hz{extra}")
    print(f"  note: {det.notes}")
    print(f"  saved: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
