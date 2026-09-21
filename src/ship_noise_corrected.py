"""
SKANN-SSL Stage -1: Ship Noise Generator (CORRECTED)
=====================================================

FIX: Changed swell modulation from 0.5 Hz to realistic 0.05-0.15 Hz

Original bug:
    swell_mod = 1.0 + 0.3 * np.sin(2 * np.pi * 0.5 * t_center + swell_phase)
    
This created harmonics at 1.0, 1.5, 2.0 Hz - interfering with DEMON analysis.

Fix:
    swell_freq = rng.uniform(0.05, 0.15)  # Real ocean swell: 7-20 second period
    swell_mod = 1.0 + 0.2 * np.sin(2 * np.pi * swell_freq * t_center + swell_phase)
    
Now harmonics stay below 0.6 Hz - well outside SR detection range (1-10 Hz).
"""

import numpy as np
from scipy import signal as scipy_signal
from dataclasses import dataclass
from typing import Optional, Tuple, List

# =============================================================================
# CONFIGURATION
# =============================================================================
FS = 16000
P_REF = 1e-6

# Physical constants
RHO_WATER = 998.0
P_AMBIENT = 101325.0
P_VAPOR = 2340.0
DELTA_P = P_AMBIENT - P_VAPOR

# High sample rate for burst generation
FS_GENERATION = 200000

# Vessel class parameters
VESSEL_CLASSES = {
    'small_craft': {
        'shaft_rate': (5.0, 12.0),
        'n_blades': (3, 4),
        'n_shaft_harmonics': 4,
        'n_bpf_harmonics': 3,
        'shaft_harmonic_decay': 6.0,
        'bpf_harmonic_decay': 4.0,
        'broadband_level': 0.3,
        'broadband_rolloff': -3.0,
        'cavitation_peak': 2000.0,
    },
    'fishing_vessel': {
        'shaft_rate': (3.0, 8.0),
        'n_blades': (3, 5),
        'n_shaft_harmonics': 5,
        'n_bpf_harmonics': 4,
        'shaft_harmonic_decay': 5.0,
        'bpf_harmonic_decay': 3.5,
        'broadband_level': 0.4,
        'broadband_rolloff': -4.0,
        'cavitation_peak': 1800.0,
    },
    'cargo_ship': {
        'shaft_rate': (1.0, 3.0),
        'n_blades': (4, 6),
        'n_shaft_harmonics': 6,
        'n_bpf_harmonics': 5,
        'shaft_harmonic_decay': 4.0,
        'bpf_harmonic_decay': 3.0,
        'broadband_level': 0.5,
        'broadband_rolloff': -5.0,
        'cavitation_peak': 800.0,
    },
    'tanker': {
        'shaft_rate': (0.8, 2.5),
        'n_blades': (4, 6),
        'n_shaft_harmonics': 7,
        'n_bpf_harmonics': 6,
        'shaft_harmonic_decay': 3.5,
        'bpf_harmonic_decay': 2.5,
        'broadband_level': 0.6,
        'broadband_rolloff': -6.0,
        'cavitation_peak': 600.0,
    },
}

CAVITATION_PROB = {
    'small_craft': 0.6,
    'fishing_vessel': 0.8,
    'cargo_ship': 0.9,
    'tanker': 0.95,
}

CAVITATION_INTENSITY_RANGE = {
    'small_craft': (0.3, 0.7),
    'fishing_vessel': (0.4, 0.8),
    'cargo_ship': (0.5, 0.9),
    'tanker': (0.6, 1.0),
}


@dataclass
class VesselParams:
    """Parameters defining a vessel's acoustic signature."""
    vessel_class: str
    shaft_rate: float
    n_blades: int
    blade_pass_freq: float
    n_shaft_harmonics: int
    n_bpf_harmonics: int
    shaft_harmonic_decay: float
    bpf_harmonic_decay: float
    broadband_level: float
    broadband_rolloff: float
    has_cavitation: bool
    cavitation_intensity: float
    generator_freq: float
    
    # Populated during generation
    equipment_base_freq: float = 0.0
    resonance_freq_1: float = 0.0
    resonance_freq_2: float = 0.0
    resonance_freq_3: float = 0.0
    cavitation_peak_freq: float = 0.0
    n_cavitation_bursts: int = 0


class ShipNoiseGenerator:
    """
    Ship noise generator with CORRECTED swell modulation.
    """
    
    CAVITATION_GAIN = 0.001
    
    def __init__(self, fs: int = FS, n_samples: int = 160000):
        self.fs = fs
        self.n_samples = n_samples
        self.nyquist = fs // 2
        self.freq_res = fs / n_samples
        self.duration = n_samples / fs
        
        self.n_bins = n_samples // 2 + 1
        self.freq_grid = np.fft.rfftfreq(n_samples, 1/fs)
        
        self.fs_gen = FS_GENERATION
        self.n_samples_gen = int(self.duration * self.fs_gen)
    
    def create_vessel_params(self, vessel_class: str,
                             rng: Optional[np.random.Generator] = None) -> VesselParams:
        """Create randomized vessel parameters."""
        if rng is None:
            rng = np.random.default_rng()
        
        vc = VESSEL_CLASSES.get(vessel_class, VESSEL_CLASSES['cargo_ship'])
        
        shaft_rate = rng.uniform(*vc['shaft_rate'])
        n_blades_range = vc['n_blades']
        n_blades = rng.integers(n_blades_range[0], n_blades_range[1] + 1)
        
        has_cavitation = rng.random() < CAVITATION_PROB.get(vessel_class, 0.5)
        cav_range = CAVITATION_INTENSITY_RANGE.get(vessel_class, (0.3, 0.7))
        cavitation_intensity = rng.uniform(*cav_range) if has_cavitation else 0.0
        
        generator_freq = rng.choice([0, 50, 60], p=[0.3, 0.5, 0.2])
        
        return VesselParams(
            vessel_class=vessel_class,
            shaft_rate=shaft_rate,
            n_blades=n_blades,
            blade_pass_freq=shaft_rate * n_blades,
            n_shaft_harmonics=vc['n_shaft_harmonics'],
            n_bpf_harmonics=vc['n_bpf_harmonics'],
            shaft_harmonic_decay=vc['shaft_harmonic_decay'],
            bpf_harmonic_decay=vc['bpf_harmonic_decay'],
            broadband_level=vc['broadband_level'],
            broadband_rolloff=vc['broadband_rolloff'],
            has_cavitation=has_cavitation,
            cavitation_intensity=cavitation_intensity,
            generator_freq=generator_freq,
        )
    
    def _add_tonal(self, spectrum: np.ndarray, freq: float, level_db: float):
        """Add a tonal component."""
        if freq <= 0 or freq >= self.nyquist:
            return
        bin_idx = int(round(freq / self.freq_res))
        if 0 < bin_idx < len(spectrum):
            amplitude = 10 ** (level_db / 20)
            spectrum[bin_idx] += amplitude
    
    def _add_shaft_harmonics(self, spectrum: np.ndarray, params: VesselParams, ref_db: float):
        for h in range(1, params.n_shaft_harmonics + 1):
            freq = h * params.shaft_rate
            level = ref_db - (h - 1) * params.shaft_harmonic_decay
            self._add_tonal(spectrum, freq, level)
    
    def _add_bpf_harmonics(self, spectrum: np.ndarray, params: VesselParams, ref_db: float):
        for h in range(1, params.n_bpf_harmonics + 1):
            freq = h * params.blade_pass_freq
            level = ref_db + 3 - (h - 1) * params.bpf_harmonic_decay
            self._add_tonal(spectrum, freq, level)
    
    def _add_generator_harmonics(self, spectrum: np.ndarray, params: VesselParams, ref_db: float):
        if params.generator_freq <= 0:
            return
        for h in range(1, 4):
            freq = h * params.generator_freq
            level = ref_db - 5 - (h - 1) * 6
            self._add_tonal(spectrum, freq, level)
    
    def _add_equipment_harmonics(self, spectrum: np.ndarray, params: VesselParams, 
                                  ref_db: float, rng: np.random.Generator) -> float:
        if params.equipment_base_freq == 0.0:
            return 0.0
        
        if params.equipment_base_freq < 0:
            base_freq = 25.0 if params.generator_freq in [0, 50] else 30.0
        else:
            base_freq = params.equipment_base_freq
        
        for h in range(1, 6):
            freq = h * base_freq
            level = ref_db - 8 - (h - 1) * 4 + rng.uniform(-2, 2)
            self._add_tonal(spectrum, freq, level)
        
        return base_freq
    
    def _add_structural_resonances(self, spectrum: np.ndarray, params: VesselParams,
                                    ref_db: float, rng: np.random.Generator) -> List[float]:
        vc = VESSEL_CLASSES.get(params.vessel_class, VESSEL_CLASSES['cargo_ship'])
        
        if params.vessel_class in ['cargo_ship', 'tanker']:
            n_resonances = rng.integers(2, 4)
            freq_range = (100, 500)
        else:
            n_resonances = rng.integers(1, 3)
            freq_range = (80, 300)
        
        resonance_freqs = []
        for i in range(n_resonances):
            freq = rng.uniform(*freq_range)
            q_factor = rng.uniform(5, 20)
            level = ref_db - 3 - i * 3 + rng.uniform(-2, 2)
            
            bandwidth = freq / q_factor
            for k in range(1, self.n_bins):
                f = self.freq_grid[k]
                if abs(f - freq) < 3 * bandwidth:
                    shape = np.exp(-((f - freq) / bandwidth) ** 2)
                    amp = 10 ** (level / 20) * shape
                    spectrum[k] += amp
            
            resonance_freqs.append(freq)
        
        return resonance_freqs
    
    def _add_broadband(self, spectrum: np.ndarray, params: VesselParams, ref_db: float):
        for k in range(1, self.n_bins):
            freq = self.freq_grid[k]
            if freq < 10:
                continue
            octaves = np.log2(freq / 1000)
            level = ref_db - 20 + params.broadband_rolloff * octaves
            level += params.broadband_level * 10
            psd_linear = 10 ** (level / 10)
            amplitude = np.sqrt(psd_linear * self.freq_res)
            spectrum[k] += amplitude * 0.1
    
    def _generate_burst_times(self, bpf: float, intensity: float,
                              rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate cavitation burst times with CORRECTED swell modulation.
        
        FIX: Changed from 0.5 Hz to realistic 0.05-0.15 Hz ocean swell.
        """
        blade_period = 1.0 / bpf
        n_passages = int(self.duration * bpf) + 2
        
        activity_fraction = 0.2 + 0.3 * intensity
        base_bursts = int(3 + 12 * intensity)
        
        burst_times = []
        burst_intensities = []
        
        # =====================================================================
        # CORRECTED: Realistic ocean swell frequency (0.05-0.15 Hz)
        # Old buggy code: swell_freq = 0.5 Hz (created 2 Hz harmonics!)
        # New correct code: swell_freq = 0.05-0.15 Hz (7-20 second period)
        # =====================================================================
        swell_freq = rng.uniform(0.05, 0.15)  # CORRECTED: Real ocean swell
        swell_phase = rng.uniform(0, 2 * np.pi)
        swell_amplitude = 0.2  # Reduced from 0.3
        
        for i in range(n_passages):
            # RPM jitter (±2%)
            jitter_amount = 0.02 * blade_period
            timing_jitter = rng.normal(0, jitter_amount)
            t_center = (i + 0.5) * blade_period + timing_jitter
            
            # Blade-to-blade intensity variation (±20%)
            blade_randomness = max(0, 1.0 + 0.2 * rng.standard_normal())
            
            # CORRECTED swell modulation
            swell_mod = 1.0 + swell_amplitude * np.sin(2 * np.pi * swell_freq * t_center + swell_phase)
            
            local_intensity = intensity * blade_randomness * swell_mod
            
            window_start = t_center - activity_fraction * blade_period / 2
            window_end = t_center + activity_fraction * blade_period / 2
            
            n_bursts = rng.poisson(base_bursts)
            
            for _ in range(n_bursts):
                t_burst = rng.uniform(window_start, window_end)
                if 0 < t_burst < self.duration:
                    burst_times.append(t_burst)
                    burst_intensities.append(rng.gamma(2, 0.5) * local_intensity)
        
        return np.array(burst_times), np.array(burst_intensities)
    
    def _generate_single_burst(self, t: np.ndarray, t_start: float,
                                burst_duration: float, f_carrier: float,
                                burst_type: str) -> np.ndarray:
        """Generate a single cavitation burst."""
        burst = np.zeros_like(t)
        t_rel = t - t_start
        mask = (t_rel >= 0) & (t_rel <= burst_duration * 3)
        
        if not np.any(mask):
            return burst
        
        t_local = t_rel[mask]
        
        if burst_type == 'collapse':
            tau = burst_duration / 3
            envelope = (t_local / tau) * np.exp(-t_local / tau)
        elif burst_type == 'cloud':
            tau = burst_duration / 2
            envelope = np.exp(-((t_local - tau) / tau) ** 2)
            envelope += 0.3 * np.exp(-((t_local - 2*tau) / (tau/2)) ** 2)
        elif burst_type == 'sheet':
            tau = burst_duration
            envelope = np.sin(np.pi * t_local / burst_duration) ** 2
            envelope *= np.exp(-t_local / (2 * burst_duration))
        else:
            envelope = np.exp(-t_local / burst_duration)
        
        carrier = np.sin(2 * np.pi * f_carrier * t_local)
        carrier += 0.3 * np.sin(2 * np.pi * 2 * f_carrier * t_local)
        carrier += 0.1 * np.sin(2 * np.pi * 3 * f_carrier * t_local)
        
        burst[mask] = envelope * carrier
        return burst
    
    def _generate_cavitation_bursts(self, params: VesselParams,
                                     rng: np.random.Generator) -> Tuple[np.ndarray, float, int]:
        """Generate cavitation with physical burst model."""
        if not params.has_cavitation or params.cavitation_intensity < 0.01:
            return np.zeros(self.n_samples), 0.0, 0
        
        t_gen = np.linspace(0, self.duration, self.n_samples_gen)
        signal_gen = np.zeros(self.n_samples_gen)
        
        vc = VESSEL_CLASSES.get(params.vessel_class, VESSEL_CLASSES['cargo_ship'])
        cav_peak = vc.get('cavitation_peak', 1000.0)
        
        peak_freqs_used = []
        peak_weights = []
        
        burst_times, burst_intensities = self._generate_burst_times(
            params.blade_pass_freq, params.cavitation_intensity, rng
        )
        
        n_bursts = len(burst_times)
        
        for t_burst, intensity in zip(burst_times, burst_intensities):
            r = rng.random()
            if r < 0.6:
                burst_type = 'collapse'
                R_bubble = rng.lognormal(np.log(50e-6), 0.5)
                burst_duration = 0.915 * R_bubble * np.sqrt(RHO_WATER / DELTA_P)
                f_carrier = cav_peak * rng.uniform(0.8, 2.0)
            elif r < 0.9:
                burst_type = 'cloud'
                burst_duration = rng.uniform(100e-6, 1e-3)
                f_carrier = cav_peak * rng.uniform(0.5, 1.5)
            else:
                burst_type = 'sheet'
                burst_duration = rng.uniform(1e-3, 10e-3)
                f_carrier = cav_peak * rng.uniform(0.3, 1.0)
            
            peak_freqs_used.append(f_carrier)
            peak_weights.append(intensity)
            
            burst = self._generate_single_burst(t_gen, t_burst, burst_duration, f_carrier, burst_type)
            signal_gen += burst * intensity
        
        # Downsample
        signal_out = scipy_signal.resample(signal_gen, self.n_samples)
        
        # Weighted average cavitation frequency
        if len(peak_freqs_used) > 0 and sum(peak_weights) > 0:
            cav_peak_used = np.average(peak_freqs_used, weights=peak_weights)
        else:
            cav_peak_used = cav_peak
        
        return signal_out, cav_peak_used, n_bursts
    
    def _generate_tonal_broadband(self, params: VesselParams,
                                   rng: np.random.Generator) -> np.ndarray:
        """Generate tonal + broadband components."""
        amplitude_spectrum = np.zeros(self.n_bins, dtype=np.float64)
        ref_level_db = 0.0
        
        self._add_shaft_harmonics(amplitude_spectrum, params, ref_level_db)
        self._add_bpf_harmonics(amplitude_spectrum, params, ref_level_db)
        self._add_generator_harmonics(amplitude_spectrum, params, ref_level_db)
        
        equipment_base_freq = self._add_equipment_harmonics(
            amplitude_spectrum, params, ref_level_db, rng
        )
        params.equipment_base_freq = equipment_base_freq
        
        resonance_freqs = self._add_structural_resonances(
            amplitude_spectrum, params, ref_level_db, rng
        )
        params.resonance_freq_1 = resonance_freqs[0] if len(resonance_freqs) > 0 else 0.0
        params.resonance_freq_2 = resonance_freqs[1] if len(resonance_freqs) > 1 else 0.0
        params.resonance_freq_3 = resonance_freqs[2] if len(resonance_freqs) > 2 else 0.0
        
        self._add_broadband(amplitude_spectrum, params, ref_level_db)
        
        phases = rng.uniform(0, 2 * np.pi, self.n_bins)
        phases[0] = 0.0
        if self.n_samples % 2 == 0:
            phases[-1] = 0.0
        
        complex_spectrum = amplitude_spectrum * np.exp(1j * phases)
        waveform = np.fft.irfft(complex_spectrum, n=self.n_samples)
        
        return waveform
    
    def generate(self, vessel_class: str = 'cargo_ship',
                 params: Optional[VesselParams] = None,
                 rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, VesselParams]:
        """Generate complete ship noise signal."""
        if rng is None:
            rng = np.random.default_rng()
        
        if params is None:
            params = self.create_vessel_params(vessel_class, rng)
        
        tonal_broadband = self._generate_tonal_broadband(params, rng)
        
        cavitation, cav_peak_used, n_bursts = self._generate_cavitation_bursts(params, rng)
        cavitation = cavitation * self.CAVITATION_GAIN
        params.cavitation_peak_freq = cav_peak_used
        params.n_cavitation_bursts = n_bursts
        
        waveform = tonal_broadband + cavitation
        
        return waveform, params
