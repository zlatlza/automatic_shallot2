import numpy as np
import json

class Oscillator:
    WAVEFORMS = ['sine', 'square', 'sawtooth', 'triangle']
    EQ_BAND_FREQUENCIES = [60, 170, 310, 600, 1000, 3000, 6000, 12000]  # Standard 8-band EQ frequencies
    
    _osc_count = 0 # Class variable to ensure unique default names

    def __init__(self):
        Oscillator._osc_count += 1
        self.name = f"Oscillator {Oscillator._osc_count}" # Default name
        self.waveform = 'sine'
        self.frequency = 440.0  # A4 note
        self.amplitude = 0.5
        self.phase = 0.0
        self.enabled = True  # New enable/disable flag
        # New parameters for sound design
        self.detune = 0.0  # Detune in cents (-100 to +100)
        self.attack = 0.1  # Attack time in seconds
        self.decay = 0.1   # Decay time in seconds
        self.sustain = 0.7 # Sustain level (0-1)
        self.release = 0.3 # Release time in seconds
        self.filter_cutoff = 1.0  # Normalized cutoff frequency (0-1)
        self.filter_resonance = 0.0  # Filter resonance (0-1)
        self.pan = 0.5  # Stereo panning (0 = left, 0.5 = center, 1 = right)
        self.eq_gains = [0.0] * 8 # Gains for 8 EQ bands in dB (-12dB to +12dB for example)
        self.load_waveform_definitions()
    
    def load_waveform_definitions(self):
        """Load waveform definitions from JSON file"""
        try:
            with open('waveform_definitions.json', 'r') as f:
                self.waveform_definitions = json.load(f)
        except FileNotFoundError:
            # Default basic waveforms if file not found
            self.waveform_definitions = {
                "sine": {"type": "basic", "description": "Pure sine wave", "harmonics": []},
                "square": {"type": "basic", "description": "Square wave", "harmonics": []},
                "sawtooth": {"type": "basic", "description": "Sawtooth wave", "harmonics": []},
                "triangle": {"type": "basic", "description": "Triangle wave", "harmonics": []}
            }
    
    def apply_envelope(self, samples: np.ndarray, sample_rate: int, duration_ms: int) -> np.ndarray:
        """Apply ADSR envelope to the samples"""
        total_samples = len(samples)
        attack_samples = min(int(self.attack * sample_rate), total_samples)
        decay_samples = min(int(self.decay * sample_rate), total_samples - attack_samples)
        release_samples = min(int(self.release * sample_rate), total_samples - attack_samples - decay_samples)
        sustain_samples = total_samples - attack_samples - decay_samples - release_samples
        
        # Create envelope segments
        envelope = np.zeros(total_samples)
        
        # Attack phase
        if attack_samples > 0:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        
        # Decay phase
        if decay_samples > 0:
            decay_end = attack_samples + decay_samples
            envelope[attack_samples:decay_end] = np.linspace(1, self.sustain, decay_samples)
        
        # Sustain phase
        if sustain_samples > 0:
            sustain_end = attack_samples + decay_samples + sustain_samples
            envelope[attack_samples + decay_samples:sustain_end] = self.sustain
        
        # Release phase
        if release_samples > 0:
            envelope[-release_samples:] = np.linspace(self.sustain, 0, release_samples)
        
        return samples * envelope
    
    def apply_filter(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply a simple lowpass filter"""
        if self.filter_cutoff >= 1.0 and self.filter_resonance == 0:
            return samples
            
        # Simple IIR lowpass filter
        alpha = self.filter_cutoff * 0.9  # Scale cutoff to reasonable range
        resonance = 1.0 - (self.filter_resonance * 0.99)  # Prevent instability
        filtered = np.zeros_like(samples)
        filtered[0] = samples[0]
        
        for i in range(1, len(samples)):
            filtered[i] = (alpha * samples[i] + (1 - alpha) * filtered[i-1]) * resonance
            
        return filtered
    
    def _biquad_filter(self, samples: np.ndarray, Fs: int, F0: float, Q: float, gain_db: float) -> np.ndarray:
        """Apply a single biquad (peaking EQ) filter."""
        if gain_db == 0:
            return samples

        A = 10**(gain_db / 40)  # Convert dB to linear gain for A (affects peak)
        w0 = 2 * np.pi * F0 / Fs
        alpha = np.sin(w0) / (2 * Q)

        # Peaking EQ coefficients
        b0 = 1 + alpha * A
        b1 = -2 * np.cos(w0)
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha / A
        
        # Normalize coefficients by a0
        b0 /= a0
        b1 /= a0
        b2 /= a0
        # a0 is 1 after normalization
        a1 /= a0
        a2 /= a0

        filtered = np.zeros_like(samples)
        x1, x2 = 0, 0  # input delay elements
        y1, y2 = 0, 0  # output delay elements

        for i in range(len(samples)):
            # Standard IIR difference equation:
            # y[n] = (b0/a0)*x[n] + (b1/a0)*x[n-1] + (b2/a0)*x[n-2] - (a1/a0)*y[n-1] - (a2/a0)*y[n-2]
            # Here a0 is already divided out from other coeffs.
            filtered[i] = b0 * samples[i] + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            
            # Update delay elements
            x2 = x1
            x1 = samples[i]
            y2 = y1
            y1 = filtered[i]
            
        return filtered

    def apply_eq(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply 8-band EQ to the samples."""
        processed_samples = samples.copy()
        q_factor = 1.41 # A common Q for graphic EQs, can be adjusted
        
        for i in range(8):
            if self.eq_gains[i] != 0.0: # Only apply if gain is not 0dB
                freq = self.EQ_BAND_FREQUENCIES[i]
                gain = self.eq_gains[i]
                processed_samples = self._biquad_filter(processed_samples, sample_rate, freq, q_factor, gain)
        return processed_samples

    def generate_samples(self, duration_ms: int, sample_rate: int = 44100) -> np.ndarray:
        num_samples = int(sample_rate * duration_ms / 1000)
        t = np.linspace(0, duration_ms / 1000, num_samples, False)
        
        # Apply detune
        freq = self.frequency * (2 ** (self.detune / 1200))
        
        # Get waveform definition
        wave_def = self.waveform_definitions.get(self.waveform, self.waveform_definitions["sine"])
        
        if wave_def["type"] == "basic":
            # Generate basic waveform
            if self.waveform == 'sine':
                samples = self.amplitude * np.sin(2 * np.pi * freq * t + self.phase)
            elif self.waveform == 'square':
                samples = self.amplitude * np.sign(np.sin(2 * np.pi * freq * t + self.phase))
            elif self.waveform == 'sawtooth':
                samples = self.amplitude * 2 * (freq * t - np.floor(0.5 + freq * t))
            else:  # triangle
                samples = self.amplitude * 2 * np.abs(2 * (freq * t - np.floor(0.5 + freq * t))) - 1
        else:
            # Generate custom waveform using harmonics
            samples = np.zeros(num_samples)
            for harmonic in wave_def["harmonics"]:
                harmonic_freq = freq * harmonic["frequency"]
                harmonic_amp = self.amplitude * harmonic["amplitude"]
                samples += harmonic_amp * np.sin(2 * np.pi * harmonic_freq * t + self.phase)
            
            # Normalize
            if len(wave_def["harmonics"]) > 0:
                max_abs_sample = max(abs(samples.min()), abs(samples.max()))
                if max_abs_sample > 0: # Avoid division by zero
                    samples = samples / max_abs_sample
                samples *= self.amplitude
        
        # Apply envelope and effects
        samples = self.apply_envelope(samples, sample_rate, duration_ms)
        samples = self.apply_filter(samples, sample_rate)
        samples = self.apply_eq(samples, sample_rate) # Apply EQ
        
        # Apply panning (will be used later in stereo conversion)
        self.current_pan = self.pan
        
        return samples 