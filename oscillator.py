import numpy as np
import json

class Oscillator:
    WAVEFORMS = ['sine', 'square', 'sawtooth', 'triangle']
    EQ_BAND_FREQUENCIES = [60, 170, 310, 600, 1000, 3000, 6000, 12000]  # Standard 8-band EQ frequencies
    
    def __init__(self):
        self.name = "" # Initialized as empty, to be set by ChordGenerator
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
        self.existing_waveforms_path = "waveform_definitions.json" # Define this path for the class instance
        self.load_waveform_definitions() # Initial load
        self.is_live_editing = False
        self.live_edit_points = []
        self.base_waveform_for_live_edit = None # Stores the name of the waveform being live-edited
    
    def load_waveform_definitions(self, force_reload=False):
        """Load waveform definitions from JSON file.
        If force_reload is True, it will always re-read the file.
        """
        if hasattr(self, 'waveform_definitions') and isinstance(self.waveform_definitions, dict) and not force_reload:
            # If definitions are already loaded as a dict and not forcing reload, do nothing.
            # This check is more for avoiding reload if not necessary, the main protection is below.
            return

        try:
            with open(self.existing_waveforms_path, 'r') as f: # Ensure it uses a consistent path attribute if defined
                # It's safer to default to an empty dict before trying to load
                loaded_data = json.load(f)
                if not isinstance(loaded_data, dict):
                    print(f"Warning: {self.existing_waveforms_path} root is not a dictionary. Using default waveforms.")
                    self.waveform_definitions = self._get_default_waveforms()
                else:
                    self.waveform_definitions = loaded_data
        except FileNotFoundError:
            # print(f"File {self.existing_waveforms_path} not found. Using default waveforms.")
            self.waveform_definitions = self._get_default_waveforms()
        except json.JSONDecodeError:
            print(f"Error: Could not decode {self.existing_waveforms_path}. Using default waveforms.")
            self.waveform_definitions = self._get_default_waveforms()
        except Exception as e: # Catch any other potential errors during file read/parse
            print(f"Unexpected error loading {self.existing_waveforms_path}: {e}. Using default waveforms.")
            self.waveform_definitions = self._get_default_waveforms()
        
        # Final safety: ensure it's a dict no matter what happened above
        if not hasattr(self, 'waveform_definitions') or not isinstance(self.waveform_definitions, dict):
            print("Critical fallback: waveform_definitions was not a dict. Resetting to defaults.")
            self.waveform_definitions = self._get_default_waveforms()
    
    def _get_default_waveforms(self):
        """Returns a dictionary of default basic waveforms."""
        return {
            "sine": {"type": "basic", "description": "Pure sine wave", "harmonics": []},
            "square": {"type": "basic", "description": "Square wave", "harmonics": []},
            "sawtooth": {"type": "basic", "description": "Sawtooth wave", "harmonics": []},
            "triangle": {"type": "basic", "description": "Triangle wave", "harmonics": []}
        }

    def set_live_edit_data(self, points=None, is_active=False):
        """Sets data for live waveform editing."""
        if is_active:
            if not self.is_live_editing or self.base_waveform_for_live_edit != self.waveform: 
                # If starting live edit, or if the underlying waveform changed while live edit was (somehow) active elsewhere
                self.base_waveform_for_live_edit = self.waveform
            self.live_edit_points = points if points is not None else []
            self.is_live_editing = True
        else:
            self.is_live_editing = False
            self.live_edit_points = [] 
            self.base_waveform_for_live_edit = None # Clear when live editing stops

    def get_waveform_cycle_points(self, waveform_name_to_get: str, num_points_target: int) -> list:
        """Generates a single cycle of a given waveform with a specific number of points."""
        wave_def = self.waveform_definitions.get(waveform_name_to_get)
        if not wave_def:
            print(f"Waveform definition for '{waveform_name_to_get}' not found. Returning zeros.")
            return [0.0] * num_points_target

        t = np.linspace(0, 1, num_points_target, endpoint=False) # Time vector for one cycle
        points = np.zeros(num_points_target)
        wave_type = wave_def.get("type")

        if wave_type == "basic":
            if waveform_name_to_get == 'sine':
                points = np.sin(2 * np.pi * t)
            elif waveform_name_to_get == 'square':
                points = np.sign(np.sin(2 * np.pi * t))
            elif waveform_name_to_get == 'sawtooth':
                points = 2 * (t - np.floor(0.5 + t))
            elif waveform_name_to_get == 'triangle':
                points = 2 * np.abs(2 * (t - np.floor(0.5 + t))) - 1
            else:
                print(f"Unknown basic waveform name: {waveform_name_to_get}. Returning zeros.")
        
        elif wave_type == "custom" and "harmonics" in wave_def:
            for harmonic in wave_def["harmonics"]:
                frequency_multiplier = harmonic.get("frequency", 1.0)
                amplitude_multiplier = harmonic.get("amplitude", 0.0)
                points += amplitude_multiplier * np.sin(2 * np.pi * frequency_multiplier * t)
            
            if wave_def["harmonics"]:
                max_abs = np.max(np.abs(points))
                if max_abs > 0:
                    points /= max_abs
        
        elif wave_type == "sculpted" and "points" in wave_def:
            defined_points = np.array(wave_def.get("points", []))
            if len(defined_points) == num_points_target:
                points = defined_points
            elif len(defined_points) > 0:
                # Interpolate/resample if lengths don't match
                x_defined = np.linspace(0, 1, len(defined_points), endpoint=False)
                points = np.interp(t, x_defined, defined_points)
            # else points remain zeros if defined_points is empty
        
        else:
            print(f"Unknown or malformed wave_def type: '{wave_type}' for '{waveform_name_to_get}'. Returning zeros.")
            return [0.0] * num_points_target # Explicitly return zeros for clarity
        
        # Ensure normalization for safety, though most paths should be normalized.
        # final_max_abs = np.max(np.abs(points))
        # if final_max_abs > 0:
        #     points /= final_max_abs
        
        return list(points)

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
        time_vector = np.linspace(0, duration_ms / 1000, num_samples, endpoint=False)
        
        actual_freq = self.frequency * (2 ** (self.detune / 1200))
        samples = np.zeros(num_samples)
        
        points_for_cycle = None
        use_point_based_generation = False

        if self.is_live_editing and self.live_edit_points and len(self.live_edit_points) > 0:
            points_for_cycle = np.array(self.live_edit_points)
            use_point_based_generation = True
        else:
            wave_def = self.waveform_definitions.get(self.waveform, self.waveform_definitions.get("sine"))
            if not wave_def: 
                wave_def = {"type": "basic", "description": "Pure sine wave"} # Ultimate fallback
            
            wave_type = wave_def.get("type")

            if wave_type == "sculpted" and "points" in wave_def and len(wave_def["points"]) > 0:
                points_for_cycle = np.array(wave_def["points"])
                use_point_based_generation = True
            elif wave_type == "basic":
                if self.waveform == 'sine':
                    samples = np.sin(2 * np.pi * actual_freq * time_vector + self.phase)
                elif self.waveform == 'square':
                    samples = np.sign(np.sin(2 * np.pi * actual_freq * time_vector + self.phase))
                elif self.waveform == 'sawtooth':
                    samples = 2 * (actual_freq * time_vector - np.floor(0.5 + actual_freq * time_vector))
                elif self.waveform == 'triangle':
                    samples = 2 * np.abs(2 * (actual_freq * time_vector - np.floor(0.5 + actual_freq * time_vector))) - 1
                else: # Fallback basic waveform (sine)
                    samples = np.sin(2 * np.pi * actual_freq * time_vector + self.phase)
                samples *= self.amplitude
            elif wave_type == "custom" and "harmonics" in wave_def:
                # Generate custom waveform using harmonics
                for harmonic in wave_def["harmonics"]:
                    harmonic_freq = actual_freq * harmonic.get("frequency", 1.0)
                    harmonic_amp = harmonic.get("amplitude", 0.0) # Amplitude here is relative to oscillator's main amplitude
                    samples += harmonic_amp * np.sin(2 * np.pi * harmonic_freq * time_vector + self.phase)
                if wave_def["harmonics"]: 
                    max_abs_sample = np.max(np.abs(samples))
                    if max_abs_sample > 0: 
                        samples = samples / max_abs_sample
                samples *= self.amplitude # Apply final oscillator amplitude
            # else: (unknown type, samples remain zeros, will be multiplied by amplitude)
            #    samples *= self.amplitude

        if use_point_based_generation and points_for_cycle is not None:
            if len(points_for_cycle) == 1:
                samples = np.full(num_samples, points_for_cycle[0])
            else:
                # Interpolate points_for_cycle over the sound duration
                # `phase` goes from 0 to 1 for each cycle of the oscillator's actual_freq
                phase = (time_vector * actual_freq) % 1.0 
                
                # `xp_coords` are the x-coordinates for `points_for_cycle` (0 to 1 range for one cycle)
                # To ensure proper wrapping for interpolation, append the first point to the end
                points_to_interp = np.concatenate((points_for_cycle, [points_for_cycle[0]]))
                xp_coords = np.linspace(0, 1, len(points_to_interp), endpoint=True)
                
                samples = np.interp(phase, xp_coords, points_to_interp)
            samples *= self.amplitude

        # Apply envelope and effects
        samples = self.apply_envelope(samples, sample_rate, duration_ms)
        samples = self.apply_filter(samples, sample_rate)
        samples = self.apply_eq(samples, sample_rate) # Apply EQ
        
        # Apply panning (will be used later in stereo conversion)
        self.current_pan = self.pan
        
        return samples 