import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pygame
import pygame.mixer
from pydub import AudioSegment
from pydub.generators import Sine
import math
import json
from typing import List, Dict, Tuple
import os
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
matplotlib.use('TkAgg')
from sequencer_ui import SequencerUI # Import the new SequencerUI class

class Oscillator:
    WAVEFORMS = ['sine', 'square', 'sawtooth', 'triangle']
    
    def __init__(self):
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
                samples = samples / max(abs(samples.min()), abs(samples.max()))
                samples *= self.amplitude
        
        # Apply envelope and effects
        samples = self.apply_envelope(samples, sample_rate, duration_ms)
        samples = self.apply_filter(samples, sample_rate)
        
        # Apply panning (will be used later in stereo conversion)
        self.current_pan = self.pan
        
        return samples

class ChordGenerator:
    NOTE_FREQUENCIES = {
        'C': 261.63, 'C#': 277.18, 'D': 293.66, 'D#': 311.13,
        'E': 329.63, 'F': 349.23, 'F#': 369.99, 'G': 392.00,
        'G#': 415.30, 'A': 440.00, 'A#': 466.16, 'B': 493.88
    }
    
    SOUND_MODES = ['mono', 'stereo', 'wide stereo']
    
    def __init__(self):
        self.oscillators = []
        self.add_oscillator()  # Start with one oscillator
        self.sound_mode = 'stereo'
        pygame.mixer.init(frequency=44100, size=-16, channels=2)
        self.load_chord_definitions()
    
    def add_oscillator(self):
        """Add a new oscillator"""
        self.oscillators.append(Oscillator())
        return len(self.oscillators) - 1  # Return new oscillator index
    
    def remove_oscillator(self, index):
        """Remove an oscillator by index"""
        if len(self.oscillators) > 1:  # Keep at least one oscillator
            del self.oscillators[index]
            return True
        return False
    
    def load_chord_definitions(self):
        """Load chord definitions from JSON file"""
        try:
            with open('chord_definitions.json', 'r') as f:
                self.chord_definitions = json.load(f)
        except FileNotFoundError:
            # If file doesn't exist, create it with default chords
            self.chord_definitions = {
                "major": {"intervals": [0, 4, 7], "description": "Major triad (1, 3, 5)"},
                "minor": {"intervals": [0, 3, 7], "description": "Minor triad (1, ♭3, 5)"},
                "diminished": {"intervals": [0, 3, 6], "description": "Diminished triad (1, ♭3, ♭5)"},
                "augmented": {"intervals": [0, 4, 8], "description": "Augmented triad (1, 3, #5)"},
                "major7": {"intervals": [0, 4, 7, 11], "description": "Major seventh (1, 3, 5, 7)"},
                "minor7": {"intervals": [0, 3, 7, 10], "description": "Minor seventh (1, ♭3, 5, ♭7)"},
                "dominant7": {"intervals": [0, 4, 7, 10], "description": "Dominant seventh (1, 3, 5, ♭7)"},
                "sus2": {"intervals": [0, 2, 7], "description": "Suspended 2nd (1, 2, 5)"},
                "sus4": {"intervals": [0, 5, 7], "description": "Suspended 4th (1, 4, 5)"}
            }
            self.save_chord_definitions()
    
    def save_chord_definitions(self):
        """Save chord definitions to JSON file"""
        with open('chord_definitions.json', 'w') as f:
            json.dump(self.chord_definitions, f, indent=4)
    
    def add_chord_definition(self, name: str, intervals: List[int], description: str = ""):
        """Add a new chord definition"""
        self.chord_definitions[name] = {
            "intervals": intervals,
            "description": description
        }
        self.save_chord_definitions()
    
    def get_chord_description(self, chord_type: str) -> str:
        """Get the description of a chord type"""
        return self.chord_definitions.get(chord_type, {}).get("description", "")
    
    def generate_chord(self, master_octave: int, notes_data: List[Dict], duration_ms: int = 1000, fade_out_ms: int = 100, master_volume: float = 1.0, master_mute: bool = False, soloed_osc_idx: int = -1) -> np.ndarray:
        """ Generates a chord based on a list of notes, each with a pitch and octave adjustment. """
        
        if not notes_data:
            # raise ValueError("No notes data provided to generate chord")
            return np.array([]) # Return empty if no notes

        # Add extra time for fade out
        total_duration = duration_ms + fade_out_ms
        final_samples = np.zeros(int(44100 * total_duration / 1000))
        # active_oscillators_count = 0 # Renamed to avoid conflict with outer scope var if any

        # Determine active oscillators based on solo state
        if soloed_osc_idx != -1 and soloed_osc_idx < len(self.oscillators) and self.oscillators[soloed_osc_idx].enabled:
            effective_oscs = [self.oscillators[soloed_osc_idx]]
        else: # No solo or soloed oscillator is disabled, consider all enabled oscillators
            effective_oscs = [osc for osc in self.oscillators if osc.enabled]

        if not effective_oscs:
            return final_samples # Return silence if no active oscillators
            
        num_notes_to_play = len(notes_data)
        combined_note_samples = np.zeros(int(44100 * total_duration / 1000))

        for i, note_info in enumerate(notes_data):
            osc_idx_for_note = i % len(effective_oscs)
            osc = effective_oscs[osc_idx_for_note]
            
            note_pitch_str = note_info['pitch']
            note_octave_adjust = note_info['octave_adjust']

            if note_pitch_str not in self.NOTE_FREQUENCIES:
                print(f"Warning: Pitch {note_pitch_str} not found in NOTE_FREQUENCIES. Skipping note.")
                continue

            base_freq_for_note = self.NOTE_FREQUENCIES[note_pitch_str]
            # The octave adjustment is relative to the master_octave, which itself is relative to octave 4 as the baseline for NOTE_FREQUENCIES
            current_note_freq = base_freq_for_note * (2 ** (master_octave - 4 + note_octave_adjust))
            
            original_osc_freq = osc.frequency # Store original to restore if osc is shared
            osc.frequency = current_note_freq
            note_specific_samples = osc.generate_samples(total_duration)
            osc.frequency = original_osc_freq # Restore if necessary
            
            combined_note_samples += note_specific_samples
            # active_oscillators_count +=1 # This isn't quite right for normalization if notes > oscs

        # Normalize based on the number of notes or actual peak, careful here
        if num_notes_to_play > 0: # Or if active_oscillators_count > 0
            # A simple approach: average, then clip. More advanced would be dynamic compression or limiting.
            # final_samples = np.clip(combined_note_samples / num_notes_to_play, -1, 1)
            # Better normalization: by actual peak of the combined signal if it exceeds 1.0
            max_val = np.max(np.abs(combined_note_samples))
            if max_val > 1.0:
                final_samples = combined_note_samples / max_val
            else:
                final_samples = combined_note_samples
        else:
            final_samples = combined_note_samples # Should be zeros if no notes played
            
        # Apply fade out at the end
        if fade_out_ms > 0:
            fade_samples_count = int(44100 * fade_out_ms / 1000)
            fade_curve = np.linspace(1, 0, fade_samples_count)
            if len(final_samples) >= fade_samples_count:
                 final_samples[-fade_samples_count:] *= fade_curve
        
        # Convert to stereo
        stereo_output_samples = np.zeros((len(final_samples), 2)) # Initialize stereo array
        if self.sound_mode == 'mono':
            stereo_output_samples = np.column_stack((final_samples, final_samples))
        elif self.sound_mode == 'wide stereo':
            phase_shift = int(44100 * 0.002)
            right_channel = np.pad(final_samples, (phase_shift, 0), 'constant')[:-phase_shift]
            if len(final_samples) == len(right_channel):
                stereo_output_samples = np.column_stack((final_samples, right_channel))
            else: # Fallback if padding/slicing results in mismatch
                stereo_output_samples = np.column_stack((final_samples, final_samples))
        else: # 'stereo' - TODO: Implement proper panning per oscillator contribution if possible
            stereo_output_samples = np.column_stack((final_samples, final_samples))

        # Apply Master Volume and Mute
        if master_mute:
            stereo_output_samples = np.zeros_like(stereo_output_samples)
        else:
            stereo_output_samples *= master_volume
            
        return stereo_output_samples

class ChordGeneratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Automatic Shallot v0.5 - Custom Chords")
        self.root.geometry("600x480") # Set initial window size
        
        # Set minimum window size and configure grid
        self.root.minsize(600, 480) # Set minsize to the target
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Create main container
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        
        # Configure main frame grid
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=1)
        
        # Initialize visualization variables
        self.waveform_plots = {}
        self.waveform_canvases = {}
        
        # Initialize audio components
        self.chord_generator = ChordGenerator()
        # self.sequencer = Sequencer() # Sequencer logic now in SequencerUI
        
        # Audio setup
        self.pygame = pygame # Make pygame accessible for SequencerUI if it needs it
        self.np = np # Make numpy accessible for SequencerUI if it needs it
        pygame.init()
        pygame.mixer.init(frequency=44100, size=-16, channels=2)
        
        # Initialize control variables
        self.init_control_variables()
        
        # Setup GUI
        self.setup_gui()
        
        # Update initial waveform previews
        for i in range(len(self.chord_generator.oscillators)):
            self.update_waveform_preview(i)
    
    def init_control_variables(self):
        """Initialize all control variables"""
        self.wave_vars = []
        self.amp_vars = []
        self.detune_vars = []
        self.attack_vars = []
        self.decay_vars = []
        self.sustain_vars = []
        self.release_vars = []
        self.cutoff_vars = []
        self.resonance_vars = []
        self.pan_vars = []
        self.enabled_vars = []
        self.solo_vars = [] # For oscillator solo states
        self.duration_var = tk.DoubleVar(value=1.0)
        self.octave_var = tk.IntVar(value=4)
        self.note_octave_vars = []
        self.note_pitch_vars = []
        self.custom_chord_notes_ui_frames = []
        self.custom_chord_notes_data = []
        self.custom_notes_canvas = None # For horizontal scrolling of notes
        self.custom_notes_content_frame = None # Frame inside the canvas
        
        self.init_oscillator_controls(0)  # Initialize first oscillator
        
        self.root_var = tk.StringVar(value='C')
        self.root_var.trace_add('write', self.on_root_or_type_change) # Centralized handler
        self.type_var = tk.StringVar(value='major')
        self.type_var.trace_add('write', self.on_root_or_type_change) # Centralized handler
        # self.bpm_var = tk.IntVar(value=self.sequencer.bpm) # Moved to SequencerUI
        self.sound_mode_var = tk.StringVar(value=self.chord_generator.sound_mode)
        
        # Master Output Controls
        self.master_volume_var = tk.DoubleVar(value=0.8) # Default to 0.8
        self.master_mute_var = tk.BooleanVar(value=False)
    
    def init_oscillator_controls(self, index):
        """Initialize control variables for a single oscillator"""
        osc = self.chord_generator.oscillators[index]
        self.wave_vars.append(tk.StringVar(value=osc.waveform))
        self.amp_vars.append(tk.DoubleVar(value=osc.amplitude))
        self.detune_vars.append(tk.DoubleVar(value=osc.detune))
        self.attack_vars.append(tk.DoubleVar(value=osc.attack))
        self.decay_vars.append(tk.DoubleVar(value=osc.decay))
        self.sustain_vars.append(tk.DoubleVar(value=osc.sustain))
        self.release_vars.append(tk.DoubleVar(value=osc.release))
        self.cutoff_vars.append(tk.DoubleVar(value=osc.filter_cutoff))
        self.resonance_vars.append(tk.DoubleVar(value=osc.filter_resonance))
        self.pan_vars.append(tk.DoubleVar(value=osc.pan))
        self.enabled_vars.append(tk.BooleanVar(value=osc.enabled))
        self.solo_vars.append(tk.BooleanVar(value=False)) # Initialize solo state for new osc
    
    def setup_gui(self):
        """Setup the main GUI layout"""
        # Left side - Oscillators and Sound Controls
        left_frame = ttk.Frame(self.main_frame)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        left_frame.grid_columnconfigure(0, weight=1)
        left_frame.grid_rowconfigure(1, weight=1)  # Make oscillator canvas expandable
        
        # Sound Mode, Master Output and Export Frame
        top_controls_outer_frame = ttk.Frame(left_frame) # New outer frame for layout
        top_controls_outer_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        top_controls_outer_frame.grid_columnconfigure(0, weight=1) # Mode frame takes space
        top_controls_outer_frame.grid_columnconfigure(1, weight=1) # Master output frame takes space
        top_controls_outer_frame.grid_columnconfigure(2, weight=0) # Export button fixed size

        mode_frame = ttk.LabelFrame(top_controls_outer_frame, text="Sound Mode", padding="5")
        mode_frame.grid(row=0, column=0, sticky="ew", padx=(0,5), pady=2)
        # mode_frame.grid_columnconfigure(1, weight=1) # Already configured by parent grid
        
        ttk.Label(mode_frame, text="Mode:").grid(row=0, column=0, padx=5)
        mode_menu = ttk.Combobox(mode_frame, textvariable=self.sound_mode_var,
                                values=ChordGenerator.SOUND_MODES, width=10) # Adjusted width
        mode_menu.grid(row=0, column=1, padx=5, sticky="ew")
        mode_menu.bind('<<ComboboxSelected>>', self.update_sound_mode)

        # Master Output Frame
        master_output_frame = ttk.LabelFrame(top_controls_outer_frame, text="Master Output", padding="5")
        master_output_frame.grid(row=0, column=1, sticky="ew", padx=5, pady=2)
        master_output_frame.grid_columnconfigure(1, weight=1) # Scale expands

        ttk.Label(master_output_frame, text="Vol:").grid(row=0, column=0, padx=2, pady=2, sticky="w")
        master_vol_scale = ttk.Scale(master_output_frame, from_=0, to=1, 
                                     variable=self.master_volume_var, 
                                     orient=tk.HORIZONTAL, length=100)
        master_vol_scale.grid(row=0, column=1, padx=2, pady=2, sticky="ew")

        master_mute_cb = ttk.Checkbutton(master_output_frame, text="Mute", 
                                       variable=self.master_mute_var)
        master_mute_cb.grid(row=0, column=2, padx=5, pady=2)
        
        export_btn = ttk.Button(top_controls_outer_frame, text="Export WAV", command=self.export_wav)
        export_btn.grid(row=0, column=2, padx=5, pady=2, sticky="e")
        
        # Create scrollable canvas for oscillators
        canvas = tk.Canvas(left_frame)
        scrollbar = ttk.Scrollbar(left_frame, orient="vertical", command=canvas.yview)
        self.oscillator_frame = ttk.Frame(canvas)
        
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack widgets
        scrollbar.grid(row=1, column=1, sticky="ns")
        canvas.grid(row=1, column=0, sticky="nsew", padx=(5, 0))
        
        # Create window in canvas
        canvas.create_window((0, 0), window=self.oscillator_frame, anchor="nw")
        
        # Add oscillator button
        add_btn = ttk.Button(left_frame, text="Add Oscillator", command=self.add_oscillator)
        add_btn.grid(row=2, column=0, columnspan=2, pady=5)
        
        # Configure scrolling
        self.oscillator_frame.bind("<Configure>", 
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        
        # Create initial oscillator
        self.create_oscillator_frame(self.oscillator_frame, 0)
        
        # Right side - Chord and Sequencer Controls
        right_frame = ttk.Frame(self.main_frame)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        right_frame.grid_rowconfigure(0, weight=0) # Chord frame, fixed size initially
        right_frame.grid_rowconfigure(1, weight=1) # Sequencer UI frame, takes remaining space

        # Chord Controls
        chord_frame = self.create_chord_frame(right_frame)
        chord_frame.grid(row=0, column=0, sticky="new", padx=5, pady=5) # Changed to new, removed ew
        
        # Sequencer UI - Instantiate SequencerUI here
        # The SequencerUI will build its components into a frame it creates inside right_frame
        # Or, we can pass a specific sub-frame of right_frame for it to populate.
        # Let's create a container for SequencerUI within right_frame.
        sequencer_ui_container = ttk.Frame(right_frame)
        sequencer_ui_container.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        self.sequencer_ui_instance = SequencerUI(sequencer_ui_container, self)
    
    def create_oscillator_frame(self, parent, index):
        """Create a frame for a single oscillator"""
        osc_frame = ttk.LabelFrame(parent, text=f"Oscillator {index + 1}", padding="2")
        osc_frame.grid(row=index, column=0, sticky="ew", padx=2, pady=1)
        osc_frame.grid_columnconfigure(1, weight=1)
        
        # Add remove button
        if len(self.chord_generator.oscillators) > 1:
            remove_btn = ttk.Button(osc_frame, text="X",
                                  command=lambda i=index: self.remove_oscillator(i),
                                  width=3)
            remove_btn.grid(row=0, column=2, padx=2, pady=1, sticky="ne")
        
        # Add waveform preview canvas
        self.create_waveform_preview(osc_frame, index)
        
        # Rest of the controls
        basic_frame = ttk.Frame(osc_frame)
        basic_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=1)
        
        # Enable/Disable checkbox
        enable_cb = ttk.Checkbutton(basic_frame, text="Enable",
                                  variable=self.enabled_vars[index],
                                  command=lambda i=index: self.update_oscillator(i))
        enable_cb.grid(row=0, column=0, padx=2)
        
        # Waveform with tooltip
        ttk.Label(basic_frame, text="Wave:").grid(row=0, column=1, padx=1)
        wave_menu = ttk.Combobox(basic_frame, textvariable=self.wave_vars[index],
                                values=list(self.chord_generator.oscillators[0].waveform_definitions.keys()),
                                width=10)
        wave_menu.grid(row=0, column=2, padx=1)
        wave_menu.bind('<<ComboboxSelected>>', lambda e, i=index: self.update_waveform_preview(i))
        
        # Create tooltip for waveform description
        self.create_waveform_tooltip(wave_menu, index)
        
        # Solo Button
        solo_cb = ttk.Checkbutton(basic_frame, text="Solo", 
                                  variable=self.solo_vars[index],
                                  command=lambda i=index: self.toggle_solo(i))
        solo_cb.grid(row=0, column=3, padx=2) # Next to Enable
        
        # Amplitude - Shifted one column to the right due to Solo button
        ttk.Label(basic_frame, text="Amp:").grid(row=0, column=4, padx=1) # Was column 3
        amp_scale = ttk.Scale(basic_frame, from_=0, to=1,
                             variable=self.amp_vars[index],
                             orient=tk.HORIZONTAL, length=80)
        amp_scale.grid(row=0, column=5, padx=1) # Was column 4
        
        # Detune - Shifted one column to the right
        ttk.Label(basic_frame, text="Det:").grid(row=0, column=6, padx=1) # Was column 5
        detune_scale = ttk.Scale(basic_frame, from_=-100, to=100,
                                variable=self.detune_vars[index],
                                orient=tk.HORIZONTAL, length=80)
        detune_scale.grid(row=0, column=7, padx=1) # Was column 6
        
        # ADSR Controls
        adsr_frame = ttk.LabelFrame(osc_frame, text="ADSR", padding="1")
        adsr_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=1)
        
        adsr_labels = ['A', 'D', 'S', 'R']
        adsr_vars = [self.attack_vars[index], self.decay_vars[index],
                    self.sustain_vars[index], self.release_vars[index]]
        
        for i, (label, var) in enumerate(zip(adsr_labels, adsr_vars)):
            ttk.Label(adsr_frame, text=label).grid(row=0, column=i*2, padx=1)
            ttk.Scale(adsr_frame, from_=0, to=2 if label != 'S' else 1,
                     variable=var, orient=tk.HORIZONTAL, length=60).grid(row=0, column=i*2+1, padx=1)
        
        # Filter and Pan Controls
        filter_frame = ttk.Frame(osc_frame)
        filter_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=1)
        
        ttk.Label(filter_frame, text="Cut:").grid(row=0, column=0, padx=1)
        ttk.Scale(filter_frame, from_=0, to=1,
                 variable=self.cutoff_vars[index],
                 orient=tk.HORIZONTAL, length=80).grid(row=0, column=1, padx=1)
        
        ttk.Label(filter_frame, text="Res:").grid(row=0, column=2, padx=1)
        ttk.Scale(filter_frame, from_=0, to=1,
                 variable=self.resonance_vars[index],
                 orient=tk.HORIZONTAL, length=80).grid(row=0, column=3, padx=1)
        
        ttk.Label(filter_frame, text="Pan:").grid(row=0, column=4, padx=1)
        ttk.Scale(filter_frame, from_=0, to=1,
                 variable=self.pan_vars[index],
                 orient=tk.HORIZONTAL, length=80).grid(row=0, column=5, padx=1)
        
        # Bind all control changes
        self.bind_oscillator_controls(index)
    
    def create_waveform_preview(self, parent, index):
        """Create a small waveform preview canvas"""
        # Create matplotlib figure
        fig = Figure(figsize=(3, 1), dpi=100)
        ax = fig.add_subplot(111)
        
        # Remove axes
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Create initial waveform plot
        t = np.linspace(0, 1, 1000)
        y = np.sin(2 * np.pi * t)
        self.waveform_plots[index] = ax.plot(t, y)[0]
        
        # Set axis limits
        ax.set_ylim(-1.1, 1.1)
        
        # Create canvas
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0, columnspan=2, padx=2, pady=1, sticky="ew")
        
        # Store canvas for updates
        self.waveform_canvases[index] = canvas
    
    def update_waveform_preview(self, index):
        """Update the waveform preview for an oscillator"""
        waveform = self.wave_vars[index].get()
        osc = self.chord_generator.oscillators[index]
        wave_def = osc.waveform_definitions.get(waveform, osc.waveform_definitions["sine"])
        
        # Generate preview waveform
        t = np.linspace(0, 1, 1000)
        if wave_def["type"] == "basic":
            if waveform == 'sine':
                y = np.sin(2 * np.pi * t)
            elif waveform == 'square':
                y = np.sign(np.sin(2 * np.pi * t))
            elif waveform == 'sawtooth':
                y = 2 * (t - np.floor(0.5 + t))
            else:  # triangle
                y = 2 * np.abs(2 * (t - np.floor(0.5 + t))) - 1
        else:
            y = np.zeros_like(t)
            for harmonic in wave_def["harmonics"]:
                y += harmonic["amplitude"] * np.sin(2 * np.pi * harmonic["frequency"] * t)
            if len(wave_def["harmonics"]) > 0:
                y = y / max(abs(y.min()), abs(y.max()))
        
        # Update plot
        self.waveform_plots[index].set_ydata(y)
        self.waveform_canvases[index].draw()
    
    def create_waveform_tooltip(self, widget, index):
        """Create tooltip for waveform description"""
        def show_tooltip(event):
            waveform = self.wave_vars[index].get()
            description = self.chord_generator.oscillators[index].waveform_definitions[waveform]["description"]
            
            x = widget.winfo_rootx() + 25
            y = widget.winfo_rooty() + 20
            
            tooltip = tk.Toplevel(widget)
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{x}+{y}")
            
            label = ttk.Label(tooltip, text=description,
                            justify=tk.LEFT, background="#ffffe0",
                            relief=tk.SOLID, borderwidth=1)
            label.pack()
            
            def hide_tooltip():
                tooltip.destroy()
            
            widget.tooltip = tooltip
            widget.after(2000, hide_tooltip)
        
        def hide_tooltip(event):
            if hasattr(widget, 'tooltip'):
                widget.tooltip.destroy()
                del widget.tooltip
        
        widget.bind('<Enter>', show_tooltip)
        widget.bind('<Leave>', hide_tooltip)
    
    def create_chord_frame(self, parent):
        """Create the chord control frame"""
        chord_frame = ttk.LabelFrame(parent, text="Chord Settings", padding="2")
        chord_frame.grid_columnconfigure(1, weight=1)
        chord_frame.grid_columnconfigure(3, weight=1)
        
        # Root note and chord type selection
        ttk.Label(chord_frame, text="Root:").grid(row=0, column=0, padx=2, pady=1)
        root_menu = ttk.Combobox(chord_frame, textvariable=self.root_var,
                                values=list(ChordGenerator.NOTE_FREQUENCIES.keys()),
                                width=3)
        root_menu.grid(row=0, column=1, padx=2, pady=1)
        
        # Octave control
        ttk.Label(chord_frame, text="Oct:").grid(row=0, column=2, padx=2, pady=1)
        octave_spin = ttk.Spinbox(chord_frame, from_=0, to=8,
                                 textvariable=self.octave_var,
                                 width=2)
        octave_spin.grid(row=0, column=3, padx=2, pady=1)
        
        # Chord type with tooltip
        ttk.Label(chord_frame, text="Type:").grid(row=0, column=4, padx=2, pady=1)
        self.type_menu = ttk.Combobox(chord_frame, textvariable=self.type_var,
                                     values=list(self.chord_generator.chord_definitions.keys()),
                                     width=10)
        self.type_menu.grid(row=0, column=5, padx=2, pady=1)
        
        # Create tooltip for chord description
        self.chord_tooltip = None
        self.type_menu.bind('<<ComboboxSelected>>', self.update_chord_tooltip)
        self.type_menu.bind('<Enter>', self.show_chord_tooltip)
        self.type_menu.bind('<Leave>', self.hide_chord_tooltip)
        
        # Duration control
        ttk.Label(chord_frame, text="Beats:").grid(row=0, column=6, padx=2, pady=1)
        duration_spin = ttk.Spinbox(chord_frame, from_=0.25, to=8,
                                  increment=0.25,
                                  textvariable=self.duration_var,
                                  width=4)
        duration_spin.grid(row=0, column=7, padx=2, pady=1)
        
        # Frame for individual note octave controls (renamed and enhanced)
        self.custom_notes_frame = ttk.LabelFrame(chord_frame, text="Chord Notes", padding="2")
        self.custom_notes_frame.grid(row=1, column=0, columnspan=8, sticky="ew", padx=2, pady=5)
        
        self.update_custom_chord_note_ui() # Initial setup

        # Add Note Button
        add_note_btn = ttk.Button(chord_frame, text="Add Note (+)", command=self.add_note_to_custom_chord)
        add_note_btn.grid(row=2, column=0, columnspan=8, pady=5)

        return chord_frame
    
    def bind_oscillator_controls(self, index):
        """Bind all control changes for an oscillator"""
        self.wave_vars[index].trace_add('write', lambda *args: self.update_oscillator(index))
        self.amp_vars[index].trace_add('write', lambda *args: self.update_oscillator(index))
        self.detune_vars[index].trace_add('write', lambda *args: self.update_oscillator(index))
        self.attack_vars[index].trace_add('write', lambda *args: self.update_oscillator(index))
        self.decay_vars[index].trace_add('write', lambda *args: self.update_oscillator(index))
        self.sustain_vars[index].trace_add('write', lambda *args: self.update_oscillator(index))
        self.release_vars[index].trace_add('write', lambda *args: self.update_oscillator(index))
        self.cutoff_vars[index].trace_add('write', lambda *args: self.update_oscillator(index))
        self.resonance_vars[index].trace_add('write', lambda *args: self.update_oscillator(index))
        self.pan_vars[index].trace_add('write', lambda *args: self.update_oscillator(index))
        self.enabled_vars[index].trace_add('write', lambda *args: self.update_oscillator(index))
        self.solo_vars[index].trace_add('write', lambda *args: self.update_oscillator(index))

    def update_oscillator(self, index):
        """Update all oscillator parameters when controls change"""
        osc = self.chord_generator.oscillators[index]
        osc.enabled = self.enabled_vars[index].get()
        osc.waveform = self.wave_vars[index].get()
        osc.amplitude = self.amp_vars[index].get()
        osc.detune = self.detune_vars[index].get()
        osc.attack = self.attack_vars[index].get()
        osc.decay = self.decay_vars[index].get()
        osc.sustain = self.sustain_vars[index].get()
        osc.release = self.release_vars[index].get()
        osc.filter_cutoff = self.cutoff_vars[index].get()
        osc.filter_resonance = self.resonance_vars[index].get()
        osc.pan = self.pan_vars[index].get()
    
    def preview_chord(self):
        """Generate and play the current chord"""
        try:
            # notes_data is now self.custom_chord_notes_data
            current_notes_data = self.get_current_custom_notes_for_sound()
            if not current_notes_data:
                print("Preview: No notes to play.")
                return

            master_vol = self.master_volume_var.get()
            master_mute_status = self.master_mute_var.get()
            
            current_solo_idx = -1
            for idx, solo_var in enumerate(self.solo_vars):
                if solo_var.get():
                    current_solo_idx = idx
                    break

            samples = self.chord_generator.generate_chord(
                master_octave=self.octave_var.get(),
                notes_data=current_notes_data,
                duration_ms=1000,
                master_volume=master_vol,
                master_mute=master_mute_status,
                soloed_osc_idx=current_solo_idx
            )
            
            if samples.size == 0:
                print("Preview: Generated empty samples. This should not happen.")
                return
            
            # Convert samples to pygame sound
            sound_array = (samples * 32767).astype(self.np.int16)
            sound = self.pygame.sndarray.make_sound(sound_array)
            
            # Play the sound
            self.pygame.mixer.stop()
            sound.play()
            
        except Exception as e:
            tk.messagebox.showerror("Error", str(e))
    
    def add_to_sequence(self):
        """Add current chord to the sequence"""
        current_notes_data = self.get_current_custom_notes_for_sound()
        if not current_notes_data:
            messagebox.showwarning("Add to Sequence", "Cannot add an empty chord to sequence.")
            return

        # self.sequencer.sequence.append(step) # Old way
        self.sequencer_ui_instance.add_step(
            master_octave=self.octave_var.get(), 
            duration=self.duration_var.get(), 
            notes_data=current_notes_data
        )
        self.sequencer_ui_instance.update_sequence_display()
    
    def update_sequence_display(self):
        """Update the sequence display text"""
        self.sequencer_ui_instance.update_sequence_display()
    
    def play_sequence(self):
        """Start playing the sequence"""
        if not self.sequencer_ui_instance.sequence:
            return
            
        self.sequencer_ui_instance.play_sequence()
    
    def stop_sequence(self):
        """Stop the sequence playback"""
        self.sequencer_ui_instance.stop_sequence()
    
    def clear_sequence(self):
        """Clear the current sequence"""
        self.sequencer_ui_instance.clear_sequence()
        self.update_sequence_display()
    
    def update_sound_mode(self, *args):
        """Update the sound mode when the control changes"""
        mode = self.sound_mode_var.get()
        if mode in ChordGenerator.SOUND_MODES:
            self.chord_generator.sound_mode = mode
    
    def export_wav(self):
        """Export the current chord as a WAV file"""
        try:
            current_notes_data = self.get_current_custom_notes_for_sound()
            if not current_notes_data:
                messagebox.showerror("Error", "Cannot export an empty chord.")
                return

            master_vol = self.master_volume_var.get()
            
            current_solo_idx_export = -1 

            samples = self.chord_generator.generate_chord(
                master_octave=self.octave_var.get(),
                notes_data=current_notes_data,
                duration_ms=2000,  # 2 seconds for export
                master_volume=master_vol,
                master_mute=False, 
                soloed_osc_idx=current_solo_idx_export
            )
            
            if samples.size == 0:
                print("Export: Generated empty samples. This should not happen.")
                return
            
            # Convert to 16-bit integer samples
            samples = (samples * 32767).astype(self.np.int16)
            
            # Create WAV file
            from scipy.io import wavfile
            filename = f"chord_{self.root_var.get()}_{self.type_var.get()}.wav"
            wavfile.write(filename, 44100, samples)
            
            tk.messagebox.showinfo("Success", f"Exported chord to {filename}")
            
        except Exception as e:
            tk.messagebox.showerror("Error", f"Failed to export WAV: {str(e)}")
    
    def update_chord_tooltip(self, event=None):
        """Update the chord description tooltip"""
        chord_type = self.type_var.get()
        description = self.chord_generator.get_chord_description(chord_type)
        if description:
            self.chord_tooltip_text = description
        # UI update logic is now handled by on_root_or_type_change
    
    def show_chord_tooltip(self, event=None):
        """Show the chord description tooltip"""
        if hasattr(self, 'chord_tooltip_text'):
            x, y, _, _ = self.type_menu.bbox("insert")
            x += self.type_menu.winfo_rootx() + 25
            y += self.type_menu.winfo_rooty() + 20
            
            # Destroy existing tooltip if it exists
            self.hide_chord_tooltip()
            
            # Create new tooltip
            self.chord_tooltip = tk.Toplevel(self.type_menu)
            self.chord_tooltip.wm_overrideredirect(True)
            self.chord_tooltip.wm_geometry(f"+{x}+{y}")
            
            label = ttk.Label(self.chord_tooltip, text=self.chord_tooltip_text,
                            justify=tk.LEFT, background="#ffffe0", relief=tk.SOLID, borderwidth=1)
            label.pack()
    
    def hide_chord_tooltip(self, event=None):
        """Hide the chord description tooltip"""
        if self.chord_tooltip:
            self.chord_tooltip.destroy()
            self.chord_tooltip = None
    
    def refresh_chord_types(self):
        """Refresh the list of available chord types"""
        current_chord_types = list(self.chord_generator.chord_definitions.keys())
        if "-- Custom --" not in current_chord_types:
             current_chord_types.insert(0, "-- Custom --") # Add custom option if not present
        self.type_menu['values'] = current_chord_types
        # self.update_custom_chord_note_ui() # This might be too early or redundant here, handled by update_chord_tooltip

    def _get_note_name(self, root_note_str: str, interval_semitones: int) -> str:
        """Calculates the note name given a root note and an interval in semitones."""
        note_names = list(ChordGenerator.NOTE_FREQUENCIES.keys()) # C, C#, D, ...
        try:
            root_index = note_names.index(root_note_str)
        except ValueError:
            return note_names[0] # Default to C if root_note_str is somehow invalid
        
        note_index = (root_index + interval_semitones) % 12
        return note_names[note_index]

    def populate_custom_chord_from_preset(self, chord_type_name: str):
        """Populates self.custom_chord_notes_data based on a selected preset."""
        if not chord_type_name or chord_type_name not in self.chord_generator.chord_definitions:
            # If invalid type, or if it was already custom, perhaps clear or use a default like single root note
            if not self.custom_chord_notes_data: # Only reset if truly empty, to avoid wiping user edits if called unexpectedly
                 self.custom_chord_notes_data = [{'pitch': self.root_var.get(), 'octave_adjust': 0}]
            return

        root_note = self.root_var.get()
        intervals = self.chord_generator.chord_definitions[chord_type_name]["intervals"]
        self.custom_chord_notes_data = []
        for interval in intervals:
            note_name = self._get_note_name(root_note, interval)
            self.custom_chord_notes_data.append({'pitch': note_name, 'octave_adjust': 0})
        # No self.type_var.set("-- Custom --") here; that's for when user *manually* edits.

    def update_custom_chord_note_ui(self):
        """Dynamically create/update UI controls for each note in self.custom_chord_notes_data, with horizontal scrolling."""
        
        # If canvas and content frame don't exist, create them within self.custom_notes_frame
        if not self.custom_notes_canvas:
            self.custom_notes_canvas = tk.Canvas(self.custom_notes_frame, height=110) # Explicit height for the canvas
            h_scrollbar = ttk.Scrollbar(self.custom_notes_frame, orient=tk.HORIZONTAL, command=self.custom_notes_canvas.xview)
            self.custom_notes_canvas.configure(xscrollcommand=h_scrollbar.set)
            
            self.custom_notes_content_frame = ttk.Frame(self.custom_notes_canvas)
            self.custom_notes_canvas.create_window((0, 0), window=self.custom_notes_content_frame, anchor="nw")

            self.custom_notes_canvas.pack(side=tk.TOP, fill=tk.X, expand=True, padx=2, pady=2)
            h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X, padx=2, pady=2)
            
            self.custom_notes_content_frame.bind("<Configure>", 
                lambda e: self.custom_notes_canvas.configure(scrollregion=self.custom_notes_canvas.bbox("all")))

        # Clear existing individual note UI frames from the content_frame
        for frame_widget in self.custom_chord_notes_ui_frames:
            frame_widget.destroy()
        self.custom_chord_notes_ui_frames = []
        self.note_pitch_vars = []
        self.note_octave_vars = []

        all_note_names = list(ChordGenerator.NOTE_FREQUENCIES.keys())
        
        for i, note_data in enumerate(self.custom_chord_notes_data):
            # Each note_ui_frame is now created inside self.custom_notes_content_frame
            note_ui_frame = ttk.Frame(self.custom_notes_content_frame, borderwidth=1, relief="sunken", padding=2)
            self.custom_chord_notes_ui_frames.append(note_ui_frame)
            note_ui_frame.pack(side=tk.LEFT, padx=3, pady=3, fill=tk.Y) # Pack horizontally

            ttk.Label(note_ui_frame, text=f"N{i+1}").pack(side=tk.TOP, pady=(0,2)) # Compact label
            
            # Pitch Combobox
            pitch_var = tk.StringVar(value=note_data['pitch'])
            self.note_pitch_vars.append(pitch_var)
            pitch_combo = ttk.Combobox(note_ui_frame, textvariable=pitch_var, values=all_note_names, width=4)
            pitch_combo.pack(side=tk.TOP, pady=2)
            pitch_combo.bind('<<ComboboxSelected>>', lambda e, idx=i, pv=pitch_var: self.on_custom_note_param_change(idx, 'pitch', pv.get()))
            pitch_combo.bind('<FocusOut>', lambda e, idx=i, pv=pitch_var: self.on_custom_note_param_change(idx, 'pitch', pv.get()))

            # Octave Adjustment Spinbox
            octave_var = tk.IntVar(value=note_data['octave_adjust'])
            self.note_octave_vars.append(octave_var)
            octave_spin = ttk.Spinbox(note_ui_frame, from_=-3, to=3, increment=1, textvariable=octave_var, width=3)
            octave_spin.pack(side=tk.TOP, pady=2)
            octave_spin.configure(command=lambda v=octave_var, idx=i: self.on_custom_note_param_change(idx, 'octave_adjust', v.get()))
            # For spinbox, direct command is usually enough, but FocusOut and Return can be added for robustness if needed
            octave_spin.bind('<FocusOut>', lambda e, v=octave_var, idx=i: self.on_custom_note_param_change(idx, 'octave_adjust', v.get()))
            octave_spin.bind('<Return>', lambda e, v=octave_var, idx=i: self.on_custom_note_param_change(idx, 'octave_adjust', v.get()))

            # Remove Note Button
            remove_btn = ttk.Button(note_ui_frame, text="-", width=2, command=lambda idx=i: self.remove_note_from_custom_chord(idx))
            remove_btn.pack(side=tk.TOP, pady=2)
        
        # After repopulating, update the scrollregion again
        self.custom_notes_content_frame.update_idletasks() # Ensure layout is calculated
        self.custom_notes_canvas.configure(scrollregion=self.custom_notes_canvas.bbox("all"))

    def on_custom_note_param_change(self, note_idx: int, param_type: str, new_value):
        """Called when a note's pitch or octave adjustment is changed in the UI."""
        if note_idx < len(self.custom_chord_notes_data):
            # Ensure the new_value for octave_adjust is an int if coming from spinbox command
            if param_type == 'octave_adjust':
                try:
                    new_value = int(new_value)
                except ValueError:
                    # Revert to old value or a default if conversion fails
                    # For now, let's try to keep the old value if possible
                    # This might need access to the tk.IntVar directly to reset if truly invalid
                    # Or simply prevent non-integer input in Spinbox if possible, though command gives string.
                    print(f"Warning: Invalid octave adjustment value '{new_value}'")
                    return # Prevent update with bad value

            if param_type == 'pitch':
                self.custom_chord_notes_data[note_idx]['pitch'] = str(new_value)
            elif param_type == 'octave_adjust':
                 self.custom_chord_notes_data[note_idx]['octave_adjust'] = new_value # Already int
            
            if self.type_var.get() != "-- Custom --":
                self.type_var.set("-- Custom --")
            # print(f"Updated custom_chord_notes_data[{note_idx}]: {self.custom_chord_notes_data[note_idx]}") # For debugging

    def add_note_to_custom_chord(self):
        """Adds a new default note to the custom chord structure and updates UI."""
        new_note_pitch = self.root_var.get() # Default to current root note
        new_note_octave_adjust = 0
        self.custom_chord_notes_data.append({'pitch': new_note_pitch, 'octave_adjust': new_note_octave_adjust})
        
        if self.type_var.get() != "-- Custom --":
            self.type_var.set("-- Custom --")
        self.update_custom_chord_note_ui()

    def remove_note_from_custom_chord(self, note_idx: int):
        """Removes a note from the custom chord structure and updates UI."""
        if 0 <= note_idx < len(self.custom_chord_notes_data):
            del self.custom_chord_notes_data[note_idx]
            if self.type_var.get() != "-- Custom --":
                self.type_var.set("-- Custom --")
            self.update_custom_chord_note_ui()
            if not self.custom_chord_notes_data: # If all notes removed, add a default one
                self.add_note_to_custom_chord()

    def add_oscillator(self):
        """Add a new oscillator"""
        index = self.chord_generator.add_oscillator()
        self.init_oscillator_controls(index)
        self.create_oscillator_frame(self.oscillator_frame, index)
        self.update_waveform_preview(index)
    
    def remove_oscillator(self, index):
        """Remove an oscillator"""
        if self.chord_generator.remove_oscillator(index):
            # Remove control variables
            for var_list in [self.wave_vars, self.amp_vars, self.detune_vars,
                           self.attack_vars, self.decay_vars, self.sustain_vars,
                           self.release_vars, self.cutoff_vars, self.resonance_vars,
                           self.pan_vars, self.enabled_vars, self.solo_vars]:
                if index < len(var_list):
                    del var_list[index]
            
            # Remove visualization
            if index in self.waveform_plots:
                del self.waveform_plots[index]
            if index in self.waveform_canvases:
                del self.waveform_canvases[index]
            
            # Destroy frame and update GUI
            for widget in self.oscillator_frame.winfo_children():
                widget.destroy()
            
            # Recreate remaining oscillators
            for i in range(len(self.chord_generator.oscillators)):
                self.create_oscillator_frame(self.oscillator_frame, i)
                self.update_waveform_preview(i)

    def on_root_or_type_change(self, *args):
        """Handles changes to root note or chord type, updating custom chord data and UI."""
        chord_type = self.type_var.get()
        if chord_type != "-- Custom --":
            self.populate_custom_chord_from_preset(chord_type)
        # Always update the UI to reflect the current state of custom_chord_notes_data
        self.update_custom_chord_note_ui()

    def get_current_custom_notes_for_sound(self) -> List[Dict]:
        """Helper to retrieve the current custom notes data, ensuring values are correct types."""
        # This method can be expanded to do any final validation or transformation if needed
        # For now, it just returns the data as is, assuming UI keeps it fairly clean.
        return [note.copy() for note in self.custom_chord_notes_data] # Return a copy

    def toggle_solo(self, soloed_index):
        """Handle solo button clicks. Implements single solo mode."""
        # This method is called AFTER the checkbutton's variable (self.solo_vars[soloed_index]) has changed state.
        is_activating_solo = self.solo_vars[soloed_index].get()

        for i, var in enumerate(self.solo_vars):
            if i == soloed_index:
                pass 
            else:
                if is_activating_solo: 
                    var.set(False) 
        
        # Sound might need an immediate update if a preview is desired on solo click
        # For now, sound generation functions will read the current solo states when called.

if __name__ == "__main__":
    root = tk.Tk()
    app = ChordGeneratorApp(root)
    root.mainloop() 
