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
from oscillator import Oscillator # Import the Oscillator class
from oscillator_renamer import OscillatorRenamerDialog # Import the new dialog

class ChordGenerator:
    NOTE_FREQUENCIES = {
        'C': 261.63, 'C#': 277.18, 'D': 293.66, 'D#': 311.13,
        'E': 329.63, 'F': 349.23, 'F#': 369.99, 'G': 392.00,
        'G#': 415.30, 'A': 440.00, 'A#': 466.16, 'B': 493.88
    }
    
    SOUND_MODES = ['mono', 'stereo', 'wide stereo']
    
    def __init__(self):
        self.oscillators = []
        self.sound_mode = 'stereo'
        pygame.mixer.init(frequency=44100, size=-16, channels=2)
        self.load_chord_definitions()
    
    def add_oscillator(self):
        """Add a new oscillator and assign a default name."""
        new_osc = Oscillator()
        # Name is set based on the current count + 1 BEFORE appending
        new_osc.name = f"osc{len(self.oscillators) + 1}"
        self.oscillators.append(new_osc)
        return len(self.oscillators) - 1 # Return index of the newly added oscillator
    
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
        """ Generates a chord based on a list of notes, each with a pitch, octave adjustment, and oscillator assignment. """
        
        if not notes_data:
            return np.array([]) 

        total_duration = duration_ms + fade_out_ms
        final_samples_shape = int(44100 * total_duration / 1000)
        combined_note_samples = np.zeros(final_samples_shape)

        # Determine effective oscillators based on global solo and enabled states
        if soloed_osc_idx != -1 and soloed_osc_idx < len(self.oscillators) and self.oscillators[soloed_osc_idx].enabled:
            effective_oscs_for_master_all = [self.oscillators[soloed_osc_idx]]
        else: 
            effective_oscs_for_master_all = [osc for osc in self.oscillators if osc.enabled]

        if not self.oscillators: # If no oscillators at all (e.g. all removed)
             return np.zeros((final_samples_shape, 2)) if self.sound_mode != 'mono' else np.zeros(final_samples_shape)

        for note_info in notes_data:
            note_pitch_str = note_info['pitch']
            note_octave_adjust = note_info['octave_adjust']
            assigned_osc_idx_for_note = note_info.get('osc_idx', -1)

            if note_pitch_str not in self.NOTE_FREQUENCIES:
                print(f"Warning: Pitch {note_pitch_str} not found. Skipping note.")
                continue

            base_freq_for_note = self.NOTE_FREQUENCIES[note_pitch_str]
            current_note_freq = base_freq_for_note * (2 ** (master_octave - 4 + note_octave_adjust))
            
            note_samples_this_note = np.zeros(final_samples_shape)
            
            oscillators_to_use_for_this_note = []
            if assigned_osc_idx_for_note == -1: # Master/All
                oscillators_to_use_for_this_note = effective_oscs_for_master_all
            elif 0 <= assigned_osc_idx_for_note < len(self.oscillators):
                specific_osc = self.oscillators[assigned_osc_idx_for_note]
                if specific_osc.enabled: # Must be enabled
                    if soloed_osc_idx != -1: # Global solo is active
                        if soloed_osc_idx == assigned_osc_idx_for_note: # And this is the soloed one
                            oscillators_to_use_for_this_note.append(specific_osc)
                    else: # No global solo, so use this specific enabled osc
                        oscillators_to_use_for_this_note.append(specific_osc)
            
            if not oscillators_to_use_for_this_note:
                continue # No active oscillator for this note, skip to next note

            for osc in oscillators_to_use_for_this_note:
                original_osc_freq = osc.frequency 
                osc.frequency = current_note_freq
                generated_samples = osc.generate_samples(total_duration)
                osc.frequency = original_osc_freq 
                note_samples_this_note += generated_samples
            
            combined_note_samples += note_samples_this_note

        final_samples = combined_note_samples
        if final_samples.any():
            max_val = np.max(np.abs(final_samples))
            if max_val > 1.0:
                final_samples = final_samples / max_val
        
        if fade_out_ms > 0:
            fade_samples_count = int(44100 * fade_out_ms / 1000)
            fade_curve = np.linspace(1, 0, fade_samples_count)
            if len(final_samples) >= fade_samples_count:
                 final_samples[-fade_samples_count:] *= fade_curve
        
        stereo_output_samples = np.zeros((len(final_samples), 2))
        if self.sound_mode == 'mono':
            stereo_output_samples = np.column_stack((final_samples, final_samples))
        elif self.sound_mode == 'wide stereo':
            phase_shift = int(44100 * 0.002)
            right_channel = np.pad(final_samples, (phase_shift, 0), 'constant')[:-phase_shift]
            if len(final_samples) == len(right_channel):
                stereo_output_samples = np.column_stack((final_samples, right_channel))
            else: 
                stereo_output_samples = np.column_stack((final_samples, final_samples))
        else: 
            # For 'stereo' mode, if we want per-oscillator panning, it's complex here
            # as samples are already mixed. Panning is currently an osc property.
            # If different notes use different oscs with different pan settings,
            # this simple column_stack won't reflect that. This needs a more advanced mixer.
            # For now, treat 'stereo' like 'mono' in terms of L/R channel creation from mixed signal.
            stereo_output_samples = np.column_stack((final_samples, final_samples))

        if master_mute:
            stereo_output_samples = np.zeros_like(stereo_output_samples)
        else:
            stereo_output_samples *= master_volume
            
        return stereo_output_samples

class ChordGeneratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Automatic Shallot v0.4")
        self.root.geometry("887x468")
        
        self.root.minsize(600, 400)
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
        # Add initial oscillator(s) here if desired for a new session
        if not self.chord_generator.oscillators: # Add one if the list is empty
             self.chord_generator.add_oscillator()
        
        # Audio setup
        self.pygame = pygame # Make pygame accessible for SequencerUI if it needs it
        self.np = np # Make numpy accessible for SequencerUI if it needs it
        pygame.init()
        self.pygame.mixer.init(frequency=44100, size=-16, channels=2)
        
        # Setup Menu
        self.setup_menu()
        
        # Initialize control variables
        self.init_control_variables()
        
        # Setup GUI
        self.setup_gui()
        
        # Bind resize event
        self.root.bind("<Configure>", self._on_main_window_configure)
        self._last_reported_size = "" # To avoid spamming console for non-size changes

        # Update initial waveform previews
        for i in range(len(self.chord_generator.oscillators)):
            self.update_waveform_preview(i)
    
    def setup_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Rename Oscillators...", command=self.open_rename_oscillators_dialog)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Potentially other menus like Edit, View, Help can be added here

    def open_rename_oscillators_dialog(self):
        dialog = OscillatorRenamerDialog(self)
        self.root.wait_window(dialog) # Wait for the dialog to close

    def init_control_variables(self):
        """Initialize all control variables"""
        # Oscillator._osc_count = 0 # REMOVED: No longer managing count here or in Oscillator class globally for naming

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
        self.eq_gain_vars = [] # List of lists, one list of 8 tk.DoubleVars per oscillator
        self.duration_var = tk.DoubleVar(value=1.0)
        self.octave_var = tk.IntVar(value=4)
        self.note_octave_vars = []
        self.note_pitch_vars = []
        self.note_osc_idx_vars = [] # For oscillator assignment per note
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
        self.eq_gain_vars.append([tk.DoubleVar(value=0.0) for _ in range(8)]) # Initialize EQ gains for new osc
    
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

        mode_frame = ttk.LabelFrame(top_controls_outer_frame, text="Mode", padding="2")
        mode_frame.grid(row=0, column=0, sticky="ew", padx=(0,2), pady=1)
        
        ttk.Label(mode_frame, text="Mode:").grid(row=0, column=0, padx=(2,1))
        mode_menu = ttk.Combobox(mode_frame, textvariable=self.sound_mode_var,
                                values=ChordGenerator.SOUND_MODES, width=8)
        mode_menu.grid(row=0, column=1, padx=(1,2), sticky="ew")
        mode_menu.bind('<<ComboboxSelected>>', self.update_sound_mode)

        # Master Output Frame
        master_output_frame = ttk.LabelFrame(top_controls_outer_frame, text="Master", padding="2")
        master_output_frame.grid(row=0, column=1, sticky="ew", padx=2, pady=1)
        master_output_frame.grid_columnconfigure(1, weight=1) # Scale expands

        ttk.Label(master_output_frame, text="Vol:").grid(row=0, column=0, padx=(2,1), pady=1, sticky="w")
        master_vol_scale = ttk.Scale(master_output_frame, from_=0, to=1, 
                                     variable=self.master_volume_var, 
                                     orient=tk.HORIZONTAL, length=70)
        master_vol_scale.grid(row=0, column=1, padx=1, pady=1, sticky="ew")

        master_mute_cb = ttk.Checkbutton(master_output_frame, text="Mute", 
                                       variable=self.master_mute_var)
        master_mute_cb.grid(row=0, column=2, padx=(2,2), pady=1)
        
        export_btn = ttk.Button(top_controls_outer_frame, text="Export", command=self.export_wav)
        export_btn.grid(row=0, column=2, padx=(2,0), pady=1, sticky="e")
        
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
        osc = self.chord_generator.oscillators[index]
        # Use the oscillator's name for the LabelFrame text
        osc_frame = ttk.LabelFrame(parent, text=osc.name, padding="1")
        osc_frame.grid(row=index, column=0, sticky="ew", padx=1, pady=1)
        osc_frame.grid_columnconfigure(1, weight=1) # Main content area
        osc_frame.grid_columnconfigure(0, weight=1) # Ensure label frame text area also expands

        # The rest of the frame content (preview, controls) starts from row 0 within this osc_frame
        # This means the previous logic of passing start_row to create_waveform_preview might need adjustment
        # if we are not putting an editable name label inside anymore.

        if len(self.chord_generator.oscillators) > 1:
            remove_btn = ttk.Button(osc_frame, text="X",
                                  command=lambda i=index: self.remove_oscillator(i),
                                  width=2)
            # Place remove button neatly, e.g., at the top-right of the content area
            # If LabelFrame text is dynamic, this button should be inside.
            # For simplicity, let's assume it's part of the main content grid of osc_frame.
            # It needs to be added to a specific row/column in osc_frame's internal grid.
            # Let's put it next to where the name *would* have been if it was editable here.
            # Grid it to the LabelFrame itself if it should appear on the border, else manage internally.
            # For now, let's try to make it appear on the top right of the content.
            # This is often done by having a sub-frame or careful grid placement.

            # Simplification: Place it on the first row, last column of internal grid of osc_frame
            # We need to define how osc_frame's columns are structured. Let's say 2 columns.
            osc_frame.grid_columnconfigure(0, weight=1) # For controls
            osc_frame.grid_columnconfigure(1, weight=0) # For remove button
            remove_btn.grid(row=0, column=1, padx=1, pady=0, sticky="ne") 

        self.create_waveform_preview(osc_frame, index, base_row=0) # Pass base_row for internal gridding
        
        basic_frame = ttk.Frame(osc_frame)
        # Basic frame now starts at row 1 if preview is at row 0
        basic_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=0)
        
        enable_cb = ttk.Checkbutton(basic_frame, text="On",
                                  variable=self.enabled_vars[index],
                                  command=lambda i=index: self.update_oscillator(i))
        enable_cb.grid(row=0, column=0, padx=(1,0))
        
        ttk.Label(basic_frame, text="Wave:").grid(row=0, column=1, padx=(2,0))
        wave_menu = ttk.Combobox(basic_frame, textvariable=self.wave_vars[index],
                                values=list(self.chord_generator.oscillators[0].waveform_definitions.keys()),
                                width=8)
        wave_menu.grid(row=0, column=2, padx=(0,1))
        wave_menu.bind('<<ComboboxSelected>>', lambda e, i=index: self.update_waveform_preview(i))
        
        self.create_waveform_tooltip(wave_menu, index)
        
        solo_cb = ttk.Checkbutton(basic_frame, text="Solo", 
                                  variable=self.solo_vars[index],
                                  command=lambda i=index: self.toggle_solo(i))
        solo_cb.grid(row=0, column=3, padx=1)
        
        ttk.Label(basic_frame, text="Amp:").grid(row=0, column=4, padx=(2,0))
        amp_scale = ttk.Scale(basic_frame, from_=0, to=1,
                             variable=self.amp_vars[index],
                             orient=tk.HORIZONTAL, length=50)
        amp_scale.grid(row=0, column=5, padx=(0,1))
        
        ttk.Label(basic_frame, text="Det:").grid(row=0, column=6, padx=(2,0))
        detune_scale = ttk.Scale(basic_frame, from_=-100, to=100,
                                variable=self.detune_vars[index],
                                orient=tk.HORIZONTAL, length=50)
        detune_scale.grid(row=0, column=7, padx=(0,1))
        
        adsr_frame = ttk.LabelFrame(osc_frame, text="ADSR", padding="1")
        adsr_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=0)
        
        adsr_labels = ['A', 'D', 'S', 'R']
        adsr_vars = [self.attack_vars[index], self.decay_vars[index],
                    self.sustain_vars[index], self.release_vars[index]]
        
        for i, (label, var) in enumerate(zip(adsr_labels, adsr_vars)):
            ttk.Label(adsr_frame, text=label).grid(row=0, column=i*2, padx=0)
            ttk.Scale(adsr_frame, from_=0, to=2 if label != 'S' else 1,
                     variable=var, orient=tk.HORIZONTAL, length=35).grid(row=0, column=i*2+1, padx=(0,1))
        
        filter_frame = ttk.Frame(osc_frame)
        filter_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=0)
        
        ttk.Label(filter_frame, text="Cut:").grid(row=0, column=0, padx=(1,0))
        ttk.Scale(filter_frame, from_=0, to=1,
                 variable=self.cutoff_vars[index],
                 orient=tk.HORIZONTAL, length=50).grid(row=0, column=1, padx=(0,1))
        
        ttk.Label(filter_frame, text="Res:").grid(row=0, column=2, padx=(2,0))
        ttk.Scale(filter_frame, from_=0, to=1,
                 variable=self.resonance_vars[index],
                 orient=tk.HORIZONTAL, length=50).grid(row=0, column=3, padx=(0,1))
        
        ttk.Label(filter_frame, text="Pan:").grid(row=0, column=4, padx=(2,0))
        ttk.Scale(filter_frame, from_=0, to=1,
                 variable=self.pan_vars[index],
                 orient=tk.HORIZONTAL, length=50).grid(row=0, column=5, padx=(0,1))
        
        # EQ Controls Frame
        eq_frame = ttk.LabelFrame(osc_frame, text="EQ", padding="1")
        eq_frame.grid(row=4, column=0, columnspan=2, sticky="ew", pady=0)
        for i in range(8):
            eq_band_label = f"{Oscillator.EQ_BAND_FREQUENCIES[i]}" # Hz
            if Oscillator.EQ_BAND_FREQUENCIES[i] >= 1000:
                eq_band_label = f"{Oscillator.EQ_BAND_FREQUENCIES[i]/1000:.0f}k"

            ttk.Label(eq_frame, text=eq_band_label, width=3, anchor="center").grid(row=0, column=i, padx=1)
            eq_scale = ttk.Scale(eq_frame, from_=-12, to=12, 
                                 variable=self.eq_gain_vars[index][i],
                                 orient=tk.VERTICAL, length=50)
            eq_scale.grid(row=1, column=i, padx=1, pady=(0,1))
            # Add a label for 0dB mark, might need better placement or a custom widget later
            # ttk.Label(eq_frame, text="0", font=("TkSmallCaptionFont",)).grid(row=2, column=i, sticky="n")

        self.bind_oscillator_controls(index)
    
    def create_waveform_preview(self, parent, index, base_row=0):
        """Create a small waveform preview canvas, gridded at base_row"""
        fig = Figure(figsize=(1.2, 0.4), dpi=70)
        ax = fig.add_subplot(111)
        
        ax.set_xticks([])
        ax.set_yticks([])
        
        t = np.linspace(0, 1, 1000)
        y = np.sin(2 * np.pi * t)
        self.waveform_plots[index] = ax.plot(t, y)[0]
        
        ax.set_ylim(-1.1, 1.1)
        
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().grid(row=base_row, column=0, columnspan=1, padx=1, pady=0, sticky="ew") # columnspan 1 if remove is in col 1
        
        self.waveform_canvases[index] = canvas
    
    def update_waveform_preview(self, index):
        """Update the waveform preview for an oscillator"""
        waveform = self.wave_vars[index].get()
        osc = self.chord_generator.oscillators[index]
        wave_def = osc.waveform_definitions.get(waveform, osc.waveform_definitions["sine"])
        
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
        chord_frame.grid_columnconfigure(1, weight=0)
        chord_frame.grid_columnconfigure(3, weight=0)
        chord_frame.grid_columnconfigure(5, weight=0)
        chord_frame.grid_columnconfigure(7, weight=0)
        
        ttk.Label(chord_frame, text="Root:").grid(row=0, column=0, padx=(2,0), pady=1)
        root_menu = ttk.Combobox(chord_frame, textvariable=self.root_var,
                                values=list(ChordGenerator.NOTE_FREQUENCIES.keys()),
                                width=2)
        root_menu.grid(row=0, column=1, padx=(0,1), pady=1)
        
        ttk.Label(chord_frame, text="Oct:").grid(row=0, column=2, padx=(2,0), pady=1)
        octave_spin = ttk.Spinbox(chord_frame, from_=0, to=8,
                                 textvariable=self.octave_var,
                                 width=1)
        octave_spin.grid(row=0, column=3, padx=(0,1), pady=1)
        
        ttk.Label(chord_frame, text="Type:").grid(row=0, column=4, padx=(2,0), pady=1)
        self.type_menu = ttk.Combobox(chord_frame, textvariable=self.type_var,
                                     values=list(self.chord_generator.chord_definitions.keys()),
                                     width=8)
        self.type_menu.grid(row=0, column=5, padx=(0,1), pady=1)
        
        ttk.Label(chord_frame, text="Beats:").grid(row=0, column=6, padx=(2,0), pady=1)
        duration_spin = ttk.Spinbox(chord_frame, from_=0.25, to=8,
                                  increment=0.25,
                                  textvariable=self.duration_var,
                                  width=3)
        duration_spin.grid(row=0, column=7, padx=(0,1), pady=1)

        self.custom_notes_frame = ttk.LabelFrame(chord_frame, text="Notes", padding="1")
        self.custom_notes_frame.grid(row=1, column=0, columnspan=8, sticky="ew", padx=1, pady=(2,1))
        
        self.update_custom_chord_note_ui()

        add_note_btn = ttk.Button(chord_frame, text="Add Note", command=self.add_note_to_custom_chord)
        add_note_btn.grid(row=2, column=0, columnspan=8, pady=(1,2))

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
        for i in range(8):
            self.eq_gain_vars[index][i].trace_add('write', lambda *args, i=i, idx=index: self.update_oscillator_eq(idx, i))

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
    
    def update_oscillator_eq(self, osc_index, eq_band_index):
        """Update a specific EQ band for an oscillator."""
        if osc_index < len(self.chord_generator.oscillators) and eq_band_index < 8:
            osc = self.chord_generator.oscillators[osc_index]
            osc.eq_gains[eq_band_index] = self.eq_gain_vars[osc_index][eq_band_index].get()
            # print(f"Osc {osc_index}, EQ Band {eq_band_index} gain: {osc.eq_gains[eq_band_index]}") # For debugging

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
                 self.custom_chord_notes_data = [{'pitch': self.root_var.get(), 'octave_adjust': 0, 'osc_idx': -1 }]
            return

        root_note = self.root_var.get()
        intervals = self.chord_generator.chord_definitions[chord_type_name]["intervals"]
        self.custom_chord_notes_data = []
        for interval in intervals:
            note_name = self._get_note_name(root_note, interval)
            self.custom_chord_notes_data.append({'pitch': note_name, 'octave_adjust': 0, 'osc_idx': -1 })
        # No self.type_var.set("-- Custom --") here; that's for when user *manually* edits.

    def update_custom_chord_note_ui(self):
        """Dynamically create/update UI controls for each note in self.custom_chord_notes_data, with horizontal scrolling."""
        
        # If canvas and content frame don\'t exist, create them within self.custom_notes_frame
        if not self.custom_notes_canvas:
            self.custom_notes_canvas = tk.Canvas(self.custom_notes_frame, height=90)
            h_scrollbar = ttk.Scrollbar(self.custom_notes_frame, orient=tk.HORIZONTAL, command=self.custom_notes_canvas.xview)
            self.custom_notes_canvas.configure(xscrollcommand=h_scrollbar.set)
            
            self.custom_notes_content_frame = ttk.Frame(self.custom_notes_canvas)
            self.custom_notes_canvas.create_window((0, 0), window=self.custom_notes_content_frame, anchor="nw")

            self.custom_notes_canvas.pack(side=tk.TOP, fill=tk.X, expand=True, padx=1, pady=1)
            h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X, padx=1, pady=1)
            
            self.custom_notes_content_frame.bind("<Configure>", 
                lambda e: self.custom_notes_canvas.configure(scrollregion=self.custom_notes_canvas.bbox("all")))

        # Clear existing individual note UI frames from the content_frame
        for frame_widget in self.custom_chord_notes_ui_frames:
            frame_widget.destroy()
        self.custom_chord_notes_ui_frames = []
        self.note_pitch_vars = []
        self.note_octave_vars = []
        self.note_osc_idx_vars = [] # Clear and repopulate

        all_note_names = list(ChordGenerator.NOTE_FREQUENCIES.keys())
        # Use oscillator names for the dropdown
        osc_names = ["master"] + [osc.name for osc in self.chord_generator.oscillators]
        
        for i, note_data in enumerate(self.custom_chord_notes_data):
            # Each note_ui_frame is now created inside self.custom_notes_content_frame
            note_ui_frame = ttk.Frame(self.custom_notes_content_frame, borderwidth=1, relief="sunken", padding=1)
            self.custom_chord_notes_ui_frames.append(note_ui_frame)
            note_ui_frame.pack(side=tk.LEFT, padx=2, pady=1, fill=tk.Y)

            ttk.Label(note_ui_frame, text=f"N{i+1}").pack(side=tk.TOP, pady=(0,1))
            
            pitch_var = tk.StringVar(value=note_data['pitch'])
            self.note_pitch_vars.append(pitch_var)
            pitch_combo = ttk.Combobox(note_ui_frame, textvariable=pitch_var, values=all_note_names, width=3)
            pitch_combo.pack(side=tk.TOP, pady=1)
            pitch_combo.bind('<<ComboboxSelected>>', lambda e, idx=i, pv=pitch_var: self.on_custom_note_param_change(idx, 'pitch', pv.get()))
            pitch_combo.bind('<FocusOut>', lambda e, idx=i, pv=pitch_var: self.on_custom_note_param_change(idx, 'pitch', pv.get()))

            octave_var = tk.IntVar(value=note_data['octave_adjust'])
            self.note_octave_vars.append(octave_var)
            octave_spin = ttk.Spinbox(note_ui_frame, from_=-3, to=3, increment=1, textvariable=octave_var, width=2)
            octave_spin.pack(side=tk.TOP, pady=1)
            octave_spin.configure(command=lambda v=octave_var, idx=i: self.on_custom_note_param_change(idx, 'octave_adjust', v.get()))
            octave_spin.bind('<FocusOut>', lambda e, v=octave_var, idx=i: self.on_custom_note_param_change(idx, 'octave_adjust', v.get()))
            octave_spin.bind('<Return>', lambda e, v=octave_var, idx=i: self.on_custom_note_param_change(idx, 'octave_adjust', v.get()))

            osc_idx_var = tk.IntVar(value=note_data.get('osc_idx', -1))
            self.note_osc_idx_vars.append(osc_idx_var)
            osc_combo = ttk.Combobox(note_ui_frame, textvariable=osc_idx_var, values=osc_names, width=15, state='readonly')
            
            # Explicitly define current_assigned_osc_idx for setting the combobox
            current_assigned_osc_idx = note_data.get('osc_idx', -1)
            
            # Determine the display name for the combobox
            display_name_for_combo = "master" # Default
            if current_assigned_osc_idx == -1:
                display_name_for_combo = "master"
            elif 0 <= current_assigned_osc_idx < len(self.chord_generator.oscillators):
                display_name_for_combo = self.chord_generator.oscillators[current_assigned_osc_idx].name
            else:
                # If idx is out of bounds (but not -1), it implies inconsistent data.
                # Set to Master/All and correct the underlying data.
                display_name_for_combo = "master"
                if i < len(self.custom_chord_notes_data):
                    self.custom_chord_notes_data[i]['osc_idx'] = -1
            
            osc_combo.set(display_name_for_combo)

            osc_combo.pack(side=tk.TOP, pady=1)
            osc_combo.bind('<<ComboboxSelected>>', 
                           lambda e, idx=i, cb=osc_combo: self.on_custom_note_param_change(idx, 'osc_idx', cb.get()))

            remove_btn = ttk.Button(note_ui_frame, text="-", width=1, command=lambda idx=i: self.remove_note_from_custom_chord(idx))
            remove_btn.pack(side=tk.TOP, pady=1)
        
        self.custom_notes_content_frame.update_idletasks()
        self.custom_notes_canvas.configure(scrollregion=self.custom_notes_canvas.bbox("all"))

    def on_custom_note_param_change(self, note_idx: int, param_type: str, new_value):
        """Called when a note\'s pitch or octave adjustment is changed in the UI."""
        if note_idx < len(self.custom_chord_notes_data):
            if param_type == 'octave_adjust':
                try: new_value = int(new_value)
                except ValueError: return

            if param_type == 'pitch':
                self.custom_chord_notes_data[note_idx]['pitch'] = str(new_value)
            elif param_type == 'octave_adjust':
                 self.custom_chord_notes_data[note_idx]['octave_adjust'] = new_value
            elif param_type == 'osc_idx':
                if new_value == "master":
                    self.custom_chord_notes_data[note_idx]['osc_idx'] = -1
                else:
                    found_osc_idx = -1
                    for idx, osc in enumerate(self.chord_generator.oscillators):
                        if osc.name == new_value:
                            found_osc_idx = idx
                            break
                    if found_osc_idx != -1:
                        self.custom_chord_notes_data[note_idx]['osc_idx'] = found_osc_idx
                    else:
                        # This case should ideally not happen if combobox is populated correctly
                        # and names are unique (which they should be if renaming is managed well).
                        print(f"Warning: Oscillator name '{new_value}' not found. Defaulting to master")
                        self.custom_chord_notes_data[note_idx]['osc_idx'] = -1
                        # Consider re-setting the combobox to 'Master/All' visually if possible
            
            if self.type_var.get() != "-- Custom --":
                self.type_var.set("-- Custom --")

    def add_note_to_custom_chord(self):
        """Adds a new default note to the custom chord structure and updates UI."""
        new_note_pitch = self.root_var.get() # Default to current root note
        new_note_octave_adjust = 0
        self.custom_chord_notes_data.append({'pitch': new_note_pitch, 'octave_adjust': new_note_octave_adjust, 'osc_idx': -1 })
        
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
        index = self.chord_generator.add_oscillator() # Name is now set in Oscillator class
        self.init_oscillator_controls(index)
        self.create_oscillator_frame(self.oscillator_frame, index)
        self.update_waveform_preview(index)
        self.update_custom_chord_note_ui() 
    
    def remove_oscillator(self, index):
        """Remove an oscillator"""
        if self.chord_generator.remove_oscillator(index):
            # Adjust osc_idx in custom_chord_notes_data due to removal
            for note_data in self.custom_chord_notes_data:
                current_assigned_idx = note_data.get('osc_idx', -1)
                if current_assigned_idx == -1: continue
                if current_assigned_idx == index: note_data['osc_idx'] = -1
                elif current_assigned_idx > index: note_data['osc_idx'] = current_assigned_idx - 1

            # Remove control variables
            for var_list in [self.wave_vars, self.amp_vars, self.detune_vars,
                           self.attack_vars, self.decay_vars, self.sustain_vars,
                           self.release_vars, self.cutoff_vars, self.resonance_vars,
                           self.pan_vars, self.enabled_vars, self.solo_vars, self.eq_gain_vars]:
                if index < len(var_list):
                    if var_list is self.eq_gain_vars: 
                        if index < len(var_list): del var_list[index]
                    elif isinstance(var_list, list) and len(var_list) > 0 and isinstance(var_list[0], tk.Variable):
                        if index < len(var_list): del var_list[index]

            if index in self.waveform_plots: del self.waveform_plots[index]
            if index in self.waveform_canvases: del self.waveform_canvases[index]
            
            # Refresh UI
            self.update_oscillator_frames_after_rename() # Rebuild all frames
            self.update_custom_chord_note_ui() 
        else:
            messagebox.showwarning("Remove Oscillator", "Cannot remove the last oscillator.")

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

    def _on_main_window_configure(self, event):
        """Called when the main window is resized or its configuration changes."""
        # Check if the event is for the root window itself, not a child widget if event propagates
        if event.widget == self.root:
            current_width = self.root.winfo_width()
            current_height = self.root.winfo_height()
            size_str = f"Window Resized to: {current_width}x{current_height}"
            # Only print if the size actually changed to reduce console spam from other configure events
            if size_str != self._last_reported_size:
                print(size_str)
                self._last_reported_size = size_str

    def update_oscillator_frames_after_rename(self):
        """Rebuilds all oscillator frames to reflect new names and order."""
        # Clear existing oscillator frames
        for widget in self.oscillator_frame.winfo_children():
            widget.destroy()
        
        # Re-create all oscillator frames. This will use the new names.
        for i in range(len(self.chord_generator.oscillators)):
            self.create_oscillator_frame(self.oscillator_frame, i)
            self.update_waveform_preview(i) # Ensure preview is also updated

if __name__ == "__main__":
    # Reset Oscillator count when app starts, if desired for consistent default naming like "Oscillator 1", "Oscillator 2" etc.
    # This is a design choice. If names are loaded from a file, this might not be wanted here.
    # Oscillator._osc_count = 0 # Moved to init_control_variables for better control if app can be re-initialized
    root = tk.Tk()
    app = ChordGeneratorApp(root)
    root.mainloop() 
