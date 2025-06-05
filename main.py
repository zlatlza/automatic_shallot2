import tkinter as tk
from tkinter import ttk, messagebox, filedialog # Added filedialog
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
from waveform_sculptor import WaveformSculptor # Import the WaveformSculptor

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
    
    def generate_chord(self, master_octave: int, notes_data: List[Dict], duration_ms: int = 1000, fade_out_ms: int = 100, master_volume: float = 1.0, master_mute: bool = False, soloed_osc_idx: int = -1, bpm: int = 120) -> np.ndarray:
        """ Generates a chord based on a list of notes, each with a pitch, octave adjustment, oscillator assignment, and individual beat length. """
        
        if not notes_data:
            return np.array([]) 

        # Overall duration for the output buffer and main fade out
        step_total_render_duration_ms = duration_ms + fade_out_ms
        final_samples_shape = int(44100 * step_total_render_duration_ms / 1000)
        combined_note_samples = np.zeros(final_samples_shape)

        # Determine effective oscillators based on global solo and enabled states
        if soloed_osc_idx != -1 and soloed_osc_idx < len(self.oscillators) and self.oscillators[soloed_osc_idx].enabled:
            effective_oscs_for_master_all = [self.oscillators[soloed_osc_idx]]
        else: 
            effective_oscs_for_master_all = [osc for osc in self.oscillators if osc.enabled]

        if not self.oscillators: # If no oscillators at all (e.g. all removed)
             return np.zeros((final_samples_shape, 2)) if self.sound_mode != 'mono' else np.zeros(final_samples_shape)

        ms_per_beat = 60000.0 / bpm

        for note_info in notes_data:
            note_pitch_str = note_info['pitch']
            note_octave_adjust = note_info['octave_adjust']
            assigned_osc_idx_for_note = note_info.get('osc_idx', -1)
            note_beat_length = note_info.get('beat_length', 1.0)

            # Calculate note-specific duration in ms, capped by the step's main duration_ms
            # This duration is for the active sound of the note before any step-level fade out.
            current_note_actual_duration_ms = min(note_beat_length * ms_per_beat, duration_ms)
            # Ensure a small positive duration if calculations result in zero or negative.
            current_note_actual_duration_ms = max(1, int(current_note_actual_duration_ms))

            if note_pitch_str not in self.NOTE_FREQUENCIES:
                print(f"Warning: Pitch {note_pitch_str} not found. Skipping note.")
                continue

            base_freq_for_note = self.NOTE_FREQUENCIES[note_pitch_str]
            current_note_freq = base_freq_for_note * (2 ** (master_octave - 4 + note_octave_adjust))
            
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

            # This buffer is for accumulating one note's sound from potentially multiple oscillators.
            # It's sized for the entire step because a note might be shorter than the step.
            single_note_contribution_buffer = np.zeros(final_samples_shape)

            for osc in oscillators_to_use_for_this_note:
                original_osc_freq = osc.frequency 
                osc.frequency = current_note_freq
                # Generate samples for the note's specific duration
                # The ADSR envelope within generate_samples will apply to this duration.
                generated_samples_for_note = osc.generate_samples(current_note_actual_duration_ms)
                osc.frequency = original_osc_freq 
                
                # Add the generated samples to the start of this note's buffer
                # If generated_samples_for_note is shorter than final_samples_shape, it's effectively padded with silence.
                len_generated = len(generated_samples_for_note)
                if len_generated > 0:
                    single_note_contribution_buffer[:len_generated] += generated_samples_for_note
            
            combined_note_samples += single_note_contribution_buffer

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
    DARK_GREY = "#3C3C3C"
    MEDIUM_GREY = "#4A4A4A"
    LIGHT_GREY_FG = "#E0E0E0"
    BUTTON_GREY = "#5A5A5A"
    BUTTON_ACTIVE_GREY = "#6A6A6A"
    CRIMSON = "#DC143C"
    SLIDER_TROUGH_GREY = "#2A2A2A"
    FIELD_BG_GREY = "#424242" # For combobox/spinbox fields
    WHITE_ISH_TEXT = "#F0F0F0"

    def __init__(self, root):
        self.root = root
        self.root.title("Automatic Shallot v0.4")
        self.root.geometry("983x534")
        self.root.configure(bg=self.DARK_GREY) # Root background

        # --- Style Setup ---
        style = ttk.Style(self.root)
        # Using 'clam' as it's generally more customizable
        # Available themes: style.theme_names() -> ('winnative', 'clam', 'alt', 'default', 'classic', 'vista', 'xpnative')
        try:
            style.theme_use('clam')
        except tk.TclError:
            print("Clam theme not available, using default.")
            # Fallback or use another theme if 'clam' isn't available

        # General widget styling
        style.configure('.', 
                        background=self.DARK_GREY, 
                        foreground=self.LIGHT_GREY_FG,
                        font=('TkDefaultFont', 9)) # Adjusted font size for potentially better fit

        style.configure('TFrame', background=self.DARK_GREY)
        style.configure('TLabel', background=self.DARK_GREY, foreground=self.LIGHT_GREY_FG)
        
        style.configure('TLabelframe', 
                        background=self.DARK_GREY, 
                        foreground=self.LIGHT_GREY_FG, 
                        bordercolor=self.MEDIUM_GREY,
                        borderwidth=1) 
        style.configure('TLabelframe.Label', 
                        background=self.DARK_GREY, 
                        foreground=self.LIGHT_GREY_FG)

        # Button styling
        style.configure('TButton', 
                        background=self.BUTTON_GREY, 
                        foreground=self.WHITE_ISH_TEXT, 
                        bordercolor=self.MEDIUM_GREY,
                        lightcolor=self.BUTTON_GREY, # For 3D effect with clam
                        darkcolor=self.BUTTON_GREY,  # For 3D effect with clam
                        padding=(6, 3)) # Added padding
        style.map('TButton', 
                  background=[('active', self.BUTTON_ACTIVE_GREY), ('pressed', self.BUTTON_ACTIVE_GREY)],
                  foreground=[('active', self.WHITE_ISH_TEXT)])

        # Checkbutton styling
        style.configure('TCheckbutton', 
                        background=self.DARK_GREY, 
                        foreground=self.LIGHT_GREY_FG)
        style.map('TCheckbutton',
                  background=[('active', self.MEDIUM_GREY)],
                  indicatorbackground=[('selected', self.CRIMSON)], # Try to make check mark crimson
                  indicatorforeground=[('selected', self.WHITE_ISH_TEXT)])


        # Combobox styling
        style.configure('TCombobox', 
                        fieldbackground=self.FIELD_BG_GREY, 
                        background=self.BUTTON_GREY, # Arrow button background
                        foreground=self.LIGHT_GREY_FG, 
                        selectbackground=self.MEDIUM_GREY, # Background of selected item in dropdown
                        selectforeground=self.WHITE_ISH_TEXT, # Text of selected item in dropdown
                        arrowcolor=self.LIGHT_GREY_FG,
                        bordercolor=self.MEDIUM_GREY)
        style.map('TCombobox', 
                  fieldbackground=[('readonly', self.FIELD_BG_GREY), ('focus', self.FIELD_BG_GREY)],
                  background=[('readonly', self.BUTTON_GREY)], # Arrow button
                  foreground=[('readonly', self.LIGHT_GREY_FG)])


        # Spinbox styling
        style.configure('TSpinbox', 
                        fieldbackground=self.FIELD_BG_GREY, 
                        background=self.BUTTON_GREY, # Button background
                        foreground=self.LIGHT_GREY_FG, 
                        arrowcolor=self.LIGHT_GREY_FG,
                        bordercolor=self.MEDIUM_GREY)
        style.map('TSpinbox', foreground=[('focus', self.WHITE_ISH_TEXT)])


        # Scale (Slider) styling - Attempting Crimson thumb
        # This is highly theme-dependent. For 'clam':
        style.configure('Horizontal.TScale', 
                        background=self.DARK_GREY, # Overall background
                        troughcolor=self.SLIDER_TROUGH_GREY)
        style.configure('Vertical.TScale', 
                        background=self.DARK_GREY, 
                        troughcolor=self.SLIDER_TROUGH_GREY)
        
        # Custom styles for TScale widgets, attempting to make thumbs smaller via widget style options
        style.configure('SmallThumb.Horizontal.TScale', sliderlength=12, sliderthickness=12, troughcolor=self.SLIDER_TROUGH_GREY, background=self.DARK_GREY)
        style.configure('SmallThumb.Vertical.TScale', sliderlength=12, sliderthickness=10, troughcolor=self.SLIDER_TROUGH_GREY, background=self.DARK_GREY)
        # Note: The 'background' for SmallThumb.*.TScale is the widget background, not thumb.
        # Troughcolor is also for the widget. Color of the thumb itself is handled below.

        # Style the slider thumb *element* - primarily for color.
        # Size changes (sliderlength/thickness) proved ineffective when applied directly to the element, so removed from here.
        style.configure('Horizontal.TScale.slider', background=self.CRIMSON, bordercolor=self.CRIMSON, lightcolor=self.CRIMSON, darkcolor=self.CRIMSON)
        style.configure('Vertical.TScale.slider', background=self.CRIMSON, bordercolor=self.CRIMSON, lightcolor=self.CRIMSON, darkcolor=self.CRIMSON)
        
        # Scrollbar styling
        style.configure("TScrollbar", 
                        background=self.BUTTON_GREY, # Background of the scrollbar itself (not trough)
                        troughcolor=self.SLIDER_TROUGH_GREY, 
                        bordercolor=self.MEDIUM_GREY, 
                        arrowcolor=self.LIGHT_GREY_FG)
        style.map("TScrollbar", 
                  background=[('active', self.BUTTON_ACTIVE_GREY)], 
                  arrowcolor=[('pressed', self.CRIMSON)])

        # --- End Style Setup ---
        
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
        self.oscillator_ui_elements = {} # To store UI elements like wave_menu for each oscillator
        
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
        file_menu.add_command(label="Import Sequence...", command=self._import_sequence_dialog)
        file_menu.add_command(label="Export Sequence...", command=self._export_sequence_dialog)
        file_menu.add_separator()
        file_menu.add_command(label="Import Oscillator Rack...", command=self.import_oscillator_rack)
        file_menu.add_command(label="Export Oscillator Rack...", command=self.export_oscillator_rack)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Potentially other menus like Edit, View, Help can be added here

    def _import_sequence_dialog(self):
        """Opens a dialog to import a sequence from a .shallot file."""
        if not self.sequencer_ui_instance:
            messagebox.showerror("Error", "Sequencer UI is not available.")
            return

        filepath = filedialog.askopenfilename(
            defaultextension=".shallot",
            filetypes=[("Shallot Files", "*.shallot"), ("All Files", "*.*")],
            title="Import Shallot Sequence File"
        )
        if not filepath:
            return # User cancelled

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                sequence_text_content = f.read()
            
            self.sequencer_ui_instance.load_sequence_from_text(sequence_text_content)
            messagebox.showinfo("Success", f"Sequence successfully imported from\\n{filepath}")
        except Exception as e:
            messagebox.showerror("Import Error", f"Failed to import sequence from {filepath}\\nError: {e}")

    def _export_sequence_dialog(self):
        """Opens a dialog to export the current sequence to a .shallot file."""
        if not self.sequencer_ui_instance:
            messagebox.showerror("Error", "Sequencer UI is not available.")
            return

        sequence_text_to_export = self.sequencer_ui_instance.get_sequence_as_text()
        if not sequence_text_to_export.strip():
            messagebox.showwarning("Export Warning", "Sequence is empty. Nothing to export.")
            return
            
        filepath = filedialog.asksaveasfilename(
            defaultextension=".shallot",
            filetypes=[("Shallot Files", "*.shallot"), ("All Files", "*.*")],
            title="Export Shallot Sequence File"
        )
        if not filepath:
            return # User cancelled

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(sequence_text_to_export)
            messagebox.showinfo("Success", f"Sequence successfully exported to\\n{filepath}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export sequence to {filepath}\\nError: {e}")

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
        self.note_beat_length_vars = [] # For individual note beat lengths
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
        left_frame = ttk.Frame(self.main_frame) # main_frame gets style from root via ChordGeneratorApp init
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
                                     orient=tk.HORIZONTAL, length=100, style='SmallThumb.Horizontal.TScale')
        master_vol_scale.grid(row=0, column=1, padx=1, pady=1, sticky="ew")

        master_mute_cb = ttk.Checkbutton(master_output_frame, text="Mute", 
                                       variable=self.master_mute_var)
        master_mute_cb.grid(row=0, column=2, padx=(2,2), pady=1)
        
        export_btn = ttk.Button(top_controls_outer_frame, text="Export", command=self.export_wav)
        export_btn.grid(row=0, column=2, padx=(2,0), pady=1, sticky="e")
        
        # Create scrollable canvas for oscillators
        # This canvas is tk.Canvas, not ttk.Frame, so styled directly
        canvas = tk.Canvas(left_frame, bg=self.DARK_GREY, highlightthickness=0) # Added bg and highlightthickness
        scrollbar = ttk.Scrollbar(left_frame, orient="vertical", command=canvas.yview)
        self.oscillator_frame = ttk.Frame(canvas) # This ttk.Frame will be styled
        
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
        right_frame.grid_columnconfigure(0, weight=1) # Allow chord_frame's column to expand for centering
        right_frame.grid_rowconfigure(0, weight=0) # Chord frame, fixed size initially
        right_frame.grid_rowconfigure(1, weight=1) # Sequencer UI frame, takes remaining space

        # Chord Controls
        chord_frame = self.create_chord_frame(right_frame)
        chord_frame.grid(row=0, column=0, sticky="n", padx=5, pady=5) # Changed sticky to "n" for centering
        
        # Sequencer UI - Instantiate SequencerUI here
        # The SequencerUI will build its components into a frame it creates inside right_frame
        # Or, we can pass a specific sub-frame of right_frame for it to populate.
        # Let's create a container for SequencerUI within right_frame.
        sequencer_ui_container = ttk.Frame(right_frame)
        sequencer_ui_container.grid(row=1, column=0, sticky="nsew", padx=5, pady=(15, 5)) # Increased top padding
        self.sequencer_ui_instance = SequencerUI(sequencer_ui_container, self)
    
    def create_oscillator_frame(self, parent, index):
        """Create a frame for a single oscillator"""
        osc = self.chord_generator.oscillators[index]
        osc_frame = ttk.LabelFrame(parent, text=osc.name, padding="1")
        osc_frame.grid(row=index, column=0, sticky="ew", padx=1, pady=1)
        osc_frame.grid_columnconfigure(0, weight=1) # Main content column will expand
        osc_frame.grid_columnconfigure(1, weight=0) # Column for remove button, no expansion

        # Initialize UI elements dictionary for this oscillator if it doesn't exist
        if index not in self.oscillator_ui_elements:
            self.oscillator_ui_elements[index] = {}

        if len(self.chord_generator.oscillators) > 1:
            remove_btn = ttk.Button(osc_frame, text="X",
                                  command=lambda i=index: self.remove_oscillator(i),
                                  width=2)
            remove_btn.grid(row=0, column=1, padx=1, pady=0, sticky="ne") 

        self.create_waveform_preview(osc_frame, index, base_row=0) 
        
        basic_frame = ttk.Frame(osc_frame)
        basic_frame.grid(row=1, column=0, columnspan=1, sticky="n", pady=0) # Centered in column 0
        
        enable_cb = ttk.Checkbutton(basic_frame, text="On",
                                  variable=self.enabled_vars[index],
                                  command=lambda i=index: self.update_oscillator(i))
        enable_cb.grid(row=0, column=0, padx=(1,0))
        
        wave_label = ttk.Label(basic_frame, text="Wave:") # Create the label
        wave_label.grid(row=0, column=1, padx=(2,0))
        self.oscillator_ui_elements[index]['wave_label'] = wave_label # Store it
        
        wave_menu = ttk.Combobox(basic_frame, textvariable=self.wave_vars[index],
                                values=list(self.chord_generator.oscillators[index].waveform_definitions.keys()),
                                width=8)
        wave_menu.grid(row=0, column=2, padx=(0,1))
        wave_menu.bind('<<ComboboxSelected>>', lambda e, i=index: self.update_waveform_preview(i))
        self.oscillator_ui_elements[index]['wave_menu'] = wave_menu # Store combobox
        
        self.create_waveform_tooltip(wave_menu, index)
        
        solo_cb = ttk.Checkbutton(basic_frame, text="Solo", 
                                  variable=self.solo_vars[index],
                                  command=lambda i=index: self.toggle_solo(i))
        solo_cb.grid(row=0, column=3, padx=1)
        self.oscillator_ui_elements[index]['solo_button'] = solo_cb # Store solo button
        
        ttk.Label(basic_frame, text="Amp:").grid(row=0, column=4, padx=(2,0))
        amp_scale = ttk.Scale(basic_frame, from_=0, to=1,
                             variable=self.amp_vars[index],
                             orient=tk.HORIZONTAL, length=40, style='SmallThumb.Horizontal.TScale')
        amp_scale.grid(row=0, column=5, padx=(0,1))
        
        ttk.Label(basic_frame, text="Det:").grid(row=0, column=6, padx=(2,0))
        detune_scale = ttk.Scale(basic_frame, from_=-100, to=100,
                                variable=self.detune_vars[index],
                                orient=tk.HORIZONTAL, length=40, style='SmallThumb.Horizontal.TScale')
        detune_scale.grid(row=0, column=7, padx=(0,1))
        
        adsr_frame = ttk.LabelFrame(osc_frame, text="ADSR", padding="1")
        adsr_frame.grid(row=2, column=0, columnspan=1, sticky="n", pady=0) # Centered in column 0
        
        adsr_labels = ['A', 'D', 'S', 'R']
        adsr_vars = [self.attack_vars[index], self.decay_vars[index],
                    self.sustain_vars[index], self.release_vars[index]]
        
        for i, (label, var) in enumerate(zip(adsr_labels, adsr_vars)):
            ttk.Label(adsr_frame, text=label).grid(row=0, column=i*2, padx=0)
            ttk.Scale(adsr_frame, from_=0, to=2 if label != 'S' else 1,
                     variable=var, orient=tk.HORIZONTAL, length=25, style='SmallThumb.Horizontal.TScale').grid(row=0, column=i*2+1, padx=(0,1))
        
        filter_frame = ttk.Frame(osc_frame)
        filter_frame.grid(row=3, column=0, columnspan=1, sticky="n", pady=0) # Centered in column 0
        
        ttk.Label(filter_frame, text="Cut:").grid(row=0, column=0, padx=(1,0))
        ttk.Scale(filter_frame, from_=0, to=1,
                 variable=self.cutoff_vars[index],
                 orient=tk.HORIZONTAL, length=40, style='SmallThumb.Horizontal.TScale').grid(row=0, column=1, padx=(0,1))
        
        ttk.Label(filter_frame, text="Res:").grid(row=0, column=2, padx=(2,0))
        ttk.Scale(filter_frame, from_=0, to=1,
                 variable=self.resonance_vars[index],
                 orient=tk.HORIZONTAL, length=40, style='SmallThumb.Horizontal.TScale').grid(row=0, column=3, padx=(0,1))
        
        ttk.Label(filter_frame, text="Pan:").grid(row=0, column=4, padx=(2,0))
        ttk.Scale(filter_frame, from_=0, to=1,
                 variable=self.pan_vars[index],
                 orient=tk.HORIZONTAL, length=40, style='SmallThumb.Horizontal.TScale').grid(row=0, column=5, padx=(0,1))
        
        # EQ Controls Frame
        eq_frame = ttk.LabelFrame(osc_frame, text="EQ", padding="1")
        eq_frame.grid(row=4, column=0, columnspan=1, sticky="n", pady=0) # Centered in column 0
        for i in range(8):
            eq_band_label = f"{Oscillator.EQ_BAND_FREQUENCIES[i]}" # Hz
            if Oscillator.EQ_BAND_FREQUENCIES[i] >= 1000:
                eq_band_label = f"{Oscillator.EQ_BAND_FREQUENCIES[i]/1000:.0f}k"

            ttk.Label(eq_frame, text=eq_band_label, width=3, anchor="center").grid(row=0, column=i, padx=1)
            eq_scale = ttk.Scale(eq_frame, from_=-12, to=12, 
                                 variable=self.eq_gain_vars[index][i],
                                 orient=tk.VERTICAL, length=60, style='SmallThumb.Vertical.TScale')
            eq_scale.grid(row=1, column=i, padx=1, pady=(0,1))
            # Add a label for 0dB mark, might need better placement or a custom widget later
            # ttk.Label(eq_frame, text="0", font=("TkSmallCaptionFont",)).grid(row=2, column=i, sticky="n")

        self.bind_oscillator_controls(index)
    
    def create_waveform_preview(self, parent, index, base_row=0):
        """Create a small waveform preview canvas, gridded at base_row"""
        fig = Figure(figsize=(1.2, 0.4), dpi=70, facecolor=self.DARK_GREY) # Figure background
        ax = fig.add_subplot(111, facecolor=self.MEDIUM_GREY) # Axes background
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_color(self.LIGHT_GREY_FG) # Adjusting spine colors
        ax.spines['bottom'].set_color(self.LIGHT_GREY_FG)
        ax.spines['left'].set_color(self.LIGHT_GREY_FG)
        ax.spines['right'].set_color(self.LIGHT_GREY_FG)
        ax.tick_params(axis='x', colors=self.LIGHT_GREY_FG) # Though ticks are off, for consistency
        ax.tick_params(axis='y', colors=self.LIGHT_GREY_FG)

        t = np.linspace(0, 1, 1000)
        y = np.sin(2 * np.pi * t)
        self.waveform_plots[index] = ax.plot(t, y, color=self.CRIMSON)[0] # Waveform line color
        
        ax.set_ylim(-1.1, 1.1)
        
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas_widget = canvas.get_tk_widget()
        # Set background of the Tkinter widget holding the Matplotlib canvas
        canvas_widget.config(bg=self.DARK_GREY, highlightthickness=0) 
        canvas_widget.grid(row=base_row, column=0, columnspan=1, padx=1, pady=0, sticky="n") # Centered in column 0 of parent (osc_frame)
        # Bind click to open sculptor
        canvas_widget.bind("<Button-1>", lambda event, i=index: self.open_waveform_sculptor(oscillator_index=i))
        
        self.waveform_canvases[index] = canvas
    
    def update_waveform_preview(self, index):
        """Update the waveform preview for an oscillator"""
        if index not in self.waveform_plots or index not in self.waveform_canvases or index not in self.oscillator_ui_elements:
            return
            
        osc = self.chord_generator.oscillators[index]
        preview_points = []
        num_preview_points = 100 

        # Update Wave label if live editing
        wave_label_widget = self.oscillator_ui_elements[index].get('wave_label')
        if wave_label_widget:
            if osc.is_live_editing:
                wave_label_widget.config(text="Wave (Live):")
            else:
                wave_label_widget.config(text="Wave:")

        if osc.is_live_editing and osc.live_edit_points and len(osc.live_edit_points) > 0:
            # Use live edit points for preview
            # We need to interpolate these to num_preview_points if their length differs
            # or just plot them directly if their length is manageable (e.g. 16 from sculptor)
            # For consistency, let's resample/interpolate to num_preview_points.
            live_pts = np.array(osc.live_edit_points)
            if len(live_pts) == num_preview_points:
                preview_points = live_pts
            elif len(live_pts) > 0:
                x_defined = np.linspace(0, 1, len(live_pts), endpoint=False)
                x_target = np.linspace(0, 1, num_preview_points, endpoint=False)
                preview_points = np.interp(x_target, x_defined, live_pts)
            else: # Should not happen if osc.live_edit_points check passed len > 0
                preview_points = np.zeros(num_preview_points)
        else:
            # Use the oscillator's currently selected defined waveform
            current_waveform_name = self.wave_vars[index].get()
            preview_points = osc.get_waveform_cycle_points(current_waveform_name, num_preview_points)

        if not isinstance(preview_points, np.ndarray):
            preview_points = np.array(preview_points)
        
        if preview_points.size == 0:
            # print(f"Warning: No preview points generated for osc {index}. Defaulting to zeros.")
            preview_points = np.zeros(num_preview_points)
        
        # Ensure y_data for plot has the same number of points as x_data used by plot
        # The plot in create_waveform_preview uses t = np.linspace(0, 1, 1000)
        # So we must ensure preview_points has 1000 samples or update the plot's x_data too.
        # Let's adjust the existing x_data or create new x_data based on num_preview_points.
        
        # The plot object (self.waveform_plots[index]) expects x and y data.
        # If its original x data (t) had a different number of points than num_preview_points,
        # we should update both x and y data for the plot, or ensure they match.
        # For simplicity, the preview plot just needs to show the shape, so 0..N-1 for x is fine.
        plot_x_data = np.arange(len(preview_points))

        self.waveform_plots[index].set_data(plot_x_data, preview_points) # Update both x and y
        
        # Adjust axes if necessary, e.g., if number of points changes x-scale
        ax = self.waveform_canvases[index].figure.axes[0]
        ax.set_xlim(plot_x_data.min(), plot_x_data.max() if len(plot_x_data) > 1 else 1)
        min_y = preview_points.min()
        max_y = preview_points.max()
        ax.set_ylim(min(min_y, -1.1) - 0.1, max(max_y, 1.1) + 0.1) # Dynamic Y-axis with padding
        
        self.waveform_canvases[index].draw()
    
    def create_waveform_tooltip(self, widget, index):
        """Create tooltip for waveform description"""
        def show_tooltip(event):
            if not (0 <= index < len(self.chord_generator.oscillators)):
                return # Invalid index

            osc = self.chord_generator.oscillators[index]
            waveform_name = self.wave_vars[index].get()
            
            # Safely get waveform definition and then its description
            wave_def = osc.waveform_definitions.get(waveform_name, {})
            description = wave_def.get("description", "N/A") # Default to N/A if no description
            
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
        if not (0 <= index < len(self.chord_generator.oscillators)):
            print(f"Error: update_oscillator called with invalid index {index}")
            return
            
        osc = self.chord_generator.oscillators[index]
        osc.enabled = self.enabled_vars[index].get()
        new_waveform_name = self.wave_vars[index].get()
        if osc.waveform != new_waveform_name:
            osc.waveform = new_waveform_name
            # If the waveform changed, live editing should be implicitly turned off
            # as the user has selected a defined waveform.
            if osc.is_live_editing:
                osc.set_live_edit_data(is_active=False)

        osc.amplitude = self.amp_vars[index].get()
        osc.detune = self.detune_vars[index].get()
        osc.attack = self.attack_vars[index].get()
        osc.decay = self.decay_vars[index].get()
        osc.sustain = self.sustain_vars[index].get()
        osc.release = self.release_vars[index].get()
        osc.filter_cutoff = self.cutoff_vars[index].get()
        osc.filter_resonance = self.resonance_vars[index].get()
        osc.pan = self.pan_vars[index].get()
        
        # After updating oscillator state, refresh its preview
        self.update_waveform_preview(index)
    
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
            
            current_bpm = self.sequencer_ui_instance.sequencer.bpm if self.sequencer_ui_instance else 120

            samples = self.chord_generator.generate_chord(
                master_octave=self.octave_var.get(),
                notes_data=current_notes_data,
                duration_ms=1000, # Overall duration for the preview event
                master_volume=master_vol,
                master_mute=master_mute_status,
                soloed_osc_idx=current_solo_idx,
                bpm=current_bpm
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
            current_bpm = self.sequencer_ui_instance.sequencer.bpm if self.sequencer_ui_instance else 120

            samples = self.chord_generator.generate_chord(
                master_octave=self.octave_var.get(),
                notes_data=current_notes_data,
                duration_ms=2000,  # Overall duration for the exported WAV event
                master_volume=master_vol,
                master_mute=False, 
                soloed_osc_idx=current_solo_idx_export,
                bpm=current_bpm
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
        # Get the desired default beat length from the main chord duration setting
        default_beat_length = self.duration_var.get()
        if default_beat_length <= 0: default_beat_length = 1.0

        if not chord_type_name or chord_type_name not in self.chord_generator.chord_definitions:
            # If invalid type, or if it was already custom, perhaps clear or use a default like single root note
            if not self.custom_chord_notes_data: # Only reset if truly empty, to avoid wiping user edits if called unexpectedly
                 self.custom_chord_notes_data = [{'pitch': self.root_var.get(), 'octave_adjust': 0, 'osc_idx': -1, 'beat_length': default_beat_length }]
            return

        root_note = self.root_var.get()
        intervals = self.chord_generator.chord_definitions[chord_type_name]["intervals"]
        self.custom_chord_notes_data = []
        for interval in intervals:
            note_name = self._get_note_name(root_note, interval)
            self.custom_chord_notes_data.append({'pitch': note_name, 'octave_adjust': 0, 'osc_idx': -1, 'beat_length': default_beat_length })
        # No self.type_var.set("-- Custom --") here; that's for when user *manually* edits.

    def update_custom_chord_note_ui(self):
        """Dynamically create/update UI controls for each note in self.custom_chord_notes_data, with horizontal scrolling."""
        
        # If canvas and content frame don't exist, create them within self.custom_notes_frame
        if not self.custom_notes_canvas:
            # This is a tk.Canvas, styled directly
            self.custom_notes_canvas = tk.Canvas(self.custom_notes_frame, height=150, bg=self.DARK_GREY, highlightthickness=0) # Increased height to 150
            h_scrollbar = ttk.Scrollbar(self.custom_notes_frame, orient=tk.HORIZONTAL, command=self.custom_notes_canvas.xview)
            self.custom_notes_canvas.configure(xscrollcommand=h_scrollbar.set)
            
            # This ttk.Frame will inherit styles
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
        self.note_beat_length_vars = [] # Clear and repopulate

        all_note_names = list(ChordGenerator.NOTE_FREQUENCIES.keys())
        # Use oscillator names for the dropdown
        osc_names = ["master"] + [osc.name for osc in self.chord_generator.oscillators]
        
        for i, note_data in enumerate(self.custom_chord_notes_data):
            # Each note_ui_frame is now created inside self.custom_notes_content_frame
            note_ui_frame = ttk.Frame(self.custom_notes_content_frame, borderwidth=1, relief="sunken", padding=1)
            self.custom_chord_notes_ui_frames.append(note_ui_frame)
            note_ui_frame.pack(side=tk.LEFT, padx=2, pady=1, fill=tk.Y)

            # Configure grid for the note_ui_frame
            note_ui_frame.grid_columnconfigure(0, weight=0) # Main controls column
            note_ui_frame.grid_columnconfigure(1, weight=0) # Column for X button
            # note_ui_frame.grid_rowconfigure(0, weight=0) # Row for label and X button
            # ... potentially configure other rows if needed for spacing ...

            # Note Number Label (Top-Left)
            ttk.Label(note_ui_frame, text=f"N{i+1}").grid(row=0, column=0, sticky="nw", pady=(0,1))
            
            # Remove Button (Top-Right)
            remove_btn = ttk.Button(note_ui_frame, text="X", width=2, command=lambda idx=i: self.remove_note_from_custom_chord(idx))
            remove_btn.grid(row=0, column=1, sticky="ne", padx=(2,0), pady=(0,1)) 

            pitch_var = tk.StringVar(value=note_data['pitch'])
            self.note_pitch_vars.append(pitch_var)
            pitch_combo = ttk.Combobox(note_ui_frame, textvariable=pitch_var, values=all_note_names, width=3)
            pitch_combo.grid(row=1, column=0, columnspan=2, pady=1, sticky="ew") # Span across both columns for width
            pitch_combo.bind('<<ComboboxSelected>>', lambda e, idx=i, pv=pitch_var: self.on_custom_note_param_change(idx, 'pitch', pv.get()))
            pitch_combo.bind('<FocusOut>', lambda e, idx=i, pv=pitch_var: self.on_custom_note_param_change(idx, 'pitch', pv.get()))

            octave_var = tk.IntVar(value=note_data['octave_adjust'])
            self.note_octave_vars.append(octave_var)
            octave_spin = ttk.Spinbox(note_ui_frame, from_=-3, to=3, increment=1, textvariable=octave_var, width=2)
            octave_spin.grid(row=2, column=0, columnspan=2, pady=1, sticky="ew") # Span
            octave_spin.configure(command=lambda v=octave_var, idx=i: self.on_custom_note_param_change(idx, 'octave_adjust', v.get()))
            octave_spin.bind('<FocusOut>', lambda e, v=octave_var, idx=i: self.on_custom_note_param_change(idx, 'octave_adjust', v.get()))
            octave_spin.bind('<Return>', lambda e, v=octave_var, idx=i: self.on_custom_note_param_change(idx, 'octave_adjust', v.get()))

            osc_idx_var = tk.IntVar(value=note_data.get('osc_idx', -1))
            self.note_osc_idx_vars.append(osc_idx_var)
            osc_combo = ttk.Combobox(note_ui_frame, textvariable=osc_idx_var, values=osc_names, width=15, state='readonly')
            
            current_assigned_osc_idx = note_data.get('osc_idx', -1)
            display_name_for_combo = "master"
            if current_assigned_osc_idx == -1:
                display_name_for_combo = "master"
            elif 0 <= current_assigned_osc_idx < len(self.chord_generator.oscillators):
                display_name_for_combo = self.chord_generator.oscillators[current_assigned_osc_idx].name
            else:
                display_name_for_combo = "master"
                if i < len(self.custom_chord_notes_data):
                    self.custom_chord_notes_data[i]['osc_idx'] = -1
            
            osc_combo.set(display_name_for_combo)
            osc_combo.grid(row=3, column=0, columnspan=2, pady=1, sticky="ew") # Span
            osc_combo.bind('<<ComboboxSelected>>', 
                           lambda e, idx=i, cb=osc_combo: self.on_custom_note_param_change(idx, 'osc_idx', cb.get()))

            # Beat Length Control
            beat_length_var = tk.DoubleVar(value=note_data.get('beat_length', 1.0))
            self.note_beat_length_vars.append(beat_length_var)
            ttk.Label(note_ui_frame, text="Len:").grid(row=4, column=0, columnspan=2, pady=(2,0), sticky="w") # Span, sticky w
            beat_length_spin = ttk.Spinbox(note_ui_frame, from_=0.1, to=8.0, increment=0.1, textvariable=beat_length_var, width=4) # Adjusted range/increment
            beat_length_spin.grid(row=5, column=0, columnspan=2, pady=1, sticky="ew") # Span
            # Using configure for command because Spinbox command doesn't pass the value directly in all tk versions easily
            beat_length_spin.configure(command=lambda v=beat_length_var, idx=i: self.on_custom_note_param_change(idx, 'beat_length', v.get()))
            # Bind FocusOut and Return to ensure value is captured if not changed by arrow keys/command
            beat_length_spin.bind('<FocusOut>', lambda e, v=beat_length_var, idx=i: self.on_custom_note_param_change(idx, 'beat_length', v.get()))
            beat_length_spin.bind('<Return>', lambda e, v=beat_length_var, idx=i: self.on_custom_note_param_change(idx, 'beat_length', v.get()))

            # The remove_btn is already gridded at the top (row=0, column=1)
        
        self.custom_notes_content_frame.update_idletasks()
        self.custom_notes_canvas.configure(scrollregion=self.custom_notes_canvas.bbox("all"))

    def on_custom_note_param_change(self, note_idx: int, param_type: str, new_value):
        """Called when a note's pitch, octave adjustment, oscillator, or beat length is changed in the UI."""
        if note_idx < len(self.custom_chord_notes_data):
            if param_type == 'octave_adjust':
                try: new_value = int(new_value)
                except ValueError: return
            elif param_type == 'beat_length':
                try: new_value = float(new_value)
                except ValueError: return
                if new_value < 0.1: new_value = 0.1 # Basic validation

            if param_type == 'pitch':
                self.custom_chord_notes_data[note_idx]['pitch'] = str(new_value)
            elif param_type == 'octave_adjust':
                 self.custom_chord_notes_data[note_idx]['octave_adjust'] = new_value
            elif param_type == 'beat_length':
                 self.custom_chord_notes_data[note_idx]['beat_length'] = new_value
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
        # Default beat_length to the main duration_var, or 1.0 if not suitable
        default_beat_length = self.duration_var.get()
        if default_beat_length <= 0: default_beat_length = 1.0

        self.custom_chord_notes_data.append({'pitch': new_note_pitch, 
                                            'octave_adjust': new_note_octave_adjust, 
                                            'osc_idx': -1,
                                            'beat_length': default_beat_length })
        
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
            
            # self.oscillator_ui_elements.pop(index, None) # remove specific index before full rebuild
            # Refresh UI - update_oscillator_frames_after_rename will clear and rebuild self.oscillator_ui_elements
            self.update_oscillator_frames_after_rename() 
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
        
        self.oscillator_ui_elements.clear() # Clear stored UI elements before rebuilding
        
        # Re-create all oscillator frames. This will use the new names.
        for i in range(len(self.chord_generator.oscillators)):
            self.create_oscillator_frame(self.oscillator_frame, i)
            self.update_waveform_preview(i) # Ensure preview is also updated

    def open_waveform_sculptor(self, oscillator_index=None):
        """Opens the waveform sculptor window."""
        sculptor_window = WaveformSculptor(self.root, 
                                           main_app_ref=self, 
                                           oscillator_index=oscillator_index)
        sculptor_window.grab_set() # Make it modal to focus user interaction

        def on_sculptor_close():
            print(f"Sculptor window for osc_idx {oscillator_index} is closing.")
            if oscillator_index is not None:
                try:
                    # No longer automatically turn off live editing on close.
                    # The live state persists if "Apply to Oscillator" was used.
                    # osc = self.chord_generator.oscillators[oscillator_index]
                    # if osc.is_live_editing: 
                    #     osc.set_live_edit_data(is_active=False)
                    #     print(f"Live editing turned off for oscillator {oscillator_index + 1}.")
                    
                    # Ensure main UI preview is up-to-date with the oscillator's current state (which might be live)
                    self.update_waveform_preview(oscillator_index) 
                    # Removing automatic preview_chord on close, sound should reflect last applied state.
                    # if hasattr(self, 'preview_chord'): 
                    #     self.preview_chord()
                except IndexError:
                    print(f"Warning: Could not access oscillator {oscillator_index} on sculptor close for UI update.")
                except Exception as e:
                    print(f"Error during sculptor close for osc {oscillator_index} UI update: {e}")
            sculptor_window.destroy()

        sculptor_window.protocol("WM_DELETE_WINDOW", on_sculptor_close)

    def populate_waveform_dropdown(self, reload_definitions=False):
        """
        Called by WaveformSculptor (or other parts of the app) to indicate 
        new waveforms might be available and oscillator dropdowns need refreshing.
        Assumes Oscillator class has a method load_waveform_definitions(force_reload=True)
        which reloads its self.waveform_definitions dictionary from the JSON file.
        """
        if reload_definitions:
            print("Main app: Received request to reload waveform definitions.")
            # Instruct each oscillator to reload its waveform definitions
            for osc_idx, osc in enumerate(self.chord_generator.oscillators):
                if hasattr(osc, 'load_waveform_definitions'):
                    try:
                        osc.load_waveform_definitions(force_reload=True)
                        # print(f"Oscillator {osc_idx} reloaded definitions.")
                    except Exception as e:
                        print(f"Error reloading definitions for oscillator {osc_idx}: {e}")
                else:
                    print(f"Warning: Oscillator {osc_idx} does not have 'load_waveform_definitions' method.")

            # Update the combobox values for each oscillator
            for i in range(len(self.chord_generator.oscillators)):
                if i in self.oscillator_ui_elements and 'wave_menu' in self.oscillator_ui_elements[i]:
                    wave_menu_combobox = self.oscillator_ui_elements[i]['wave_menu']
                    current_osc = self.chord_generator.oscillators[i]
                    
                    # Ensure waveform_definitions is populated
                    if hasattr(current_osc, 'waveform_definitions') and current_osc.waveform_definitions:
                        new_wave_values = list(current_osc.waveform_definitions.keys())
                    else:
                        new_wave_values = [] # Default to empty list if no definitions found
                        print(f"Warning: Oscillator {i} has no waveform definitions after reload attempt.")

                    current_selection = self.wave_vars[i].get()
                    
                    wave_menu_combobox['values'] = new_wave_values
                    
                    if current_selection not in new_wave_values and new_wave_values:
                        self.wave_vars[i].set(new_wave_values[0])
                        wave_menu_combobox.set(new_wave_values[0])
                    elif not new_wave_values:
                        self.wave_vars[i].set("")
                        wave_menu_combobox.set("")
                    else: # Selection is valid or new_wave_values is empty
                         wave_menu_combobox.set(current_selection) 
                else:
                    print(f"Warning: Could not find wave_menu for oscillator {i} to refresh.")
            
            print("Main app: Waveform dropdowns updated.")
            # After updating values, a general UI refresh for oscillators might be needed
            # if other parts depend on these definitions directly.
            # For now, just updating the comboboxes.
            # self.update_oscillator_frames_after_rename() # This is too heavy, avoid if possible

    def load_step_into_chord_settings(self, step_data: Dict):
        """Loads data from a sequencer step into the main chord/note editing UI."""
        if not isinstance(step_data, dict):
            print("Error: Invalid step_data format received.")
            return

        # Set master octave for the chord
        self.octave_var.set(step_data.get('master_octave', 4))
        
        # Set overall duration for the chord step
        self.duration_var.set(step_data.get('duration', 1.0))
        
        # Load notes data into the custom chord section
        # Ensure we're creating copies of note dicts to avoid direct modification of sequencer data
        loaded_notes_data = []
        for note_info in step_data.get('notes_data', []):
            if isinstance(note_info, dict):
                # Ensure all necessary keys are present with defaults, similar to _on_sequence_line_click
                complete_note = {
                    'pitch': note_info.get('pitch', 'C'),
                    'octave_adjust': note_info.get('octave_adjust', 0),
                    'osc_idx': note_info.get('osc_idx', -1),
                    'beat_length': note_info.get('beat_length', self.duration_var.get() or 1.0)
                }
                loaded_notes_data.append(complete_note.copy())
            else:
                print(f"Warning: Invalid note_info detected in step_data: {note_info}")

        self.custom_chord_notes_data = loaded_notes_data
        
        # Set chord type to "-- Custom --" as we are loading specific notes
        self.type_var.set("-- Custom --")
        
        # Refresh the custom notes UI to display the loaded notes
        self.update_custom_chord_note_ui()
        
        # Optional: Give focus to a relevant widget, e.g., the first note's pitch combobox if notes were loaded.
        # This might require more access to UI elements within update_custom_chord_note_ui or storing them.
        # For now, just updating data and UI is sufficient.
        print(f"Loaded step data into chord settings: Oct:{self.octave_var.get()}, Dur:{self.duration_var.get()}, Notes:{len(self.custom_chord_notes_data)}")

    def export_oscillator_rack(self):
        """Exports all current oscillator settings to a .shallotrack JSON file."""
        all_oscillators_data = []
        for index, osc_object in enumerate(self.chord_generator.oscillators):
            osc_data = {
                'name': osc_object.name,
                'enabled': self.enabled_vars[index].get(),
                'waveform_selection': self.wave_vars[index].get(),
                'is_live_editing': osc_object.is_live_editing,
                'live_edit_points': osc_object.live_edit_points if osc_object.is_live_editing else [],
                'amplitude': self.amp_vars[index].get(),
                'detune': self.detune_vars[index].get(),
                'attack': self.attack_vars[index].get(),
                'decay': self.decay_vars[index].get(),
                'sustain': self.sustain_vars[index].get(),
                'release': self.release_vars[index].get(),
                'filter_cutoff': self.cutoff_vars[index].get(),
                'filter_resonance': self.resonance_vars[index].get(),
                'pan': self.pan_vars[index].get(),
                'eq_gains': [var.get() for var in self.eq_gain_vars[index]]
            }
            all_oscillators_data.append(osc_data)

        if not all_oscillators_data:
            messagebox.showwarning("Export Rack", "No oscillators to export.")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".shallotrack",
            filetypes=[("Shallot Rack Files", "*.shallotrack"), ("All Files", "*.*")],
            title="Export Oscillator Rack"
        )
        if not filepath:
            return # User cancelled

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(all_oscillators_data, f, indent=4)
            messagebox.showinfo("Export Successful", f"Oscillator rack exported to\n{filepath}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export oscillator rack.\nError: {e}")

    def import_oscillator_rack(self):
        """Imports oscillator settings from a .shallotrack JSON file, replacing the current rack."""
        filepath = filedialog.askopenfilename(
            defaultextension=".shallotrack",
            filetypes=[("Shallot Rack Files", "*.shallotrack"), ("All Files", "*.*")],
            title="Import Oscillator Rack"
        )
        if not filepath:
            return # User cancelled

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                loaded_oscillators_data = json.load(f)
            
            if not isinstance(loaded_oscillators_data, list):
                raise ValueError("Invalid file format: expected a list of oscillators.")

            # Clear existing oscillators and their UI/data
            for widget in self.oscillator_frame.winfo_children():
                widget.destroy()
            
            self.chord_generator.oscillators = []
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
            self.solo_vars = [] # Reset solo states as well
            self.eq_gain_vars = []
            self.waveform_plots = {}
            self.waveform_canvases = {}
            self.oscillator_ui_elements = {}

            if not loaded_oscillators_data: # If the file was empty or contained an empty list
                # Add a default oscillator to not leave the UI empty
                self.add_oscillator() # This will call init_osc_controls, create_frame, update_preview
                messagebox.showinfo("Import Info", "Imported rack was empty. Reset to one default oscillator.")
                self.update_custom_chord_note_ui() # Refresh osc names in note dropdowns
                return

            # Populate with imported oscillators
            for osc_idx, osc_data in enumerate(loaded_oscillators_data):
                new_internal_idx = self.chord_generator.add_oscillator() # Creates Oscillator object
                self.init_oscillator_controls(new_internal_idx) # Creates tk.Vars for it

                osc_object = self.chord_generator.oscillators[new_internal_idx]
                
                # Set name on object first for UI frame title
                osc_object.name = osc_data.get('name', f"osc{new_internal_idx + 1}")

                # Set tk.Vars from loaded data
                self.enabled_vars[new_internal_idx].set(osc_data.get('enabled', True))
                self.wave_vars[new_internal_idx].set(osc_data.get('waveform_selection', 'sine'))
                self.amp_vars[new_internal_idx].set(osc_data.get('amplitude', 0.7))
                self.detune_vars[new_internal_idx].set(osc_data.get('detune', 0.0))
                self.attack_vars[new_internal_idx].set(osc_data.get('attack', 0.01))
                self.decay_vars[new_internal_idx].set(osc_data.get('decay', 0.1))
                self.sustain_vars[new_internal_idx].set(osc_data.get('sustain', 0.8))
                self.release_vars[new_internal_idx].set(osc_data.get('release', 0.2))
                self.cutoff_vars[new_internal_idx].set(osc_data.get('filter_cutoff', 1.0))
                self.resonance_vars[new_internal_idx].set(osc_data.get('filter_resonance', 0.0))
                self.pan_vars[new_internal_idx].set(osc_data.get('pan', 0.5))
                
                eq_gains_data = osc_data.get('eq_gains', [0.0] * 8)
                for band_idx, gain_val in enumerate(eq_gains_data):
                    if band_idx < len(self.eq_gain_vars[new_internal_idx]):
                        self.eq_gain_vars[new_internal_idx][band_idx].set(gain_val)
                
                # Set live editing state and points directly on the oscillator object
                is_live = osc_data.get('is_live_editing', False)
                live_points = osc_data.get('live_edit_points', [])
                osc_object.set_live_edit_data(points=live_points if is_live else None, is_active=is_live)

                # Apply tk.Var values to the oscillator object's attributes
                self.update_oscillator(new_internal_idx) # This also calls update_waveform_preview

            # Rebuild all UI frames for oscillators
            self.update_oscillator_frames_after_rename() 
            self.update_custom_chord_note_ui() # Refresh osc names in note dropdowns

            messagebox.showinfo("Import Successful", f"Oscillator rack imported from\n{filepath}")

        except ValueError as ve:
             messagebox.showerror("Import Error", f"Invalid file content in {filepath}.\nError: {ve}")
        except Exception as e:
            messagebox.showerror("Import Error", f"Failed to import oscillator rack from {filepath}.\nError: {e}")

if __name__ == "__main__":
    # Reset Oscillator count when app starts, if desired for consistent default naming like "Oscillator 1", "Oscillator 2" etc.
    # This is a design choice. If names are loaded from a file, this might not be wanted here.
    # Oscillator._osc_count = 0 # Moved to init_control_variables for better control if app can be re-initialized
    root = tk.Tk()
    app = ChordGeneratorApp(root)
    root.mainloop() 
