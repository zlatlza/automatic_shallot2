import tkinter as tk
from tkinter import ttk, messagebox
from typing import List, Dict
import numpy as np
import pygame
import re # For parsing sequence text

# Assuming ChordGenerator and its NOTE_FREQUENCIES are in main.py or accessible
# For now, we might need to pass some app_instance methods or ChordGenerator instance if needed directly

class Sequencer:
    def __init__(self):
        self.sequence = []
        self.bpm = 120
        self.current_step = 0
        self.is_playing = False
        
    def add_step(self, master_octave: int, duration: float, notes_data: List[Dict]):
        self.sequence.append({
            'master_octave': master_octave,
            'duration': duration,
            'notes_data': notes_data
        })
        
    def clear_sequence(self):
        self.sequence = []
        self.current_step = 0
        
    def get_step_duration_ms(self, duration: float) -> int:
        beats_per_second = self.bpm / 60
        ms_per_beat = 1000 / beats_per_second
        return int(ms_per_beat * duration)

class SequencerUI:
    def __init__(self, master_frame, app_instance):
        self.master_frame = master_frame
        self.app = app_instance # Reference to ChordGeneratorApp
        self.sequencer = Sequencer() # Logic instance

        # Control Variables that belong to the sequencer UI
        self.bpm_var = tk.IntVar(value=self.sequencer.bpm)
        self.bpm_var.trace_add('write', self._update_tempo)

        self.sequence_timer = None # For scheduling next step

        self._setup_gui()

    def _setup_gui(self):
        # Sequencer Controls Frame
        seq_controls_frame = ttk.LabelFrame(self.master_frame, text="Sequencer", padding="2")
        seq_controls_frame.grid(row=0, column=0, sticky="ew", padx=2, pady=(2,1))
        seq_controls_frame.grid_columnconfigure(1, weight=1)
        
        ttk.Label(seq_controls_frame, text="BPM:").grid(row=0, column=0, padx=(2,1), pady=1)
        bpm_spin = ttk.Spinbox(seq_controls_frame, from_=40, to=300,
                              textvariable=self.bpm_var, width=3)
        bpm_spin.grid(row=0, column=1, padx=(1,2), pady=1)

        # Sequence Display Text Area
        self.sequence_text = tk.Text(self.master_frame, height=5, width=40)
        self.sequence_text.grid(row=1, column=0, sticky="ew", padx=2, pady=1)

        # "Apply Text Changes" Button
        apply_text_btn = ttk.Button(self.master_frame, text="Apply", 
                                    command=self._apply_text_to_sequence)
        apply_text_btn.grid(row=2, column=0, sticky="ew", padx=2, pady=(1,2))

        # Play Controls Frame
        play_controls_frame = ttk.Frame(self.master_frame)
        play_controls_frame.grid(row=3, column=0, sticky="ew", padx=2, pady=1)
        
        buttons = [
            ("Preview Chord", self.app.preview_chord),
            ("Add to Seq", self.add_to_sequence),
            ("Play Seq", self.play_sequence),
            ("Stop", self.stop_sequence),
            ("Clear Seq", self.clear_sequence)
        ]
        
        for i, (text, command) in enumerate(buttons):
            btn = ttk.Button(play_controls_frame, text=text, command=command)
            btn.grid(row=0, column=i, padx=2, pady=1)
            play_controls_frame.grid_columnconfigure(i, weight=1)

    def _update_tempo(self, *args):
        try:
            bpm = self.bpm_var.get()
            if 40 <= bpm <= 300:
                self.sequencer.bpm = bpm
        except tk.TclError:
            pass

    def _apply_text_to_sequence(self):
        """Parses the content of the sequence_text widget and updates the internal sequence data."""
        all_text = self.sequence_text.get("1.0", tk.END).strip()
        if not all_text:
            # If text area is empty, clear the sequence
            if messagebox.askyesno("Clear Sequence?", "Text area is empty. Clear the current sequence?"):
                self.sequencer.clear_sequence()
                self.update_sequence_display()
            return

        lines = all_text.split('\n')
        new_sequence_data = []
        
        # Regex for the overall line structure
        line_regex = re.compile(r"^\d+\.\s*Notes:\s*(.+?)\s*\(Oct:\s*(-?\d+),\s*([\d.]+)\s*beats\)"
                               )
        # Regex for individual notes like "C+0" or "Db-1"
        note_regex = re.compile(r"([A-Ga-g][#b]?)\s*([+-]\d+)")
        # Valid note pitches from ChordGenerator (assuming it's accessible like this)
        # This is a simplification; ideally, ChordGenerator.NOTE_FREQUENCIES would be passed or accessed more directly
        valid_pitches = list(self.app.chord_generator.NOTE_FREQUENCIES.keys())

        for line_num, line_content in enumerate(lines, 1):
            line_content = line_content.strip()
            if not line_content: # Skip empty lines if any
                continue

            match = line_regex.match(line_content)
            if not match:
                messagebox.showerror("Parsing Error", f"Line {line_num} does not match expected format.\nExpected: 'X. Notes: P+O, ... (Oct: M, D beats)'")
                return

            notes_summary_str, master_octave_str, duration_str = match.groups()
            
            try:
                master_octave = int(master_octave_str)
                duration = float(duration_str)
                if duration <= 0:
                    raise ValueError("Duration must be positive.")
            except ValueError as e:
                messagebox.showerror("Parsing Error", f"Line {line_num}: Invalid master octave or duration.\nMaster Octave: '{master_octave_str}', Duration: '{duration_str}'. Error: {e}")
                return

            parsed_notes_data = []
            if notes_summary_str.lower() == "(no notes)":
                # Allow explicitly stating no notes, treat as empty step for sound gen (will be skipped)
                pass 
            else:
                note_parts = notes_summary_str.split(',')
                if not note_parts or not note_parts[0].strip(): # Handle case of empty notes_summary_str
                     messagebox.showerror("Parsing Error", f"Line {line_num}: Notes section is empty or invalid.")
                     return

                for note_idx, note_part_str in enumerate(note_parts, 1):
                    note_part_str = note_part_str.strip()
                    note_match = note_regex.match(note_part_str)
                    if not note_match:
                        messagebox.showerror("Parsing Error", f"Line {line_num}, Note {note_idx} ('{note_part_str}'): Invalid note format. Expected 'P+O' (e.g., C#+1).")
                        return
                    
                    pitch, octave_adjust_str = note_match.groups()
                    try:
                        octave_adjust = int(octave_adjust_str)
                    except ValueError:
                        messagebox.showerror("Parsing Error", f"Line {line_num}, Note {note_idx} ('{note_part_str}'): Invalid octave adjustment. Must be integer.")
                        return

                    # Validate pitch case-insensitively but store the canonical form (uppercase)
                    canonical_pitch = pitch.upper().replace('B#', 'C').replace('E#', 'F') # Basic enharmonic equivalents for validation
                    # A more robust enharmonic handling might be needed if users input B# etc. expecting it
                    # For now, we check against the defined NOTE_FREQUENCIES keys
                    
                    # The regex for pitch already captures sharps and flats correctly (e.g. C#, Db)
                    # We need to ensure the captured pitch (potentially like 'db') is compared correctly to keys like 'C#', 'Db'
                    # Simplest is to ensure valid_pitches contains all expected forms or normalize input pitch before check.
                    # Current NOTE_FREQUENCIES uses 'C#', 'D#', 'F#', 'G#', 'A#' for sharps. Flats are not used in keys.
                    # Let's assume user might type 'Db' and we should recognize it if 'C#' is the key.
                    # This requires a mapping or more complex validation than direct `in valid_pitches` if we support typed flats. 
                    # For now, we stick to what NOTE_FREQUENCIES defines.
                    # Let's make the check: pitch.upper() in valid_pitches OR an equivalent is in valid_pitches.
                    # The simplest for now: ensure user types sharps as per NOTE_FREQUENCIES or we add flat equivalents to NOTE_FREQUENCIES.
                    # For this iteration, we will assume the user enters notes as they appear in NOTE_FREQUENCIES (e.g. C# not Db unless Db is also a key)
                    # So, pitch.upper() should be sufficient if NOTE_FREQUENCIES keys are all uppercase.

                    found_pitch = False
                    final_pitch_to_store = ""
                    # Check for direct match (e.g. C, D, E) or sharp match (e.g. C#)
                    if pitch.upper() in valid_pitches:
                        found_pitch = True
                        final_pitch_to_store = pitch.upper()
                    else: 
                        # Handle common flat-to-sharp conversions for validation if user typed a flat
                        # e.g. if user types Db, convert to C# for checking against valid_pitches
                        enharmonic_map = {
                            'DB': 'C#', 'EB': 'D#', 'FB': 'E', 
                            'GB': 'F#', 'AB': 'G#', 'BB': 'A#'
                        }
                        normalized_input_pitch = pitch.upper()
                        if normalized_input_pitch in enharmonic_map and enharmonic_map[normalized_input_pitch] in valid_pitches:
                            found_pitch = True
                            final_pitch_to_store = enharmonic_map[normalized_input_pitch]
                        elif normalized_input_pitch.endswith('B') and len(normalized_input_pitch) == 2 and normalized_input_pitch[0]+"#" in valid_pitches: # e.g. User typed CB, check for B
                             # This case is less common. Cb is B. Fb is E.
                             pass # Covered by Fb -> E above for example

                    if not found_pitch:
                        messagebox.showerror("Parsing Error", f"Line {line_num}, Note {note_idx} ('{pitch}'): Invalid or unsupported note pitch. Use standard sharps (e.g., C#, F#). Supported: {valid_pitches}")
                        return
                    
                    parsed_notes_data.append({'pitch': final_pitch_to_store, 'octave_adjust': octave_adjust})
            
            new_sequence_data.append({
                'master_octave': master_octave,
                'duration': duration,
                'notes_data': parsed_notes_data
            })

        # If all parsing is successful
        self.sequencer.sequence = new_sequence_data # Replace the old sequence
        self.update_sequence_display() # Refresh the text display (normalizes format)
        messagebox.showinfo("Sequence Updated", "Sequence successfully updated from text.")

    def add_to_sequence(self):
        current_notes_data = self.app.get_current_custom_notes_for_sound()
        if not current_notes_data:
            messagebox.showwarning("Add to Sequence", "Cannot add an empty chord to sequence.")
            return

        self.sequencer.add_step(
            master_octave=self.app.octave_var.get(), 
            duration=self.app.duration_var.get(), 
            notes_data=current_notes_data
        )
        self.update_sequence_display()
    
    def update_sequence_display(self):
        self.sequence_text.delete(1.0, tk.END)
        for i, step in enumerate(self.sequencer.sequence):
            notes_summary = ", ".join([f"{n['pitch']}{n['octave_adjust']:+d}" for n in step['notes_data']])
            if not notes_summary: notes_summary = "(No notes)"
            self.sequence_text.insert(tk.END, 
                f"{i+1}. Notes: {notes_summary} (Oct: {step['master_octave']}, {step['duration']} beats)\n")

    def play_sequence(self):
        if not self.sequencer.sequence:
            messagebox.showinfo("Play Sequence", "Sequence is empty.")
            return
            
        self.sequencer.is_playing = True
        self.sequencer.current_step = 0
        self.play_next_step()
    
    def play_next_step(self):
        if not self.sequencer.is_playing or self.sequencer.current_step >= len(self.sequencer.sequence):
            self.sequencer.current_step = 0
            self.sequencer.is_playing = False
            if self.sequence_timer:
                self.app.root.after_cancel(self.sequence_timer)
                self.sequence_timer = None
            return
            
        step = self.sequencer.sequence[self.sequencer.current_step]
        
        notes_data_for_step = step.get('notes_data', [])
        if not notes_data_for_step:
            print(f"Sequencer step {self.sequencer.current_step + 1} has no notes. Skipping.")
            self.sequencer.current_step += 1
            if self.sequencer.is_playing and self.sequencer.current_step < len(self.sequencer.sequence):
                ms_per_beat = 60000 / self.sequencer.bpm
                next_time = int(ms_per_beat * step['duration'])
                self.sequence_timer = self.app.root.after(max(1, next_time), self.play_next_step)
            else:
                self.sequencer.is_playing = False
            return

        ms_per_beat = 60000 / self.sequencer.bpm
        crossfade_ms = min(100, ms_per_beat / 4)

        master_vol = self.app.master_volume_var.get()
        master_mute_status = self.app.master_mute_var.get()

        current_solo_idx = -1 
        for idx, solo_var in enumerate(self.app.solo_vars):
            if solo_var.get():
                current_solo_idx = idx
                break

        samples = self.app.chord_generator.generate_chord(
            master_octave=step['master_octave'],
            notes_data=notes_data_for_step,
            duration_ms=self.sequencer.get_step_duration_ms(step['duration']),
            fade_out_ms=int(crossfade_ms),
            master_volume=master_vol,
            master_mute=master_mute_status,
            soloed_osc_idx=current_solo_idx
        )
        
        if samples.size > 0:
            sound_array = (samples * 32767).astype(self.app.np.int16) # Use app's numpy
            sound = self.app.pygame.sndarray.make_sound(sound_array) # Use app's pygame
            sound.play()
        else:
            print(f"Step {self.sequencer.current_step + 1}: Generated empty samples.")
        
        self.sequencer.current_step += 1
        next_event_time_ms = self.sequencer.get_step_duration_ms(step['duration']) - int(crossfade_ms)
        next_event_time_ms = max(1, next_event_time_ms)
        
        if self.sequence_timer:
             self.app.root.after_cancel(self.sequence_timer)
        self.sequence_timer = self.app.root.after(next_event_time_ms, self.play_next_step)
    
    def stop_sequence(self):
        self.sequencer.is_playing = False
        if self.sequence_timer:
            self.app.root.after_cancel(self.sequence_timer)
            self.sequence_timer = None
        self.app.pygame.mixer.stop() # Use app's pygame
    
    def clear_sequence(self):
        self.stop_sequence() # Stop playback before clearing
        self.sequencer.clear_sequence()
        self.update_sequence_display() 