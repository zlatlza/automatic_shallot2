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
        seq_controls_frame = ttk.LabelFrame(self.master_frame, text="Sequencer", padding="2")
        seq_controls_frame.grid(row=0, column=0, sticky="ew", padx=2, pady=(2,1))
        seq_controls_frame.grid_columnconfigure(1, weight=1)
        
        ttk.Label(seq_controls_frame, text="BPM:").grid(row=0, column=0, padx=(2,1), pady=1)
        bpm_spin = ttk.Spinbox(seq_controls_frame, from_=40, to=300,
                              textvariable=self.bpm_var, width=3)
        bpm_spin.grid(row=0, column=1, padx=(1,2), pady=1)

        # --- Sequence Text Area with Scrollbars ---
        text_frame = ttk.Frame(self.master_frame)
        text_frame.grid(row=1, column=0, sticky="nsew", padx=2, pady=1)
        self.master_frame.grid_rowconfigure(1, weight=1) # Allow text_frame to expand
        self.master_frame.grid_columnconfigure(0, weight=1)

        self.sequence_text = tk.Text(text_frame, height=5, width=40, wrap=tk.NONE) # wrap=tk.NONE for horizontal scroll
        self.sequence_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        v_scrollbar_text = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.sequence_text.yview)
        v_scrollbar_text.pack(side=tk.RIGHT, fill=tk.Y)
        self.sequence_text.config(yscrollcommand=v_scrollbar_text.set)

        h_scrollbar_text = ttk.Scrollbar(self.master_frame, orient=tk.HORIZONTAL, command=self.sequence_text.xview)
        h_scrollbar_text.grid(row=2, column=0, sticky="ew", padx=2)
        self.sequence_text.config(xscrollcommand=h_scrollbar_text.set)
        # --- End Sequence Text Area ---

        apply_text_btn = ttk.Button(self.master_frame, text="Apply", 
                                    command=self._apply_text_to_sequence)
        # Row for apply button needs to be after h_scrollbar_text
        apply_text_btn.grid(row=3, column=0, sticky="ew", padx=2, pady=(1,2))

        play_controls_frame = ttk.Frame(self.master_frame)
        # Row for play_controls needs to be after apply_text_btn
        play_controls_frame.grid(row=4, column=0, sticky="ew", padx=2, pady=1)
        
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
        print("--- _apply_text_to_sequence called ---") # DEBUG
        all_text = self.sequence_text.get("1.0", tk.END).strip()
        print(f"Raw text from widget:\n'''{all_text}'''") # DEBUG

        if not all_text:
            if messagebox.askyesno("Clear Sequence?", "Text area is empty. Clear the current sequence?"):
                self.sequencer.clear_sequence()
                self.update_sequence_display()
            return

        lines = all_text.split('\n')
        print(f"Split lines: {lines}") # DEBUG
        new_sequence_data = []
        
        # Regex for overall line: "1. Oct:4 | Dur:1.0 beats | Notes: C+0[osc1], E+0[osc1]"
        # Groups: 1=step_num (ignored), 2=master_octave, 3=duration, 4=notes_string
        line_regex = re.compile(r"^(\d+)\.\s*Oct:(-?\d+)\s*\|\s*Dur:([\d.]+)\s*beats\s*\|\s*Notes:\s*(.+)$")
        
        # Regex for individual note: "C+0[osc1]" or "G#-1[Master/All]" or "A+0[Some Osc Name]"
        # Groups: 1=pitch, 2=octave_adjust, 3=osc_name
        # Allowing spaces in osc_name: [\w\s\-/()!]+ for more flexible osc names, or be stricter if names are constrained.
        # For now, let's assume osc names can have word characters, spaces, hyphens.
        note_regex = re.compile(r"([A-Ga-g][#b]?)([+-]\d+)\[([\\w\\s\\-]+|\"(?:[^\"]+)\"|\'(?:[^\']+)\')\]")
        # Simpler osc_name if no spaces: \[([\\w\\-]+)\]
        # Current osc_name_regex: ([\\w\\s\\-]+) - allows words, spaces, hyphens. 
        # If oscillator names can contain brackets or other special characters, this regex for osc_name would need to be more robust.
        # Let's use a version that matches anything inside brackets not being a closing bracket: \[([^]]+)\]
        note_regex_final = re.compile(r"([A-Ga-g][#b]?)([+-]\d+)\[([^]]+)\]")

        valid_pitches = list(self.app.chord_generator.NOTE_FREQUENCIES.keys())
        all_osc_objects = self.app.chord_generator.oscillators

        for line_num, line_content in enumerate(lines, 1):
            line_content = line_content.strip()
            print(f"Processing line {line_num}: '{line_content}'") # DEBUG
            if not line_content: 
                print(f"Line {line_num} is empty, skipping.") # DEBUG
                continue

            match = line_regex.match(line_content)
            if not match:
                print(f"DEBUG: line_regex MISMATCH on line: '{line_content}'") # DEBUG
                messagebox.showerror("Parsing Error", f"Line {line_num}: Structure mismatch.\nExpected format: 'X. Oct:M | Dur:D beats | Notes: P+O[Osc], ...'")
                return

            _step_num_str, master_octave_str, duration_str, notes_summary_str = match.groups()
            
            try:
                master_octave = int(master_octave_str)
                duration = float(duration_str)
                if duration <= 0: raise ValueError("Duration must be positive.")
            except ValueError as e:
                messagebox.showerror("Parsing Error", f"Line {line_num}: Invalid master octave or duration. Error: {e}")
                return

            parsed_notes_data = []
            if notes_summary_str.strip().lower() == "(no notes)":
                pass # Valid empty step
            else:
                note_parts_str = notes_summary_str.split(',')
                if not note_parts_str or not note_parts_str[0].strip():
                    messagebox.showerror("Parsing Error", f"Line {line_num}: Notes section is empty or invalid after 'Notes:'.")
                    return

                for note_idx_in_line, note_part_str in enumerate(note_parts_str, 1):
                    note_part_str = note_part_str.strip()
                    note_match = note_regex_final.match(note_part_str)
                    if not note_match:
                        messagebox.showerror("Parsing Error", 
                                             f"Line {line_num}, Note entry '{note_part_str}': Invalid format.\nExpected: 'P+/-O[OscName]' (e.g., C+0[osc1] or G#-1[master])")
                        return
                    
                    pitch_str, octave_adjust_str, osc_name_str = note_match.groups()
                    
                    try:
                        octave_adjust = int(octave_adjust_str)
                    except ValueError:
                        messagebox.showerror("Parsing Error", f"Line {line_num}, Note '{pitch_str}': Invalid octave adjustment '{octave_adjust_str}'.")
                        return

                    # Pitch validation (simplified from original, can be expanded with enharmonics)
                    canonical_pitch = pitch_str.upper()
                    if canonical_pitch not in valid_pitches:
                         # Try common flat to sharp conversion for robust parsing
                        enharmonic_map = {'DB':'C#', 'EB':'D#', 'FB':'E', 'GB':'F#', 'AB':'G#', 'BB':'A#'}
                        if canonical_pitch in enharmonic_map and enharmonic_map[canonical_pitch] in valid_pitches:
                            canonical_pitch = enharmonic_map[canonical_pitch]
                        else:
                            messagebox.showerror("Parsing Error", f"Line {line_num}, Note '{pitch_str}': Invalid pitch. Supported: {valid_pitches}")
                            return
                    
                    # Oscillator name/index resolution
                    assigned_osc_idx = -1 # Default to Master/All
                    osc_name_str_cleaned = osc_name_str.strip()
                    
                    if osc_name_str_cleaned.lower() == "master/all":
                        assigned_osc_idx = -1
                    else:
                        found_osc = False
                        for current_osc_idx, osc_obj in enumerate(all_osc_objects):
                            if osc_obj.name == osc_name_str_cleaned:
                                assigned_osc_idx = current_osc_idx
                                found_osc = True
                                break
                        if not found_osc:
                            # Handle special debug name from display like "osc_idx_2(!)_"
                            if osc_name_str_cleaned.startswith("osc_idx_") and osc_name_str_cleaned.endswith("(!)_"):
                                try:
                                    parsed_original_idx = int(osc_name_str_cleaned.split('_')[2])
                                    # This was an oscillator that existed but was removed. What to do?
                                    # Option: default to Master/All with a warning, or error out.
                                    # For now, let's default and allow the user to fix it.
                                    print(f"Warning: Line {line_num}, Note '{pitch_str}': Oscillator '{osc_name_str}' (original index {parsed_original_idx}) no longer exists. Defaulting to Master/All.")
                                    assigned_osc_idx = -1 # Default to Master/All
                                except: # Broad except if parsing the debug name fails
                                    messagebox.showerror("Parsing Error", f"Line {line_num}, Note '{pitch_str}': Oscillator name '{osc_name_str}' not found.")
                                    return
                            else:
                                messagebox.showerror("Parsing Error", f"Line {line_num}, Note '{pitch_str}': Oscillator name '{osc_name_str}' not found.")
                                return

                    parsed_notes_data.append({
                        'pitch': canonical_pitch, 
                        'octave_adjust': octave_adjust, 
                        'osc_idx': assigned_osc_idx
                    })
           
            new_sequence_data.append({
                'master_octave': master_octave,
                'duration': duration,
                'notes_data': parsed_notes_data
            })

        self.sequencer.sequence = new_sequence_data
        self.update_sequence_display() # Refresh display with potentially normalized/corrected data
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
        self.sequence_text.delete(1.0, tk.END) # Clear existing text
        
        for i, step in enumerate(self.sequencer.sequence):
            notes_summary_parts = []
            for note_data in step['notes_data']:
                pitch = note_data['pitch']
                oct_adj = note_data['octave_adjust']
                osc_idx = note_data.get('osc_idx', -1) 
                
                osc_display_name = "master"
                if 0 <= osc_idx < len(self.app.chord_generator.oscillators):
                    osc_display_name = self.app.chord_generator.oscillators[osc_idx].name
                elif osc_idx != -1: # Should ideally not happen if data is clean
                    # This indicates an invalid index, perhaps an oscillator was removed
                    # The parser will need to handle this gracefully, maybe default to Master/All
                    osc_display_name = f"osc_idx_{osc_idx}(!)_" # Make it distinct for parsing/debugging
                
                notes_summary_parts.append(f"{pitch}{oct_adj:+d}[{osc_display_name}]")
            
            notes_final_summary = ", ".join(notes_summary_parts) if notes_summary_parts else "(No notes)"
            
            line = f"{i+1}. Oct:{step['master_octave']} | Dur:{step['duration']} beats | Notes: {notes_final_summary}\n"
            self.sequence_text.insert(tk.END, line)

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