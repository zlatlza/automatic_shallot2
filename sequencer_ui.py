import tkinter as tk
from tkinter import ttk, messagebox
from typing import List, Dict
import numpy as np
import pygame

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
        seq_controls_frame = ttk.LabelFrame(self.master_frame, text="Sequencer", padding="5")
        seq_controls_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5) # Adjusted row from 1
        seq_controls_frame.grid_columnconfigure(1, weight=1)
        
        ttk.Label(seq_controls_frame, text="BPM:").grid(row=0, column=0, padx=2, pady=1)
        bpm_spin = ttk.Spinbox(seq_controls_frame, from_=40, to=300,
                              textvariable=self.bpm_var, width=4)
        bpm_spin.grid(row=0, column=1, padx=2, pady=1)

        # Sequence Display Text Area
        self.sequence_text = tk.Text(self.master_frame, height=6, width=50)
        self.sequence_text.grid(row=1, column=0, sticky="ew", padx=5, pady=5) # Adjusted row from 2

        # Play Controls Frame
        play_controls_frame = ttk.Frame(self.master_frame)
        play_controls_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5) # Adjusted row from 3
        
        buttons = [
            # "Preview Chord" is app-level, not sequencer specific for adding
            ("Add to Sequence", self.add_to_sequence),
            ("Play Sequence", self.play_sequence),
            ("Stop", self.stop_sequence),
            ("Clear Sequence", self.clear_sequence)
        ]
        
        for i, (text, command) in enumerate(buttons):
            btn = ttk.Button(play_controls_frame, text=text, command=command)
            btn.grid(row=0, column=i, padx=5, pady=5)
            play_controls_frame.grid_columnconfigure(i, weight=1) # Make buttons expand

    def _update_tempo(self, *args):
        try:
            bpm = self.bpm_var.get()
            if 40 <= bpm <= 300:
                self.sequencer.bpm = bpm
        except tk.TclError:
            pass

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