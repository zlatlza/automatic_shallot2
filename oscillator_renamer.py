import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from main import ChordGeneratorApp # To avoid circular import for type hinting

class OscillatorRenamerDialog(tk.Toplevel):
    def __init__(self, parent: 'ChordGeneratorApp'):
        super().__init__(parent.root)
        self.transient(parent.root)
        self.title("Rename Oscillators")
        self.parent_app = parent
        self.chord_generator = parent.chord_generator
        
        self.entries = {} # To store Entry widgets, keyed by oscillator index

        self.main_frame = ttk.Frame(self, padding="10")
        self.main_frame.pack(expand=True, fill=tk.BOTH)

        self.create_widgets()
        self.load_oscillator_names()

        self.grab_set() # Make dialog modal
        self.protocol("WM_DELETE_WINDOW", self._on_close) # Handle window close button

        # Center the dialog on the parent window
        self.update_idletasks()
        parent_x = parent.root.winfo_x()
        parent_y = parent.root.winfo_y()
        parent_width = parent.root.winfo_width()
        parent_height = parent.root.winfo_height()
        
        dialog_width = self.winfo_width()
        dialog_height = self.winfo_height()
        
        position_x = parent_x + (parent_width // 2) - (dialog_width // 2)
        position_y = parent_y + (parent_height // 2) - (dialog_height // 2)
        
        self.geometry(f"{dialog_width}x{dialog_height}+{position_x}+{position_y}")


    def create_widgets(self):
        ttk.Label(self.main_frame, text="Edit Oscillator Names:", font=('TkDefaultFont', 10, 'bold')).pack(pady=(0,10))

        self.osc_list_frame = ttk.Frame(self.main_frame)
        self.osc_list_frame.pack(expand=True, fill=tk.BOTH, pady=5)

        # Buttons
        button_frame = ttk.Frame(self.main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))

        self.apply_button = ttk.Button(button_frame, text="Apply", command=self._apply_changes)
        self.apply_button.pack(side=tk.RIGHT, padx=5)

        self.cancel_button = ttk.Button(button_frame, text="Cancel", command=self._on_close)
        self.cancel_button.pack(side=tk.RIGHT)

    def load_oscillator_names(self):
        # Clear any existing widgets in osc_list_frame
        for widget in self.osc_list_frame.winfo_children():
            widget.destroy()
        self.entries.clear()

        for i, osc in enumerate(self.chord_generator.oscillators):
            row_frame = ttk.Frame(self.osc_list_frame)
            row_frame.pack(fill=tk.X, pady=2)

            label_text = f"Osc {i + 1}:" # Original identifier
            ttk.Label(row_frame, text=label_text, width=10).pack(side=tk.LEFT, padx=(0,5))
            
            entry_var = tk.StringVar(value=osc.name)
            entry = ttk.Entry(row_frame, textvariable=entry_var, width=30)
            entry.pack(side=tk.LEFT, expand=True, fill=tk.X)
            self.entries[i] = entry_var # Store the StringVar

    def _apply_changes(self):
        any_changes = False
        for index, str_var in self.entries.items():
            new_name = str_var.get().strip()
            if not new_name: # Prevent empty names, revert to a default or original
                # For simplicity, let's just prevent empty and keep original if attempted
                # Or, assign a default like "Osc N"
                original_osc_name = self.chord_generator.oscillators[index].name
                str_var.set(original_osc_name) # Reset entry to original name if it was cleared
                continue # Skip applying empty name

            if self.chord_generator.oscillators[index].name != new_name:
                self.chord_generator.oscillators[index].name = new_name
                any_changes = True
        
        if any_changes:
            # Refresh the main app's UI elements that depend on oscillator names
            self.parent_app.update_oscillator_frames_after_rename() # A new method we'll add to main app
            self.parent_app.update_custom_chord_note_ui() # This updates note comboboxes
            if hasattr(self.parent_app, 'sequencer_ui_instance') and self.parent_app.sequencer_ui_instance:
                self.parent_app.sequencer_ui_instance.update_sequence_display() # Refresh sequencer display
        
        self.destroy() # Close dialog

    def _on_close(self):
        self.destroy()

if __name__ == '__main__':
    # This is for testing the dialog independently if needed
    # You'd need to mock the parent app and chord_generator
    class MockOscillator:
        def __init__(self, name):
            self.name = name

    class MockChordGenerator:
        def __init__(self):
            self.oscillators = [MockOscillator("Default Osc 1"), MockOscillator("Synth Pad"), MockOscillator("Bass Wobble")]
    
    class MockApp:
        def __init__(self):
            self.root = tk.Tk()
            self.root.title("Mock Parent App")
            self.chord_generator = MockChordGenerator()
            # Dummy methods that the dialog might call
            self.update_oscillator_frames_after_rename = lambda: print("Mock: update_oscillator_frames_after_rename called")
            self.update_custom_chord_note_ui = lambda: print("Mock: update_custom_chord_note_ui called")
            
            # Button to launch the dialog
            btn = ttk.Button(self.root, text="Open Renamer", command=self.open_renamer_dialog)
            btn.pack(padx=20, pady=20)
            self.root.geometry("300x100")

        def open_renamer_dialog(self):
            dialog = OscillatorRenamerDialog(self)
            self.root.wait_window(dialog) # Wait for dialog to close

    app = MockApp()
    app.root.mainloop() 