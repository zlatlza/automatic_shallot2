import tkinter as tk
from tkinter import ttk, messagebox
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class WaveformSculptor(tk.Toplevel):
    def __init__(self, master, main_app_ref, oscillator_index=None, existing_waveforms_path="waveform_definitions.json", num_points=16):
        super().__init__(master)
        self.title_base = "Waveform Sculptor"
        self.main_app_ref = main_app_ref 
        self.oscillator_index = oscillator_index
        self.master_window = master 
        self.existing_waveforms_path = existing_waveforms_path
        self.num_points = num_points
        self.geometry("600x500") # Adjust height for new buttons
        self._last_reported_sculptor_size = "" # For resize helper

        self.waveform_values = [tk.DoubleVar(value=0.0) for _ in range(self.num_points)]
        loaded_waveform_name = "MyNewWaveform" # Default for new waveform
        title_suffix = "(New)"

        if self.oscillator_index is not None and self.main_app_ref:
            title_suffix = f"(Oscillator {self.oscillator_index + 1})"
            try:
                osc = self.main_app_ref.chord_generator.oscillators[self.oscillator_index]
                initial_points = []
                if osc.is_live_editing and osc.live_edit_points:
                    initial_points = list(osc.live_edit_points) # Use a copy
                    loaded_waveform_name = osc.base_waveform_for_live_edit or osc.waveform
                    title_suffix += " - Editing Unsaved"
                else:
                    loaded_waveform_name = osc.waveform
                    initial_points = osc.get_waveform_cycle_points(loaded_waveform_name, self.num_points)
                
                if len(initial_points) == self.num_points:
                    for i, point_val in enumerate(initial_points):
                        self.waveform_values[i].set(float(point_val))
                else:
                    print(f"Warning: Initial points length ({len(initial_points)}) didn't match num_points ({self.num_points}). Using defaults.")
            except IndexError:
                messagebox.showerror("Error", "Invalid oscillator index provided to sculptor.")
                title_suffix = "(Error)"
            except Exception as e:
                print(f"Error loading initial waveform for sculptor: {e}")
                messagebox.showerror("Error", f"Could not load initial waveform: {e}")
                title_suffix = "(Error)"
        
        self.title(f"{self.title_base} {title_suffix}")

        # Frame for Matplotlib plot
        plot_frame = ttk.Frame(self)
        plot_frame.pack(pady=10, padx=10, fill="x")

        self.fig, self.ax = plt.subplots(figsize=(5, 2)) # Smaller figure size
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill="both", expand=True)

        # Frame for sliders
        sliders_frame = ttk.Frame(self)
        sliders_frame.pack(pady=10, padx=10, fill="both", expand=True)

        # Configure columns for sliders_frame to expand
        for i in range(self.num_points):
            sliders_frame.columnconfigure(i, weight=1)

        self.sliders = []
        for i in range(self.num_points):
            slider = ttk.Scale(sliders_frame, from_=1.0, to=-1.0, orient="vertical",
                               variable=self.waveform_values[i], command=self.update_plot)
            slider.grid(row=0, column=i, padx=2, pady=5, sticky="ns")
            self.sliders.append(slider)
            # Add labels for sliders (0 to num_points-1)
            # label = ttk.Label(sliders_frame, text=str(i))
            # label.grid(row=1, column=i, padx=2)


        # Frame for name entry and action buttons
        control_frame = ttk.Frame(self)
        control_frame.pack(pady=10, padx=10, fill="x")

        ttk.Label(control_frame, text="Waveform Name:").pack(side="left", padx=5)
        self.name_entry = ttk.Entry(control_frame, width=20) # Shorter width
        self.name_entry.insert(0, loaded_waveform_name)
        self.name_entry.pack(side="left", padx=5)

        # Frame for main action buttons (Preview, Apply, Save)
        action_buttons_frame = ttk.Frame(self)
        action_buttons_frame.pack(pady=(0,5), padx=10, fill="x") # Less pady at top

        self.preview_button = ttk.Button(action_buttons_frame, text="Preview", command=self.preview_sculpted_waveform)
        self.preview_button.pack(side="left", padx=2)
        
        self.apply_button = ttk.Button(action_buttons_frame, text="Apply to Osc", command=self.apply_to_oscillator)
        self.apply_button.pack(side="left", padx=2)

        self.save_button = ttk.Button(action_buttons_frame, text="Save Waveform", command=self.save_waveform)
        self.save_button.pack(side="left", padx=2)

        if self.oscillator_index is None: # Disable preview and apply if no specific oscillator
            self.preview_button.config(state="disabled")
            self.apply_button.config(state="disabled")

        # Frame for closing buttons
        closing_buttons_frame = ttk.Frame(self)
        closing_buttons_frame.pack(pady=(5,10), padx=10, fill="x")

        self.apply_close_button = ttk.Button(closing_buttons_frame, text="Apply & Close", command=self.apply_and_close)
        self.apply_close_button.pack(side="right", padx=2) # Pack right

        self.close_button = ttk.Button(closing_buttons_frame, text="Close", command=self.cancel_and_close)
        self.close_button.pack(side="right", padx=2) # Pack right, will be to the left of Apply & Close

        if self.oscillator_index is None:
            self.apply_close_button.config(state="disabled")

        self.update_plot() # Initial plot
        self.canvas.mpl_connect('button_press_event', self.on_canvas_click)
        self.protocol("WM_DELETE_WINDOW", self.cancel_and_close) # Handle 'X' button
        self.bind("<Configure>", self._on_resize_configure) # Bind resize event

    def _on_resize_configure(self, event):
        """Called when the sculptor window is resized or its configuration changes."""
        # Check if the event is for this Toplevel window itself
        if event.widget == self:
            current_width = self.winfo_width()
            current_height = self.winfo_height()
            size_str = f"WaveformSculptor Resized to: {current_width}x{current_height}"
            if size_str != self._last_reported_sculptor_size:
                print(size_str)
                self._last_reported_sculptor_size = size_str

    def update_plot(self, _=None):
        self.ax.clear()
        points = [var.get() for var in self.waveform_values]
        
        # We need to create an x-axis for these points.
        # If num_points is 16, x_indices will be 0, 1, ..., 15
        x_indices = np.arange(self.num_points)
        
        # To make the plot look like a continuous wave, we can either plot discrete points
        # or interpolate. For a simple representation of slider values, connecting them is fine.
        # If we want it to look like a single cycle, we should probably repeat the first point at the end
        # and adjust x accordingly. For now, let's just connect the points.
        
        plot_x = np.linspace(0, self.num_points -1, self.num_points) # Points from 0 to N-1
        
        # For a smoother visual wave, we can interpolate, but for direct representation of sliders,
        # just plotting points or a line connecting them is okay.
        # Let's use a simple line plot connecting the slider values.
        self.plotted_line, = self.ax.plot(plot_x, points, marker='o', linestyle='-')

        self.ax.set_ylim(-1.1, 1.1)
        self.ax.set_xlim(-0.5, self.num_points - 0.5) # Adjust x-limits based on number of points
        self.ax.set_xticks(x_indices) # Show ticks for each point
        self.ax.set_xticklabels([str(i) for i in x_indices]) # Label ticks 0 to N-1
        self.ax.set_title("Current Waveform")
        self.ax.grid(True)
        self.canvas.draw()

    def preview_sculpted_waveform(self):
        if self.oscillator_index is None or self.main_app_ref is None:
            messagebox.showinfo("Preview", "Cannot preview: No target oscillator specified.")
            return

        current_points = [var.get() for var in self.waveform_values]
        osc = self.main_app_ref.chord_generator.oscillators[self.oscillator_index]

        # Store original live edit state to restore after preview
        original_is_live = osc.is_live_editing
        original_live_points = list(osc.live_edit_points) if osc.live_edit_points is not None else []

        try:
            # Temporarily set for preview
            osc.set_live_edit_data(points=current_points, is_active=True)
            
            if hasattr(self.main_app_ref, 'preview_chord'):
                print("Sculptor: Triggering preview_chord in main app for sculpted points.")
                self.main_app_ref.preview_chord()
            else:
                messagebox.showerror("Preview Error", "Main application does not support direct preview.")

        except Exception as e:
            messagebox.showerror("Preview Error", f"An error occurred during preview: {e}")
        finally:
            # Restore original live edit state
            # This ensures that if "Apply" wasn't pressed, the preview doesn't make it stick.
            # If "Apply" WAS pressed, original_is_live would be True, so it correctly remains True.
            osc.set_live_edit_data(points=original_live_points if original_is_live else [], is_active=original_is_live)
            # No need to update main_app_ref.update_waveform_preview here, as this preview is ephemeral.

    def apply_to_oscillator(self):
        if self.oscillator_index is not None and self.main_app_ref:
            current_points = [var.get() for var in self.waveform_values]
            try:
                osc = self.main_app_ref.chord_generator.oscillators[self.oscillator_index]
                osc.set_live_edit_data(points=current_points, is_active=True)
                
                self.main_app_ref.update_waveform_preview(self.oscillator_index)
                print(f"Applied current sculpted waveform to Oscillator {self.oscillator_index + 1}")

                # Attempt to trigger preview in main app
                if hasattr(self.main_app_ref, 'preview_chord'):
                    self.main_app_ref.preview_chord() # This will play the current chord setup including live edit

            except IndexError:
                messagebox.showerror("Error", "Cannot apply: Invalid oscillator index.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to apply waveform: {e}")
        else:
            messagebox.showwarning("Apply Error", "No target oscillator specified for applying changes.")

    def on_canvas_click(self, event):
        if event.inaxes != self.ax or event.button != 1: # Check if click is in axes and is a left click
            return

        # Transform click from display to data coordinates
        x_data, y_data = self.ax.transData.inverted().transform((event.x, event.y))

        if not (0 <= x_data < self.num_points and -1.1 <= y_data <= 1.1):
             # Click is outside the relevant data area (e.g., too far left/right on x-axis, or outside y-plot limits)
             return

        # Find the closest point index based on x_data
        # self.num_points defines the number of discrete points on our x-axis (0 to num_points-1)
        # x_data is the click position in terms of these point indices
        closest_point_index = int(round(x_data))

        if 0 <= closest_point_index < self.num_points:
            # Clamp y_data to be within [-1.0, 1.0] as per slider limits
            clamped_y_value = max(-1.0, min(1.0, y_data))
            
            self.waveform_values[closest_point_index].set(clamped_y_value)
            # The slider linked to this DoubleVar will update automatically.
            self.update_plot() # Redraw the plot to reflect the change
            # Optionally, if auto-apply is desired on every click edit:
            # if self.oscillator_index is not None: self.apply_to_oscillator()

    def save_waveform(self):
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showerror("Error", "Waveform name cannot be empty.")
            return

        # The values from sliders are directly usable.
        waveform_data = [var.get() for var in self.waveform_values]

        try:
            with open(self.existing_waveforms_path, 'r+') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = {} # Initialize if file is empty or malformed
                
                if name in data:
                    if not messagebox.askyesno("Confirm", f"Waveform '{name}' already exists. Overwrite?"):
                        return
                
                data[name] = {"type": "sculpted", 
                              "points": waveform_data, 
                              "description": f"User-sculpted waveform: {name}"} # Add default description
                f.seek(0) # Go to the beginning of the file
                json.dump(data, f, indent=4)
                f.truncate() # Remove old content if new content is shorter
            
            messagebox.showinfo("Success", f"Waveform '{name}' saved successfully.")
            
            # First, notify the main app to refresh its waveform list globally
            # This ensures all oscillators, including the one being edited, get the new definition list.
            if hasattr(self.master_window, 'populate_waveform_dropdown'): 
                 if hasattr(self.main_app_ref, 'populate_waveform_dropdown'):
                     self.main_app_ref.populate_waveform_dropdown(reload_definitions=True)
                 else:
                    print("Debug: main_app_ref does not have populate_waveform_dropdown") 
            elif hasattr(self.main_app_ref, 'populate_waveform_dropdown'): 
                self.main_app_ref.populate_waveform_dropdown(reload_definitions=True)

            # Now, if this sculptor was tied to a specific oscillator, 
            # turn off live editing for it and set its waveform to the newly saved one.
            if self.oscillator_index is not None and self.main_app_ref:
                try:
                    osc = self.main_app_ref.chord_generator.oscillators[self.oscillator_index]
                    osc.set_live_edit_data(is_active=False) # Turn off live mode
                    
                    # Set the current oscillator's selection to the new waveform name.
                    # This will trigger update_oscillator -> update_waveform_preview in main_app.
                    self.main_app_ref.wave_vars[self.oscillator_index].set(name)
                    print(f"Oscillator {self.oscillator_index + 1} set to use newly saved waveform '{name}'.")

                    if hasattr(self.main_app_ref, 'preview_chord'):
                        self.main_app_ref.preview_chord()
                except IndexError:
                    print(f"Warning: Could not auto-select waveform for oscillator {self.oscillator_index + 1} after save.")
                except Exception as e:
                     print(f"Error post-save for oscillator {self.oscillator_index + 1}: {e}")

            self.destroy() # Close the sculptor window after saving

        except IOError:
            messagebox.showerror("Error", f"Could not open or write to {self.existing_waveforms_path}")
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")

    def cancel_and_close(self):
        """Closes the sculptor. The main app's on_sculptor_close will handle oscillator state."""
        # Any unsaved changes in the sculptor UI (that were not "Applied") are lost.
        # If changes were "Applied", the oscillator in main_app is already in live_edit mode.
        # main_app.on_sculptor_close (bound to WM_DELETE_WINDOW) will ensure its UI is synced.
        self.destroy()

    def apply_and_close(self):
        """Applies changes to the oscillator and then closes the sculptor."""
        if self.oscillator_index is not None:
            self.apply_to_oscillator()
            # No need to re-check errors from apply_to_oscillator, it shows its own messageboxes
        self.destroy() # This will trigger on_sculptor_close in main.py for UI updates

if __name__ == '__main__':
    # Example usage (for testing the sculptor independently)
    root = tk.Tk()
    root.title("Main App (Test)")
    
    # Dummy main_app_ref for testing WaveformSculptor independently
    class DummyMainApp:
        def __init__(self):
            class DummyOscillator:
                def __init__(self):
                    self.waveform_definitions = {"sine": {"type": "basic", "points": []}}
                    self.waveform = "sine"
                def set_live_edit_data(self, points, is_active):
                    print(f"DummyOscillator: live edit data set (active: {is_active}), points: {points[:5]}...")
                def load_waveform_definitions(self, force_reload=False):
                    print("DummyOscillator: load_waveform_definitions called")
            class DummyChordGen:
                def __init__(self):
                    self.oscillators = [DummyOscillator() for _ in range(2)] # Two dummy oscillators
            self.chord_generator = DummyChordGen()
            self.wave_vars = [tk.StringVar(value="sine") for _ in range(2)] # Dummy wave_vars

        def update_waveform_preview(self, index):
            print(f"DummyMainApp: update_waveform_preview for oscillator {index}")
        def populate_waveform_dropdown(self, reload_definitions=False):
            print(f"DummyMainApp: populate_waveform_dropdown (reload: {reload_definitions})")

    dummy_app = DummyMainApp()

    def open_sculptor_for_osc_0():
        WaveformSculptor(root, main_app_ref=dummy_app, oscillator_index=0)
    
    def open_new_sculptor():
        WaveformSculptor(root, main_app_ref=dummy_app) # No specific oscillator

    open_button_osc0 = ttk.Button(root, text="Open Sculptor (Osc 0)", command=open_sculptor_for_osc_0)
    open_button_osc0.pack(pady=10)

    open_button_new = ttk.Button(root, text="Open Sculptor (New)", command=open_new_sculptor)
    open_button_new.pack(pady=10)
    
    # Add a dummy populate_waveform_dropdown to root for testing callback if main_app_ref fails
    def dummy_populate():
        print("Main app's populate_waveform_dropdown would be called here.")
    root.populate_waveform_dropdown = dummy_populate

    root.mainloop() 