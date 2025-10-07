#!/usr/bin/env python3
"""
Human Positioning Widget

Pure GUI widget for human model positioning and configuration interface.
Handles only UI interactions and delegates all business logic to controller.

Author: Dimitrije Stojanovic
Date: December 2025
Created: Phase 3 Refactoring - GUI/Business Logic Separation
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

from .base_widget import BaseWidget, WidgetError


class HumanPositioningWidget(BaseWidget):
    """
    GUI widget for human model positioning and configuration.

    Provides interface for:
    - Human model positioning (X, Y, Z coordinates)
    - Model scaling and rotation
    - Model type selection (seated/standing)
    - Optimal position calculation
    - Model generation and manipulation
    - Statistics display and file operations
    """

    def __init__(self, parent: tk.Widget, controller: Optional[Any] = None,
                 widget_config: Optional[Dict[str, Any]] = None):
        """
        Initialize human positioning widget.

        Args:
            parent: Parent tkinter widget
            controller: Controller for handling human positioning business logic
            widget_config: Widget configuration options
        """
        # Required controller methods
        self._required_controller_methods = [
            'calculate_optimal_position',
            'generate_human_model',
            'update_model_position',
            'update_model_scale',
            'update_model_rotation',
            'get_model_statistics',
            'save_combined_model',
            'get_supported_model_types'
        ]

        # Initialize UI variables
        self.position_vars: Dict[str, tk.DoubleVar] = {}
        self.scale_var = tk.DoubleVar(value=1.0)
        self.rotation_var = tk.DoubleVar(value=0.0)
        self.model_type_var = tk.StringVar(value="seated")
        self.point_count_var = tk.StringVar(value="0 points")
        self.status_var = tk.StringVar(value="Ready")

        # UI components
        self.position_frame: Optional[tk.LabelFrame] = None
        self.parameters_frame: Optional[tk.LabelFrame] = None
        self.controls_frame: Optional[tk.LabelFrame] = None
        self.stats_frame: Optional[tk.LabelFrame] = None

        super().__init__(parent, controller, widget_config)

    def _create_widget(self) -> None:
        """Create the human positioning widget UI."""
        self.frame = tk.Frame(self.parent)

        # Create main sections
        self._create_position_section()
        self._create_parameters_section()
        self._create_controls_section()
        self._create_statistics_section()

        # Validate controller
        if not self.validate_controller(self._required_controller_methods):
            raise WidgetError("Controller missing required methods for HumanPositioningWidget")

        # Initialize with default values
        self._initialize_default_values()

    def _create_position_section(self) -> None:
        """Create the position configuration section."""
        self.position_frame = tk.LabelFrame(self.frame, text="Human Position", padx=10, pady=5)
        self.position_frame.pack(fill=tk.X, padx=5, pady=5)

        # Initialize position variables
        for coord in ['x', 'y', 'z']:
            self.position_vars[coord] = tk.DoubleVar(value=0.0)

        # Position controls
        pos_grid = tk.Frame(self.position_frame)
        pos_grid.pack(fill=tk.X)

        # X position
        tk.Label(pos_grid, text="X:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        x_entry = tk.Entry(pos_grid, textvariable=self.position_vars['x'], width=12)
        x_entry.grid(row=0, column=1, padx=(0, 10), sticky=tk.W)

        # Y position
        tk.Label(pos_grid, text="Y:").grid(row=0, column=2, sticky=tk.W, padx=(0, 5))
        y_entry = tk.Entry(pos_grid, textvariable=self.position_vars['y'], width=12)
        y_entry.grid(row=0, column=3, padx=(0, 10), sticky=tk.W)

        # Z position
        tk.Label(pos_grid, text="Z:").grid(row=0, column=4, sticky=tk.W, padx=(0, 5))
        z_entry = tk.Entry(pos_grid, textvariable=self.position_vars['z'], width=12)
        z_entry.grid(row=0, column=5, sticky=tk.W)

        # Auto-position button
        auto_button = tk.Button(self.position_frame, text="Calculate Optimal Position",
                               command=self._on_calculate_optimal_position)
        auto_button.pack(pady=(10, 0))

        # Bind position changes
        for var in self.position_vars.values():
            var.trace_add('write', self._on_position_changed)

    def _create_parameters_section(self) -> None:
        """Create the model parameters section."""
        self.parameters_frame = tk.LabelFrame(self.frame, text="Model Parameters", padx=10, pady=5)
        self.parameters_frame.pack(fill=tk.X, padx=5, pady=5)

        # Model type selection
        type_frame = tk.Frame(self.parameters_frame)
        type_frame.pack(fill=tk.X, pady=(0, 5))

        tk.Label(type_frame, text="Model Type:").pack(side=tk.LEFT)
        type_combo = ttk.Combobox(type_frame, textvariable=self.model_type_var,
                                 values=["seated", "standing"], state="readonly", width=10)
        type_combo.pack(side=tk.LEFT, padx=(10, 0))
        type_combo.bind('<<ComboboxSelected>>', self._on_model_type_changed)

        # Scale control
        scale_frame = tk.Frame(self.parameters_frame)
        scale_frame.pack(fill=tk.X, pady=2)

        tk.Label(scale_frame, text="Scale:").pack(side=tk.LEFT)
        scale_entry = tk.Entry(scale_frame, textvariable=self.scale_var, width=8)
        scale_entry.pack(side=tk.LEFT, padx=(10, 10))
        scale_scale = tk.Scale(scale_frame, from_=0.1, to=3.0, resolution=0.1,
                              variable=self.scale_var, orient=tk.HORIZONTAL, length=200)
        scale_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Rotation control
        rotation_frame = tk.Frame(self.parameters_frame)
        rotation_frame.pack(fill=tk.X, pady=2)

        tk.Label(rotation_frame, text="Rotation:").pack(side=tk.LEFT)
        rotation_entry = tk.Entry(rotation_frame, textvariable=self.rotation_var, width=8)
        rotation_entry.pack(side=tk.LEFT, padx=(10, 10))
        rotation_scale = tk.Scale(rotation_frame, from_=0, to=360, resolution=1,
                                 variable=self.rotation_var, orient=tk.HORIZONTAL, length=200)
        rotation_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Bind parameter changes
        self.scale_var.trace_add('write', self._on_parameters_changed)
        self.rotation_var.trace_add('write', self._on_parameters_changed)

    def _create_controls_section(self) -> None:
        """Create the controls section."""
        self.controls_frame = tk.LabelFrame(self.frame, text="Model Controls", padx=10, pady=5)
        self.controls_frame.pack(fill=tk.X, padx=5, pady=5)

        # Button frame
        button_frame = tk.Frame(self.controls_frame)
        button_frame.pack(fill=tk.X)

        # Control buttons
        tk.Button(button_frame, text="Generate Model",
                 command=self._on_generate_model,
                 bg='lightgreen').pack(side=tk.LEFT, padx=(0, 5))

        tk.Button(button_frame, text="Reset Parameters",
                 command=self._on_reset_parameters).pack(side=tk.LEFT, padx=(0, 5))

        tk.Button(button_frame, text="Save Combined Model",
                 command=self._on_save_model).pack(side=tk.LEFT, padx=(0, 5))

        # Model info display
        info_frame = tk.Frame(self.controls_frame)
        info_frame.pack(fill=tk.X, pady=(10, 0))

        tk.Label(info_frame, text="Model points:").pack(side=tk.LEFT)
        count_label = tk.Label(info_frame, textvariable=self.point_count_var,
                              font=('TkDefaultFont', 9, 'bold'))
        count_label.pack(side=tk.LEFT, padx=(10, 0))

        # Status display
        status_frame = tk.Frame(self.controls_frame)
        status_frame.pack(fill=tk.X, pady=(5, 0))

        tk.Label(status_frame, text="Status:").pack(side=tk.LEFT)
        status_label = tk.Label(status_frame, textvariable=self.status_var)
        status_label.pack(side=tk.LEFT, padx=(10, 0))

    def _create_statistics_section(self) -> None:
        """Create the model statistics section."""
        self.stats_frame = tk.LabelFrame(self.frame, text="Model Statistics", padx=10, pady=5)
        self.stats_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Statistics text widget
        self.stats_text = tk.Text(self.stats_frame, height=8, wrap=tk.WORD)
        scrollbar = tk.Scrollbar(self.stats_frame, orient=tk.VERTICAL, command=self.stats_text.yview)
        self.stats_text.config(yscrollcommand=scrollbar.set)

        self.stats_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.stats_text.insert(tk.END, "No human model generated yet.")
        self.stats_text.config(state=tk.DISABLED)

    def _initialize_default_values(self) -> None:
        """Initialize with default values."""
        try:
            # Initialize model type based on controller support
            supported_types = self.safe_call_controller('get_supported_model_types')
            if supported_types and isinstance(supported_types, list):
                # Update combobox values if we got different types from controller
                type_combo = None
                for child in self.parameters_frame.winfo_children():
                    for widget in child.winfo_children():
                        if isinstance(widget, ttk.Combobox):
                            widget['values'] = supported_types
                            break

            self.status_var.set("Ready to generate human model")

        except Exception as e:
            self.logger.error(f"Error initializing default values: {str(e)}")
            self.status_var.set("Initialization error")

    def _on_calculate_optimal_position(self) -> None:
        """Handle calculate optimal position."""
        try:
            optimal_pos = self.safe_call_controller('calculate_optimal_position')
            if optimal_pos and len(optimal_pos) == 3:
                self.set_position(optimal_pos)
                self.status_var.set("Optimal position calculated")
                self.trigger_callback('optimal_position_calculated', optimal_pos)
            else:
                self.status_var.set("Failed to calculate optimal position")
                messagebox.showwarning("Warning", "Could not calculate optimal position")

        except Exception as e:
            self.logger.error(f"Error calculating optimal position: {str(e)}")
            messagebox.showerror("Error", f"Position calculation error: {str(e)}")

    def _on_position_changed(self, *args) -> None:
        """Handle position changes."""
        self._update_model_if_exists()

    def _on_parameters_changed(self, *args) -> None:
        """Handle parameter changes."""
        self._update_model_if_exists()

    def _on_model_type_changed(self, *args) -> None:
        """Handle model type changes."""
        self._update_model_if_exists()

    def _update_model_if_exists(self) -> None:
        """Update model if one exists, otherwise just update status."""
        try:
            # Check if we have a model through controller
            stats = self.safe_call_controller('get_model_statistics')
            if stats and 'error' not in stats and stats.get('has_model', False):
                # Update the existing model
                position = self.get_position()
                if position:
                    # Update position
                    self.safe_call_controller('update_model_position', position)
                    # Update scale
                    self.safe_call_controller('update_model_scale', self.scale_var.get())
                    # Update rotation
                    self.safe_call_controller('update_model_rotation', self.rotation_var.get())

                    self._update_statistics()
                    self.status_var.set("Model updated")

        except Exception as e:
            self.logger.error(f"Error updating model: {str(e)}")

    def _on_generate_model(self) -> None:
        """Handle generate model."""
        try:
            position = self.get_position()
            if not position:
                messagebox.showerror("Error", "Invalid position values")
                return

            # Generate model via controller
            model_result = self.safe_call_controller(
                'generate_human_model',
                position=position,
                scale=self.scale_var.get(),
                rotation=self.rotation_var.get(),
                model_type=self.model_type_var.get()
            )

            if model_result and hasattr(model_result, 'has_model') and model_result.has_model:
                self.point_count_var.set(f"{model_result.point_count} points")
                self.status_var.set("Human model generated")
                self._update_statistics()
                self.trigger_callback('model_generated', model_result)
            else:
                self.status_var.set("Model generation failed")
                messagebox.showerror("Error", "Failed to generate human model")

        except Exception as e:
            self.logger.error(f"Error generating model: {str(e)}")
            messagebox.showerror("Error", f"Model generation error: {str(e)}")

    def _on_reset_parameters(self) -> None:
        """Handle reset parameters."""
        self.position_vars['x'].set(0.0)
        self.position_vars['y'].set(0.0)
        self.position_vars['z'].set(0.0)
        self.scale_var.set(1.0)
        self.rotation_var.set(0.0)
        self.model_type_var.set("seated")
        self.status_var.set("Parameters reset")
        self.trigger_callback('parameters_reset')

    def _on_save_model(self) -> None:
        """Handle save combined model."""
        try:
            # Get save file path
            file_path = filedialog.asksaveasfilename(
                title="Save Combined Model",
                defaultextension=".npy",
                filetypes=[
                    ("NumPy arrays", "*.npy"),
                    ("All files", "*.*")
                ]
            )

            if not file_path:
                return

            # Save via controller
            success = self.safe_call_controller('save_combined_model', Path(file_path), True)
            if success:
                self.status_var.set(f"Model saved to {Path(file_path).name}")
                messagebox.showinfo("Success", f"Combined model saved to:\n{file_path}")
                self.trigger_callback('model_saved', file_path)
            else:
                self.status_var.set("Save failed")
                messagebox.showerror("Error", "Failed to save combined model")

        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            messagebox.showerror("Error", f"Save error: {str(e)}")

    def _update_statistics(self) -> None:
        """Update the statistics display."""
        try:
            stats = self.safe_call_controller('get_model_statistics')
            if stats and 'error' not in stats:
                self.stats_text.config(state=tk.NORMAL)
                self.stats_text.delete(1.0, tk.END)

                # Format statistics
                stats_text = "Human Model Statistics:\n\n"

                if stats.get('has_model', False):
                    stats_text += f"Model type: {stats.get('model_type', 'N/A')}\n"
                    stats_text += f"Scale: {stats.get('scale', 'N/A')}\n"
                    stats_text += f"Rotation: {stats.get('rotation', 'N/A')}°\n"
                    stats_text += f"Point count: {stats.get('point_count', 'N/A')}\n"

                    if 'position' in stats and stats['position']:
                        pos = stats['position']
                        stats_text += f"Position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})\n"

                    if 'bounds' in stats:
                        bounds = stats['bounds']
                        stats_text += f"\nModel bounds:\n"
                        for axis, bound in bounds.items():
                            stats_text += f"{axis.upper()}: [{bound[0]:.2f}, {bound[1]:.2f}]\n"

                    if 'centroid' in stats:
                        centroid = stats['centroid']
                        stats_text += f"\nCentroid: ({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f})\n"

                    if 'volume' in stats:
                        stats_text += f"Model volume: {stats['volume']:.3f}\n"
                else:
                    stats_text += "No human model generated.\n"

                self.stats_text.insert(tk.END, stats_text)
                self.stats_text.config(state=tk.DISABLED)

            else:
                self.stats_text.config(state=tk.NORMAL)
                self.stats_text.delete(1.0, tk.END)
                error_msg = stats.get('error', 'No statistics available') if stats else 'No statistics available'
                self.stats_text.insert(tk.END, f"Statistics Error: {error_msg}")
                self.stats_text.config(state=tk.DISABLED)

        except Exception as e:
            self.logger.error(f"Error updating statistics: {str(e)}")
            self.stats_text.config(state=tk.NORMAL)
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(tk.END, f"Error loading statistics: {str(e)}")
            self.stats_text.config(state=tk.DISABLED)

    def get_position(self) -> Optional[Tuple[float, float, float]]:
        """
        Get current position from UI.

        Returns:
            Tuple of (x, y, z) or None if invalid
        """
        try:
            position = (
                self.position_vars['x'].get(),
                self.position_vars['y'].get(),
                self.position_vars['z'].get()
            )
            return position
        except tk.TclError as e:
            self.logger.error(f"Error getting position: {str(e)}")
            return None

    def set_position(self, position: Tuple[float, float, float]) -> None:
        """
        Set position in UI.

        Args:
            position: Tuple of (x, y, z)
        """
        try:
            x, y, z = position

            self.position_vars['x'].set(x)
            self.position_vars['y'].set(y)
            self.position_vars['z'].set(z)

            self.logger.debug(f"Set position: {position}")

        except Exception as e:
            self.logger.error(f"Error setting position: {str(e)}")
            raise WidgetError(f"Cannot set position: {str(e)}")

    def get_model_parameters(self) -> Dict[str, Any]:
        """Get current model parameters."""
        return {
            'position': self.get_position(),
            'scale': self.scale_var.get(),
            'rotation': self.rotation_var.get(),
            'model_type': self.model_type_var.get()
        }

    def set_model_parameters(self, parameters: Dict[str, Any]) -> None:
        """Set model parameters."""
        if 'position' in parameters and parameters['position']:
            self.set_position(parameters['position'])
        if 'scale' in parameters:
            self.scale_var.set(parameters['scale'])
        if 'rotation' in parameters:
            self.rotation_var.set(parameters['rotation'])
        if 'model_type' in parameters:
            self.model_type_var.set(parameters['model_type'])

    def get_widget_info(self) -> Dict[str, Any]:
        """Get human positioning widget information."""
        base_info = super().get_widget_info()
        base_info.update({
            'widget_type': 'HumanPositioningWidget',
            'current_position': self.get_position(),
            'model_parameters': self.get_model_parameters(),
            'point_count': self.point_count_var.get(),
            'status': self.status_var.get(),
        })
        return base_info