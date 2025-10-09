#!/usr/bin/env python3
"""
Cube Selection Widget

Pure GUI widget for cube-based point cloud selection interface.
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


class CubeSelectionWidget(BaseWidget):
    """
    GUI widget for cube-based point cloud selection.

    Provides interface for:
    - Cube bounds configuration (min/max X, Y, Z)
    - Real-time point count preview
    - Selection operations (apply, reset, save)
    - Cube bounds validation and feedback
    - Selection statistics display
    """

    def __init__(self, parent: tk.Widget, controller: Optional[Any] = None,
                 widget_config: Optional[Dict[str, Any]] = None):
        """
        Initialize cube selection widget.

        Args:
            parent: Parent tkinter widget
            controller: Controller for handling cube selection business logic
            widget_config: Widget configuration options
        """
        # Required controller methods
        self._required_controller_methods = [
            'get_point_cloud_bounds',
            'calculate_default_cube_bounds',
            'validate_cube_bounds',
            'count_points_in_cube',
            'select_points_in_cube',
            'get_selection_statistics',
            'save_selection'
        ]

        # Initialize UI variables
        self.cube_vars: Dict[str, tk.DoubleVar] = {}
        self.point_count_var = tk.StringVar(value="0 points")
        self.status_var = tk.StringVar(value="Ready")

        # UI components
        self.bounds_frame: Optional[tk.LabelFrame] = None
        self.controls_frame: Optional[tk.LabelFrame] = None
        self.stats_frame: Optional[tk.LabelFrame] = None

        super().__init__(parent, controller, widget_config)

    def _create_widget(self) -> None:
        """Create the cube selection widget UI."""
        self.frame = tk.Frame(self.parent)

        # Create main sections
        self._create_bounds_section()
        self._create_controls_section()
        self._create_statistics_section()

        # Validate controller
        if not self.validate_controller(self._required_controller_methods):
            raise WidgetError("Controller missing required methods for CubeSelectionWidget")

        # Initialize with default values
        self._initialize_default_bounds()

    def _create_bounds_section(self) -> None:
        """Create the cube bounds configuration section."""
        self.bounds_frame = tk.LabelFrame(self.frame, text="Cube Bounds", padx=10, pady=5)
        self.bounds_frame.pack(fill=tk.X, padx=5, pady=5)

        # Initialize variables for cube bounds
        bound_names = ['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax']
        for name in bound_names:
            self.cube_vars[name] = tk.DoubleVar()

        # Create entry widgets in a grid
        row = 0

        # X bounds
        x_frame = tk.Frame(self.bounds_frame)
        x_frame.pack(fill=tk.X, pady=2)

        tk.Label(x_frame, text="X:").pack(side=tk.LEFT)
        tk.Label(x_frame, text="Min").pack(side=tk.LEFT, padx=(10, 5))
        x_min_entry = tk.Entry(x_frame, textvariable=self.cube_vars['xmin'], width=10)
        x_min_entry.pack(side=tk.LEFT, padx=(0, 10))
        tk.Label(x_frame, text="Max").pack(side=tk.LEFT, padx=(0, 5))
        x_max_entry = tk.Entry(x_frame, textvariable=self.cube_vars['xmax'], width=10)
        x_max_entry.pack(side=tk.LEFT)

        # Y bounds
        y_frame = tk.Frame(self.bounds_frame)
        y_frame.pack(fill=tk.X, pady=2)

        tk.Label(y_frame, text="Y:").pack(side=tk.LEFT)
        tk.Label(y_frame, text="Min").pack(side=tk.LEFT, padx=(10, 5))
        y_min_entry = tk.Entry(y_frame, textvariable=self.cube_vars['ymin'], width=10)
        y_min_entry.pack(side=tk.LEFT, padx=(0, 10))
        tk.Label(y_frame, text="Max").pack(side=tk.LEFT, padx=(0, 5))
        y_max_entry = tk.Entry(y_frame, textvariable=self.cube_vars['ymax'], width=10)
        y_max_entry.pack(side=tk.LEFT)

        # Z bounds
        z_frame = tk.Frame(self.bounds_frame)
        z_frame.pack(fill=tk.X, pady=2)

        tk.Label(z_frame, text="Z:").pack(side=tk.LEFT)
        tk.Label(z_frame, text="Min").pack(side=tk.LEFT, padx=(10, 5))
        z_min_entry = tk.Entry(z_frame, textvariable=self.cube_vars['zmin'], width=10)
        z_min_entry.pack(side=tk.LEFT, padx=(0, 10))
        tk.Label(z_frame, text="Max").pack(side=tk.LEFT, padx=(0, 5))
        z_max_entry = tk.Entry(z_frame, textvariable=self.cube_vars['zmax'], width=10)
        z_max_entry.pack(side=tk.LEFT)

        # Bind events for real-time updates
        for var in self.cube_vars.values():
            var.trace_add('write', self._on_bounds_changed)

    def _create_controls_section(self) -> None:
        """Create the controls section."""
        self.controls_frame = tk.LabelFrame(self.frame, text="Controls", padx=10, pady=5)
        self.controls_frame.pack(fill=tk.X, padx=5, pady=5)

        # Button frame
        button_frame = tk.Frame(self.controls_frame)
        button_frame.pack(fill=tk.X)

        # Control buttons
        tk.Button(button_frame, text="Reset to Default",
                 command=self._on_reset_bounds).pack(side=tk.LEFT, padx=(0, 5))

        tk.Button(button_frame, text="Apply Selection",
                 command=self._on_apply_selection,
                 bg='lightgreen').pack(side=tk.LEFT, padx=(0, 5))

        tk.Button(button_frame, text="Save Selection",
                 command=self._on_save_selection).pack(side=tk.LEFT, padx=(0, 5))

        # Point count display
        count_frame = tk.Frame(self.controls_frame)
        count_frame.pack(fill=tk.X, pady=(10, 0))

        tk.Label(count_frame, text="Points in cube:").pack(side=tk.LEFT)
        count_label = tk.Label(count_frame, textvariable=self.point_count_var,
                              font=('TkDefaultFont', 9, 'bold'))
        count_label.pack(side=tk.LEFT, padx=(10, 0))

        # Status display
        status_frame = tk.Frame(self.controls_frame)
        status_frame.pack(fill=tk.X, pady=(5, 0))

        tk.Label(status_frame, text="Status:").pack(side=tk.LEFT)
        status_label = tk.Label(status_frame, textvariable=self.status_var)
        status_label.pack(side=tk.LEFT, padx=(10, 0))

    def _create_statistics_section(self) -> None:
        """Create the statistics section."""
        self.stats_frame = tk.LabelFrame(self.frame, text="Selection Statistics", padx=10, pady=5)
        self.stats_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Statistics text widget
        self.stats_text = tk.Text(self.stats_frame, height=6, wrap=tk.WORD)
        scrollbar = tk.Scrollbar(self.stats_frame, orient=tk.VERTICAL, command=self.stats_text.yview)
        self.stats_text.config(yscrollcommand=scrollbar.set)

        self.stats_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.stats_text.insert(tk.END, "No selection statistics available.")
        self.stats_text.config(state=tk.DISABLED)

    def _initialize_default_bounds(self) -> None:
        """Initialize cube bounds with default values from controller."""
        try:
            default_bounds = self.safe_call_controller('calculate_default_cube_bounds')
            if default_bounds:
                self.set_cube_bounds(default_bounds)
                self.status_var.set("Initialized with default bounds")
            else:
                # Fallback values
                self.set_cube_bounds((0.0, 1.0, 0.0, 1.0, 0.0, 1.0))
                self.status_var.set("Using fallback bounds")
        except Exception as e:
            self.logger.error(f"Error initializing default bounds: {str(e)}")
            self.status_var.set("Error initializing bounds")

    def _on_bounds_changed(self, *args) -> None:
        """Handle cube bounds changes (real-time updates)."""
        try:
            cube_bounds = self.get_cube_bounds()
            if cube_bounds:
                # Validate bounds
                validation_result = self.safe_call_controller('validate_cube_bounds', cube_bounds)
                if validation_result and validation_result.get('valid', False):
                    # Update point count
                    point_count = self.safe_call_controller('count_points_in_cube', cube_bounds)
                    if point_count is not None:
                        self.point_count_var.set(f"{point_count} points")
                        self.status_var.set("Valid cube bounds")
                    else:
                        self.point_count_var.set("? points")
                        self.status_var.set("Cannot count points")
                else:
                    self.point_count_var.set("Invalid")
                    error_msg = "Invalid bounds"
                    if validation_result and 'errors' in validation_result:
                        error_msg = "; ".join(validation_result['errors'][:2])  # Show first 2 errors
                    self.status_var.set(error_msg)
        except Exception as e:
            self.logger.error(f"Error in bounds change handler: {str(e)}")
            self.status_var.set("Error updating bounds")

    def _on_reset_bounds(self) -> None:
        """Handle reset to default bounds."""
        self._initialize_default_bounds()
        self.trigger_callback('bounds_reset')

    def _on_apply_selection(self) -> None:
        """Handle apply selection."""
        try:
            cube_bounds = self.get_cube_bounds()
            if not cube_bounds:
                messagebox.showerror("Error", "Invalid cube bounds")
                return

            # Apply selection via controller
            selection_result = self.safe_call_controller('select_points_in_cube', cube_bounds)
            if selection_result:
                self.status_var.set("Selection applied")
                self._update_statistics()
                self.trigger_callback('selection_applied', selection_result)
            else:
                self.status_var.set("Selection failed")
                messagebox.showerror("Error", "Failed to apply selection")

        except Exception as e:
            self.logger.error(f"Error applying selection: {str(e)}")
            messagebox.showerror("Error", f"Selection error: {str(e)}")

    def _on_save_selection(self) -> None:
        """Handle save selection."""
        try:
            # Get save file path
            file_path = filedialog.asksaveasfilename(
                title="Save Selection",
                defaultextension=".npy",
                filetypes=[
                    ("NumPy arrays", "*.npy"),
                    ("All files", "*.*")
                ]
            )

            if not file_path:
                return

            # Save via controller
            success = self.safe_call_controller('save_selection', Path(file_path))
            if success:
                self.status_var.set(f"Selection saved to {Path(file_path).name}")
                messagebox.showinfo("Success", f"Selection saved to:\n{file_path}")
                self.trigger_callback('selection_saved', file_path)
            else:
                self.status_var.set("Save failed")
                messagebox.showerror("Error", "Failed to save selection")

        except Exception as e:
            self.logger.error(f"Error saving selection: {str(e)}")
            messagebox.showerror("Error", f"Save error: {str(e)}")

    def _update_statistics(self) -> None:
        """Update the statistics display."""
        try:
            stats = self.safe_call_controller('get_selection_statistics')
            if stats and 'error' not in stats:
                self.stats_text.config(state=tk.NORMAL)
                self.stats_text.delete(1.0, tk.END)

                # Format statistics
                stats_text = "Selection Statistics:\n\n"
                stats_text += f"Total points: {stats.get('total_points', 'N/A')}\n"
                stats_text += f"Selected points: {stats.get('selected_points', 'N/A')}\n"
                stats_text += f"Selection percentage: {stats.get('selection_percentage', 0):.1f}%\n"

                if 'centroid' in stats:
                    centroid = stats['centroid']
                    stats_text += f"Centroid: ({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f})\n"

                if 'point_density' in stats:
                    stats_text += f"Point density: {stats['point_density']:.2f} points/unit³\n"

                if 'bounds' in stats:
                    bounds = stats['bounds']
                    stats_text += f"\nActual bounds:\n"
                    stats_text += f"X: [{bounds['x'][0]:.2f}, {bounds['x'][1]:.2f}]\n"
                    stats_text += f"Y: [{bounds['y'][0]:.2f}, {bounds['y'][1]:.2f}]\n"
                    stats_text += f"Z: [{bounds['z'][0]:.2f}, {bounds['z'][1]:.2f}]\n"

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

    def get_cube_bounds(self) -> Optional[Tuple[float, float, float, float, float, float]]:
        """
        Get current cube bounds from UI.

        Returns:
            Tuple of (xmin, xmax, ymin, ymax, zmin, zmax) or None if invalid
        """
        try:
            bounds = (
                self.cube_vars['xmin'].get(),
                self.cube_vars['xmax'].get(),
                self.cube_vars['ymin'].get(),
                self.cube_vars['ymax'].get(),
                self.cube_vars['zmin'].get(),
                self.cube_vars['zmax'].get()
            )
            return bounds
        except tk.TclError as e:
            self.logger.error(f"Error getting cube bounds: {str(e)}")
            return None

    def set_cube_bounds(self, bounds: Tuple[float, float, float, float, float, float]) -> None:
        """
        Set cube bounds in UI.

        Args:
            bounds: Tuple of (xmin, xmax, ymin, ymax, zmin, zmax)
        """
        try:
            xmin, xmax, ymin, ymax, zmin, zmax = bounds

            self.cube_vars['xmin'].set(xmin)
            self.cube_vars['xmax'].set(xmax)
            self.cube_vars['ymin'].set(ymin)
            self.cube_vars['ymax'].set(ymax)
            self.cube_vars['zmin'].set(zmin)
            self.cube_vars['zmax'].set(zmax)

            self.logger.debug(f"Set cube bounds: {bounds}")

        except Exception as e:
            self.logger.error(f"Error setting cube bounds: {str(e)}")
            raise WidgetError(f"Cannot set cube bounds: {str(e)}")

    def get_widget_info(self) -> Dict[str, Any]:
        """Get cube selection widget information."""
        base_info = super().get_widget_info()
        base_info.update({
            'widget_type': 'CubeSelectionWidget',
            'current_bounds': self.get_cube_bounds(),
            'point_count': self.point_count_var.get(),
            'status': self.status_var.get(),
        })
        return base_info