#!/usr/bin/env python3
"""
LIDAR GUI Application

A comprehensive GUI interface for LIDAR point cloud processing and visualization.
This application integrates all the point cloud analysis tools with a user-friendly interface.

Features:
- File loading for E57, PCD, and ROS bag formats
- Vehicle analysis with interior extraction
- Human model positioning
- Results visualization
- macOS-optimized VTK handling

Author: Dimitrije Stojanovic
Date: September 2025
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
import sys
import gc
import numpy as np
from pathlib import Path
import queue

# Add the point_cloud module to path
sys.path.append(str(Path(__file__).parent))

try:
    import pyvista as pv
    VTK_AVAILABLE = True
except ImportError:
    VTK_AVAILABLE = False
    print("Warning: PyVista/VTK not available. 3D visualization disabled.")

try:
    from point_cloud.e57_vehicle_analysis import E57VehicleAnalyzer
    from point_cloud.visualise_raw_lidar_files import read_points
    from point_cloud.visualize_output import OutputVisualizer
    from point_cloud.interactive_human_positioner import InteractiveHumanPositioner
    from point_cloud.interactive_cube_selector import CubeSelectionManager
    from point_cloud.vtk_utils import VTKSafetyManager
    from point_cloud.error_handling import (ErrorHandler, with_error_handling, 
                                           validate_file_path)
    from point_cloud.config import FileConfig, GUIConfig
    MODULES_AVAILABLE = True
except ImportError as e:
    MODULES_AVAILABLE = False
    print(f"Warning: Point cloud modules not available: {e}")


# VTKEnvironmentManager is now provided by vtk_utils module - keeping for backward compatibility
VTKEnvironmentManager = VTKSafetyManager


class WorkerThread(threading.Thread):
    """Worker thread for long-running operations."""
    
    def __init__(self, target_func, args=(), kwargs=None, callback=None, error_callback=None):
        super().__init__()
        self.target_func = target_func
        self.args = args
        self.kwargs = kwargs or {}
        self.callback = callback
        self.error_callback = error_callback
        self.daemon = True
        
    def run(self):
        try:
            result = self.target_func(*self.args, **self.kwargs)
            if self.callback:
                self.callback(result)
        except Exception as e:
            if self.error_callback:
                self.error_callback(e)
            else:
                print(f"Worker thread error: {e}")


class LidarGUIApp:
    """Main LIDAR GUI Application."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("LIDAR Point Cloud Analysis Suite")
        self.root.geometry(GUIConfig.MAIN_WINDOW_SIZE)
        
        # Set up VTK environment for cross-platform compatibility
        VTKSafetyManager.setup_vtk_environment()
        
        # Initialize error handling
        self.error_handler = ErrorHandler("lidar_gui")
        
        # Application state
        self.current_file = None
        self.active_plotters = []
        self.analysis_results = {}
        self.message_queue = queue.Queue()
        self.multi_file_list = []  # List of selected files for multi-visualization
        
        # Create GUI
        self.setup_gui()
        
        # Set up cleanup
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Check for required modules
        if not MODULES_AVAILABLE:
            self.log_message("Warning: Some point cloud modules not available. Limited functionality.")
        if not VTK_AVAILABLE:
            self.log_message("Warning: VTK not available. 3D visualization disabled.")
    
    def setup_gui(self):
        """Set up the main GUI interface."""
        # Create main notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs
        self.create_file_tab()
        self.create_analysis_tab()
        self.create_visualization_tab()
        self.create_multi_file_visualization_tab()
        self.create_cube_selection_tab()
        self.create_human_model_tab()
        self.create_log_tab()
        
        # Status bar
        self.status_bar = ttk.Label(self.root, text="Ready", relief=tk.SUNKEN)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Process message queue periodically
        self.root.after(100, self.process_message_queue)
    
    def create_file_tab(self):
        """Create the file operations tab."""
        file_frame = ttk.Frame(self.notebook)
        self.notebook.add(file_frame, text="File Operations")
        
        # File selection section
        file_section = ttk.LabelFrame(file_frame, text="File Selection", padding="10")
        file_section.pack(fill=tk.X, padx=10, pady=5)
        
        self.file_path_var = tk.StringVar()
        file_entry = ttk.Entry(file_section, textvariable=self.file_path_var, width=70)
        file_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        browse_btn = ttk.Button(file_section, text="Browse...", command=self.browse_file)
        browse_btn.pack(side=tk.RIGHT, padx=(10, 0))
        
        # File info section
        info_section = ttk.LabelFrame(file_frame, text="File Information", padding="10")
        info_section.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.info_text = scrolledtext.ScrolledText(info_section, height=15, state=tk.DISABLED)
        self.info_text.pack(fill=tk.BOTH, expand=True)
        
        # Load file button
        load_btn = ttk.Button(file_frame, text="Load File", command=self.load_file)
        load_btn.pack(pady=10)
    
    def create_analysis_tab(self):
        """Create the analysis tab."""
        analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(analysis_frame, text="Analysis")
        
        # Analysis options
        options_section = ttk.LabelFrame(analysis_frame, text="Analysis Options", padding="10")
        options_section.pack(fill=tk.X, padx=10, pady=5)
        
        # Analysis type
        ttk.Label(options_section, text="Analysis Type:").pack(anchor=tk.W)
        self.analysis_type_var = tk.StringVar(value="vehicle")
        analysis_frame_inner = ttk.Frame(options_section)
        analysis_frame_inner.pack(fill=tk.X, pady=5)
        
        ttk.Radiobutton(analysis_frame_inner, text="Vehicle Analysis", 
                       variable=self.analysis_type_var, value="vehicle").pack(side=tk.LEFT)
        ttk.Radiobutton(analysis_frame_inner, text="Raw Visualization", 
                       variable=self.analysis_type_var, value="raw").pack(side=tk.LEFT, padx=(20, 0))
        
        # Analysis parameters
        params_frame = ttk.Frame(options_section)
        params_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(params_frame, text="Visualization Mode:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.viz_mode_var = tk.StringVar(value="interactive")
        viz_combo = ttk.Combobox(params_frame, textvariable=self.viz_mode_var,
                                values=["interactive", "combined", "detailed", "focused", "all"],
                                state="readonly", width=15)
        viz_combo.grid(row=0, column=1, sticky=tk.W)
        
        self.enable_cube_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_section, text="Enable Interactive Cube Selection",
                       variable=self.enable_cube_var).pack(anchor=tk.W, pady=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(analysis_frame, variable=self.progress_var,
                                          maximum=100, length=400)
        self.progress_bar.pack(pady=10)
        
        # Start analysis button
        self.analysis_btn = ttk.Button(analysis_frame, text="Start Analysis",
                                     command=self.start_analysis, state=tk.DISABLED)
        self.analysis_btn.pack(pady=10)
        
        # Results section
        results_section = ttk.LabelFrame(analysis_frame, text="Analysis Results", padding="10")
        results_section.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.results_text = scrolledtext.ScrolledText(results_section, height=10, state=tk.DISABLED)
        self.results_text.pack(fill=tk.BOTH, expand=True)
    
    def create_visualization_tab(self):
        """Create the visualization tab."""
        viz_frame = ttk.Frame(self.notebook)
        self.notebook.add(viz_frame, text="Visualization")
        
        # Visualization options
        viz_options = ttk.LabelFrame(viz_frame, text="Visualization Options", padding="10")
        viz_options.pack(fill=tk.X, padx=10, pady=5)
        
        # First row: Point size and Color mode
        row1_frame = ttk.Frame(viz_options)
        row1_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(row1_frame, text="Point Size:").pack(side=tk.LEFT)
        self.point_size_var = tk.DoubleVar(value=2.0)
        size_spin = ttk.Spinbox(row1_frame, from_=0.1, to=10.0, increment=0.1,
                               textvariable=self.point_size_var, width=10)
        size_spin.pack(side=tk.LEFT, padx=(5, 20))
        
        ttk.Label(row1_frame, text="Color By:").pack(side=tk.LEFT)
        self.color_mode_var = tk.StringVar(value="intensity")
        color_combo = ttk.Combobox(row1_frame, textvariable=self.color_mode_var,
                                  values=["intensity", "elevation", "rgb", "ring"],
                                  state="readonly", width=15)
        color_combo.pack(side=tk.LEFT, padx=(5, 0))
        
        # Second row: Stride and render options
        row2_frame = ttk.Frame(viz_options)
        row2_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(row2_frame, text="Stride (every Nth point):").pack(side=tk.LEFT)
        self.stride_var = tk.IntVar(value=1)
        stride_spin = ttk.Spinbox(row2_frame, from_=1, to=100, increment=1,
                                 textvariable=self.stride_var, width=10)
        stride_spin.pack(side=tk.LEFT, padx=(5, 20))
        
        self.spheres_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row2_frame, text="Render as Spheres",
                       variable=self.spheres_var).pack(side=tk.LEFT)
        
        # Third row: E57 scan selection
        row3_frame = ttk.Frame(viz_options)
        row3_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(row3_frame, text="E57 Scans (e.g. 0,1,2 or leave empty for all):").pack(side=tk.LEFT)
        self.scan_indices_var = tk.StringVar()
        scan_entry = ttk.Entry(row3_frame, textvariable=self.scan_indices_var, width=20)
        scan_entry.pack(side=tk.LEFT, padx=(5, 0))
        
        # Fourth row: ROS bag options
        row4_frame = ttk.Frame(viz_options)
        row4_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(row4_frame, text="ROS Topic:").pack(side=tk.LEFT)
        self.ros_topic_var = tk.StringVar()
        topic_entry = ttk.Entry(row4_frame, textvariable=self.ros_topic_var, width=20)
        topic_entry.pack(side=tk.LEFT, padx=(5, 20))
        
        ttk.Label(row4_frame, text="Frame Index:").pack(side=tk.LEFT)
        self.frame_index_var = tk.IntVar(value=0)
        frame_spin = ttk.Spinbox(row4_frame, from_=0, to=1000, increment=1,
                                textvariable=self.frame_index_var, width=10)
        frame_spin.pack(side=tk.LEFT, padx=(5, 0))
        
        # Visualization buttons
        buttons_frame = ttk.Frame(viz_frame)
        buttons_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(buttons_frame, text="View Raw File",
                  command=self.visualize_raw_file).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="View Raw Analysis",
                  command=self.visualize_raw).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="View Analysis Results",
                  command=self.visualize_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="View Output Files",
                  command=self.visualize_output).pack(side=tk.LEFT, padx=5)
        
        # Visualization info
        viz_info = ttk.LabelFrame(viz_frame, text="Visualization Info", padding="10")
        viz_info.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.viz_info_text = scrolledtext.ScrolledText(viz_info, height=8, state=tk.DISABLED)
        self.viz_info_text.pack(fill=tk.BOTH, expand=True)
        
        # Add helpful text
        self.viz_info_text.config(state=tk.NORMAL)
        help_text = """Visualization Help:

• View Raw File: Directly visualize the selected file with full options
• View Raw Analysis: Show analysis results (requires running analysis first)
• View Analysis Results: Show segmented analysis results
• View Output Files: Load and display saved .npy files

Options:
• Stride: Skip every Nth point (higher = faster, lower quality)
• E57 Scans: Comma-separated scan indices (e.g. 0,1,2)
• ROS Topic: Specific topic name (auto-detect if empty)
• Color modes: intensity, elevation, rgb, ring (Velodyne)"""
        self.viz_info_text.insert(tk.END, help_text)
        self.viz_info_text.config(state=tk.DISABLED)

    def create_multi_file_visualization_tab(self):
        """Create the multi-file visualization tab."""
        multi_viz_frame = ttk.Frame(self.notebook)
        self.notebook.add(multi_viz_frame, text="Multi-File Visualization")

        # File selection section
        file_select_section = ttk.LabelFrame(multi_viz_frame, text="File Selection", padding="10")
        file_select_section.pack(fill=tk.X, padx=10, pady=5)

        # Buttons for file selection
        button_frame = ttk.Frame(file_select_section)
        button_frame.pack(fill=tk.X, pady=5)

        ttk.Button(button_frame, text="Select Multiple Files",
                  command=self.select_multiple_files).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Add Single File",
                  command=self.add_single_file).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear Selection",
                  command=self.clear_file_selection).pack(side=tk.LEFT, padx=5)

        # Selected files list
        files_frame = ttk.LabelFrame(multi_viz_frame, text="Selected Files", padding="10")
        files_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Create listbox with scrollbar for selected files
        listbox_frame = ttk.Frame(files_frame)
        listbox_frame.pack(fill=tk.BOTH, expand=True)

        self.multi_files_listbox = tk.Listbox(listbox_frame, height=10)
        scrollbar_files = ttk.Scrollbar(listbox_frame, orient=tk.VERTICAL, command=self.multi_files_listbox.yview)
        self.multi_files_listbox.config(yscrollcommand=scrollbar_files.set)

        self.multi_files_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_files.pack(side=tk.RIGHT, fill=tk.Y)

        # File management buttons
        file_mgmt_frame = ttk.Frame(files_frame)
        file_mgmt_frame.pack(fill=tk.X, pady=(10, 0))

        ttk.Button(file_mgmt_frame, text="Remove Selected",
                  command=self.remove_selected_file).pack(side=tk.LEFT, padx=5)
        ttk.Button(file_mgmt_frame, text="View File Info",
                  command=self.view_file_info).pack(side=tk.LEFT, padx=5)

        # Visualization options
        viz_options_section = ttk.LabelFrame(multi_viz_frame, text="Visualization Options", padding="10")
        viz_options_section.pack(fill=tk.X, padx=10, pady=5)

        # Options row 1: Point size and Color mode
        options_row1 = ttk.Frame(viz_options_section)
        options_row1.pack(fill=tk.X, pady=5)

        ttk.Label(options_row1, text="Point Size:").pack(side=tk.LEFT)
        self.multi_point_size_var = tk.DoubleVar(value=2.0)
        size_spin = ttk.Spinbox(options_row1, from_=0.1, to=10.0, increment=0.1,
                               textvariable=self.multi_point_size_var, width=10)
        size_spin.pack(side=tk.LEFT, padx=(5, 20))

        ttk.Label(options_row1, text="Color By:").pack(side=tk.LEFT)
        self.multi_color_mode_var = tk.StringVar(value="intensity")
        color_combo = ttk.Combobox(options_row1, textvariable=self.multi_color_mode_var,
                                  values=["intensity", "elevation", "rgb", "ring"],
                                  state="readonly", width=15)
        color_combo.pack(side=tk.LEFT, padx=(5, 0))

        # Options row 2: Stride and window management
        options_row2 = ttk.Frame(viz_options_section)
        options_row2.pack(fill=tk.X, pady=5)

        ttk.Label(options_row2, text="Stride (every Nth point):").pack(side=tk.LEFT)
        self.multi_stride_var = tk.IntVar(value=1)
        stride_spin = ttk.Spinbox(options_row2, from_=1, to=100, increment=1,
                                 textvariable=self.multi_stride_var, width=10)
        stride_spin.pack(side=tk.LEFT, padx=(5, 20))

        self.multi_spheres_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_row2, text="Render as Spheres",
                       variable=self.multi_spheres_var).pack(side=tk.LEFT)

        # Window arrangement options
        window_options_section = ttk.LabelFrame(multi_viz_frame, text="Window Arrangement", padding="10")
        window_options_section.pack(fill=tk.X, padx=10, pady=5)

        self.window_arrangement_var = tk.StringVar(value="cascade")
        arrangement_frame = ttk.Frame(window_options_section)
        arrangement_frame.pack(fill=tk.X, pady=5)

        ttk.Radiobutton(arrangement_frame, text="Cascade Windows",
                       variable=self.window_arrangement_var, value="cascade").pack(side=tk.LEFT)
        ttk.Radiobutton(arrangement_frame, text="Grid Layout",
                       variable=self.window_arrangement_var, value="grid").pack(side=tk.LEFT, padx=(20, 0))
        ttk.Radiobutton(arrangement_frame, text="Custom Positions",
                       variable=self.window_arrangement_var, value="custom").pack(side=tk.LEFT, padx=(20, 0))

        # Visualization control buttons
        viz_control_frame = ttk.Frame(multi_viz_frame)
        viz_control_frame.pack(fill=tk.X, padx=10, pady=10)

        self.visualize_all_btn = ttk.Button(viz_control_frame, text="Visualize All Files",
                                          command=self.visualize_all_files, state=tk.DISABLED)
        self.visualize_all_btn.pack(side=tk.LEFT, padx=5)

        ttk.Button(viz_control_frame, text="Visualize Selected",
                  command=self.visualize_selected_file).pack(side=tk.LEFT, padx=5)

        ttk.Button(viz_control_frame, text="Close All Windows",
                  command=self.close_all_multi_windows).pack(side=tk.RIGHT, padx=5)

        # Status and info
        status_section = ttk.LabelFrame(multi_viz_frame, text="Status", padding="10")
        status_section.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.multi_viz_status_text = scrolledtext.ScrolledText(status_section, height=6, state=tk.DISABLED)
        self.multi_viz_status_text.pack(fill=tk.BOTH, expand=True)

        # Add initial help text
        self.multi_viz_status_text.config(state=tk.NORMAL)
        help_text = """Multi-File Visualization Help:

1. Use "Select Multiple Files" to choose multiple point cloud files at once
2. Or use "Add Single File" to add files one by one to your selection
3. Selected files will appear in the list above
4. Choose visualization options (point size, coloring, stride)
5. Select window arrangement:
   • Cascade: Windows arranged in overlapping cascade
   • Grid: Windows arranged in a grid pattern
   • Custom: Manual positioning (you arrange windows yourself)
6. Click "Visualize All Files" to open all selected files in separate windows
7. Or select a specific file and use "Visualize Selected" for single file viewing

Each file will open in its own independent visualization window for easy comparison."""
        self.multi_viz_status_text.insert(tk.END, help_text)
        self.multi_viz_status_text.config(state=tk.DISABLED)

    def create_cube_selection_tab(self):
        """Create the cube selection tab."""
        cube_frame = ttk.Frame(self.notebook)
        self.notebook.add(cube_frame, text="Cube Selection")
        
        # File selection for cube operations
        file_section = ttk.LabelFrame(cube_frame, text="Point Cloud Input", padding="10")
        file_section.pack(fill=tk.X, padx=10, pady=5)
        
        # Option to use currently loaded file or select new one
        self.cube_file_source_var = tk.StringVar(value="current")
        file_source_frame = ttk.Frame(file_section)
        file_source_frame.pack(fill=tk.X, pady=5)
        
        ttk.Radiobutton(file_source_frame, text="Use Current File", 
                       variable=self.cube_file_source_var, value="current").pack(side=tk.LEFT)
        ttk.Radiobutton(file_source_frame, text="Select New File", 
                       variable=self.cube_file_source_var, value="new").pack(side=tk.LEFT, padx=(20, 0))
        ttk.Radiobutton(file_source_frame, text="Use Analysis Results", 
                       variable=self.cube_file_source_var, value="analysis").pack(side=tk.LEFT, padx=(20, 0))
        
        # File path for new file selection
        new_file_frame = ttk.Frame(file_section)
        new_file_frame.pack(fill=tk.X, pady=5)
        
        self.cube_file_path_var = tk.StringVar()
        ttk.Entry(new_file_frame, textvariable=self.cube_file_path_var, width=50).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(new_file_frame, text="Browse...", command=self.browse_cube_file).pack(side=tk.RIGHT, padx=(10, 0))
        
        # Cube selection options
        options_section = ttk.LabelFrame(cube_frame, text="Cube Selection Options", padding="10")
        options_section.pack(fill=tk.X, padx=10, pady=5)
        
        # Window size
        size_frame = ttk.Frame(options_section)
        size_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(size_frame, text="Window Size:").pack(side=tk.LEFT)
        self.cube_window_width_var = tk.IntVar(value=1600)
        self.cube_window_height_var = tk.IntVar(value=1200)
        
        ttk.Spinbox(size_frame, from_=800, to=3840, increment=100, 
                   textvariable=self.cube_window_width_var, width=8).pack(side=tk.LEFT, padx=(10, 5))
        ttk.Label(size_frame, text="x").pack(side=tk.LEFT)
        ttk.Spinbox(size_frame, from_=600, to=2160, increment=100,
                   textvariable=self.cube_window_height_var, width=8).pack(side=tk.LEFT, padx=(5, 0))
        
        # Output filename
        output_frame = ttk.Frame(options_section)
        output_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(output_frame, text="Output Filename (without .npy):").pack(side=tk.LEFT)
        self.cube_output_name_var = tk.StringVar(value="interior_cockpit")
        ttk.Entry(output_frame, textvariable=self.cube_output_name_var, width=30).pack(side=tk.LEFT, padx=(10, 0))
        
        # Control buttons
        button_frame = ttk.Frame(cube_frame)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.start_cube_btn = ttk.Button(button_frame, text="Start Cube Selection",
                                        command=self.start_cube_selection)
        self.start_cube_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="Load Previous Selection",
                  command=self.load_cube_selection).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="View Selection",
                  command=self.view_cube_selection).pack(side=tk.LEFT, padx=5)
        
        # Results and status
        results_section = ttk.LabelFrame(cube_frame, text="Selection Results", padding="10")
        results_section.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.cube_results_text = scrolledtext.ScrolledText(results_section, height=15, state=tk.DISABLED)
        self.cube_results_text.pack(fill=tk.BOTH, expand=True)
        
        # Add initial help text
        self.cube_results_text.config(state=tk.NORMAL)
        help_text = """Cube Selection Help:

1. Choose your point cloud source:
   • Current File: Use the file selected in File Operations tab
   • Select New File: Browse for a different point cloud file  
   • Analysis Results: Use points from completed analysis

2. Adjust options:
   • Window Size: Visualization window dimensions
   • Output Filename: Name for saved selection (without .npy extension)

3. Click "Start Cube Selection" to open interactive 3D selection tool:
   • Drag cube handles to position and resize selection area
   • Green points = inside selection, Gray points = outside
   • Close window when satisfied with selection
   • Points will be automatically saved

4. Use "Load Previous Selection" to view saved selections
5. Use "View Selection" to visualize the current selection

The cube selector provides real-time visual feedback and saves results automatically."""
        self.cube_results_text.insert(tk.END, help_text)
        self.cube_results_text.config(state=tk.DISABLED)
    
    def create_human_model_tab(self):
        """Create the human model tab."""
        human_frame = ttk.Frame(self.notebook)
        self.notebook.add(human_frame, text="Human Model")
        
        # Human model options
        model_options = ttk.LabelFrame(human_frame, text="Human Model Options", padding="10")
        model_options.pack(fill=tk.X, padx=10, pady=5)
        
        # Model parameters
        ttk.Label(model_options, text="Human Model Scale:").pack(anchor=tk.W)
        self.human_scale_var = tk.DoubleVar(value=1.0)
        scale_frame = ttk.Frame(model_options)
        scale_frame.pack(fill=tk.X, pady=5)
        scale_spin = ttk.Spinbox(scale_frame, from_=0.5, to=2.0, increment=0.1,
                                textvariable=self.human_scale_var, width=10)
        scale_spin.pack(side=tk.LEFT)
        
        ttk.Label(model_options, text="Rotation (degrees):").pack(anchor=tk.W, pady=(10, 0))
        self.human_rotation_var = tk.DoubleVar(value=0.0)
        rotation_frame = ttk.Frame(model_options)
        rotation_frame.pack(fill=tk.X, pady=5)
        rotation_spin = ttk.Spinbox(rotation_frame, from_=0, to=360, increment=15,
                                   textvariable=self.human_rotation_var, width=10)
        rotation_spin.pack(side=tk.LEFT)
        
        # Human model buttons
        human_buttons = ttk.Frame(human_frame)
        human_buttons.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(human_buttons, text="Position Human Model",
                  command=self.position_human_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(human_buttons, text="Add to Scene",
                  command=self.add_human_to_scene).pack(side=tk.LEFT, padx=5)
        ttk.Button(human_buttons, text="Remove Human",
                  command=self.remove_human).pack(side=tk.LEFT, padx=5)
        
        # Human model info
        human_info = ttk.LabelFrame(human_frame, text="Human Model Status", padding="10")
        human_info.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.human_info_text = scrolledtext.ScrolledText(human_info, height=10, state=tk.DISABLED)
        self.human_info_text.pack(fill=tk.BOTH, expand=True)
    
    def create_log_tab(self):
        """Create the log tab."""
        log_frame = ttk.Frame(self.notebook)
        self.notebook.add(log_frame, text="Log")
        
        # Log text area
        self.log_text = scrolledtext.ScrolledText(log_frame, height=25, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Log controls
        log_controls = ttk.Frame(log_frame)
        log_controls.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        ttk.Button(log_controls, text="Clear Log", command=self.clear_log).pack(side=tk.LEFT)
        ttk.Button(log_controls, text="Save Log", command=self.save_log).pack(side=tk.LEFT, padx=(10, 0))
    
    def browse_file(self):
        """Browse for a point cloud file."""
        filetypes = [
            ("E57 files", "*.e57"),
            ("PCD files", "*.pcd"),
            ("ROS Bag files", "*.bag"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Select Point Cloud File",
            filetypes=filetypes
        )
        
        if filename:
            self.file_path_var.set(filename)
            self.current_file = filename
            self.analysis_btn.config(state=tk.NORMAL)
            self.log_message(f"Selected file: {filename}")
    
    @with_error_handling("load_file", reraise=False)
    def load_file(self):
        """Load and display file information."""
        if not self.current_file:
            messagebox.showerror("Error", "No file selected.")
            return
        
        try:
            # Validate file path for security
            validated_path = validate_file_path(
                self.current_file, must_exist=True,
                allowed_extensions=list(FileConfig.SUPPORTED_EXTENSIONS['e57']) + 
                                 list(FileConfig.SUPPORTED_EXTENSIONS['pcd']) +
                                 list(FileConfig.SUPPORTED_EXTENSIONS['ros_bag'])
            )
            self.current_file = str(validated_path)
        except Exception as e:
            self.error_handler.log_error(e, "file validation")
            messagebox.showerror("Error", f"Invalid file: {e}")
            return
        
        self.log_message(f"Loading file information for: {self.current_file}")
        
        # Run file loading in worker thread
        def load_file_info():
            try:
                file_path = Path(self.current_file)
                file_size = file_path.stat().st_size
                
                info = f"File: {file_path.name}\n"
                info += f"Path: {file_path}\n"
                info += f"Size: {file_size / (1024*1024):.2f} MB\n"
                info += f"Extension: {file_path.suffix}\n\n"
                
                if MODULES_AVAILABLE:
                    if file_path.suffix.lower() == '.e57':
                        info += "File type: E57 LIDAR format\n"
                        info += "Supported operations:\n"
                        info += "- Vehicle analysis\n"
                        info += "- Interior extraction\n"
                        info += "- 3D visualization\n"
                    elif file_path.suffix.lower() == '.pcd':
                        info += "File type: Point Cloud Data (PCD)\n"
                        info += "Supported operations:\n"
                        info += "- Raw visualization\n"
                        info += "- Point cloud analysis\n"
                    elif file_path.suffix.lower() == '.bag':
                        info += "File type: ROS Bag file\n"
                        info += "Supported operations:\n"
                        info += "- Topic extraction\n"
                        info += "- Point cloud visualization\n"
                
                return info
                
            except Exception as e:
                return f"Error loading file info: {str(e)}"
        
        def update_info(result):
            self.info_text.config(state=tk.NORMAL)
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(tk.END, result)
            self.info_text.config(state=tk.DISABLED)
            self.log_message("File information loaded.")
        
        def handle_error(error):
            self.log_message(f"Error loading file info: {error}")
            messagebox.showerror("Error", f"Failed to load file information: {error}")
        
        worker = WorkerThread(load_file_info, callback=update_info, error_callback=handle_error)
        worker.start()
    
    def start_analysis(self):
        """Start point cloud analysis."""
        if not self.current_file or not MODULES_AVAILABLE:
            messagebox.showerror("Error", "No file selected or modules not available.")
            return
        
        analysis_type = self.analysis_type_var.get()
        self.log_message(f"Starting {analysis_type} analysis...")
        
        self.analysis_btn.config(state=tk.DISABLED)
        self.progress_var.set(0)
        
        def run_analysis():
            try:
                if analysis_type == "vehicle":
                    return self.run_vehicle_analysis()
                else:
                    return self.run_raw_analysis()
            except Exception as e:
                raise e
        
        def analysis_complete(result):
            self.analysis_btn.config(state=tk.NORMAL)
            self.progress_var.set(100)
            self.analysis_results = result
            self.display_analysis_results(result)
            self.log_message("Analysis completed successfully.")
        
        def analysis_error(error):
            self.analysis_btn.config(state=tk.NORMAL)
            self.progress_var.set(0)
            self.log_message(f"Analysis failed: {error}")
            messagebox.showerror("Analysis Error", f"Analysis failed: {error}")
        
        worker = WorkerThread(run_analysis, callback=analysis_complete, error_callback=analysis_error)
        worker.start()
    
    def run_vehicle_analysis(self):
        """Run vehicle analysis."""
        analyzer = E57VehicleAnalyzer(self.current_file)
        
        # Update progress
        self.message_queue.put(("progress", 20))
        self.message_queue.put(("log", "Loading E57 data..."))
        
        analyzer.load_point_cloud_file()
        points = analyzer.points
        
        self.message_queue.put(("progress", 30))
        self.message_queue.put(("log", "Preprocessing points..."))
        
        analyzer.preprocess_points()
        
        self.message_queue.put(("progress", 50))
        self.message_queue.put(("log", "Identifying ground points..."))
        
        analyzer.separate_ground_points()
        ground_points = analyzer.ground_points
        non_ground_points = analyzer.non_ground_points
        
        self.message_queue.put(("progress", 70))
        self.message_queue.put(("log", "Identifying vehicle..."))
        
        analyzer.identify_vehicle_points()
        vehicle_points = analyzer.vehicle_points
        
        self.message_queue.put(("progress", 90))
        self.message_queue.put(("log", "Extracting interior..."))
        
        analyzer.find_vehicle_interior()
        interior_points = analyzer.interior_points
        
        self.message_queue.put(("progress", 100))
        
        # Ensure we have valid results before returning
        results = {
            "points": points if points is not None else np.array([]),
            "ground_points": ground_points if ground_points is not None else np.array([]),
            "non_ground_points": non_ground_points if non_ground_points is not None else np.array([]),
            "vehicle_points": vehicle_points if vehicle_points is not None else np.array([]),
            "interior_points": interior_points if interior_points is not None else np.array([])
        }
        
        # Log the results
        self.message_queue.put(("log", f"Analysis complete: {len(results['points']):,} total points"))
        self.message_queue.put(("log", f"  Ground: {len(results['ground_points']):,}, Vehicle: {len(results['vehicle_points']):,}, Interior: {len(results['interior_points']):,}"))
        
        return results
    
    def run_raw_analysis(self):
        """Run raw file analysis."""
        self.message_queue.put(("progress", 50))
        self.message_queue.put(("log", "Loading raw point cloud data..."))
        
        points, intensity, rgb = read_points(self.current_file)
        
        self.message_queue.put(("progress", 100))
        
        return {
            "points": points,
            "intensity": intensity,
            "rgb": rgb
        }
    
    def display_analysis_results(self, results):
        """Display analysis results."""
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        
        result_str = "Analysis Results:\n\n"
        
        if "points" in results:
            result_str += f"Total points: {len(results['points']):,}\n"
        
        if "ground_points" in results:
            result_str += f"Ground points: {len(results['ground_points']):,}\n"
        
        if "non_ground_points" in results:
            result_str += f"Non-ground points: {len(results['non_ground_points']):,}\n"
        
        if "vehicle_points" in results:
            result_str += f"Vehicle points: {len(results['vehicle_points']):,}\n"
        
        if "interior_points" in results:
            result_str += f"Interior points: {len(results['interior_points']):,}\n"
        
        if "intensity" in results and results["intensity"] is not None:
            result_str += f"Has intensity data: Yes\n"
        
        if "rgb" in results and results["rgb"] is not None:
            result_str += f"Has RGB data: Yes\n"
        
        self.results_text.insert(tk.END, result_str)
        self.results_text.config(state=tk.DISABLED)
    
    def visualize_raw_file(self):
        """Visualize raw point cloud file directly using visualise_raw_lidar_files functionality."""
        if not VTK_AVAILABLE or not self.current_file:
            messagebox.showwarning("Warning", "No file selected or VTK not available.")
            return
        
        if not os.path.exists(self.current_file):
            messagebox.showerror("Error", "Selected file does not exist.")
            return
        
        self.log_message(f"Opening raw file visualization for: {os.path.basename(self.current_file)}")
        
        def load_and_visualize():
            try:
                # Prepare parameters
                scan_indices = None
                if self.scan_indices_var.get().strip():
                    try:
                        scan_indices = [int(x.strip()) for x in self.scan_indices_var.get().split(',')]
                    except ValueError:
                        self.message_queue.put(("log", "Warning: Invalid scan indices format, using all scans"))
                        scan_indices = None
                
                stride = max(1, self.stride_var.get())
                topic = self.ros_topic_var.get().strip() or None
                frame_index = self.frame_index_var.get()
                
                self.message_queue.put(("log", f"Loading with stride={stride}, topic={topic}, frame={frame_index}"))
                
                # Load the point cloud data
                points, intensity, extra_data = read_points(
                    self.current_file, 
                    scan_indices=scan_indices,
                    stride=stride,
                    topic=topic,
                    frame_index=frame_index
                )
                
                self.message_queue.put(("log", f"Loaded {len(points):,} points"))
                
                return points, intensity, extra_data
                
            except Exception as e:
                raise Exception(f"Failed to load point cloud: {str(e)}")
        
        def create_visualization(data):
            points, intensity, extra_data = data
            
            plotter = pv.Plotter(window_size=(1280, 800))
            plotter.set_background("black")
            
            point_cloud = pv.PolyData(points)
            
            # Determine coloring based on user selection and available data
            color_mode = self.color_mode_var.get()
            point_size = self.point_size_var.get()
            render_as_spheres = self.spheres_var.get()
            
            if color_mode == "intensity" and intensity is not None:
                point_cloud["intensity"] = intensity
                plotter.add_mesh(point_cloud, scalars="intensity", 
                               point_size=point_size, render_points_as_spheres=render_as_spheres)
                
            elif color_mode == "elevation":
                elevation = points[:, 2]  # Z coordinate
                point_cloud["elevation"] = elevation
                plotter.add_mesh(point_cloud, scalars="elevation", 
                               point_size=point_size, render_points_as_spheres=render_as_spheres)
                
            elif color_mode == "rgb" and extra_data is not None:
                # Check if extra_data looks like RGB (3 columns)
                if extra_data.shape[1] >= 3:
                    point_cloud["rgb"] = extra_data[:, :3]
                    plotter.add_mesh(point_cloud, scalars="rgb", rgb=True,
                                   point_size=point_size, render_points_as_spheres=render_as_spheres)
                else:
                    # Fall back to white
                    plotter.add_mesh(point_cloud, color="white", 
                                   point_size=point_size, render_points_as_spheres=render_as_spheres)
                    
            elif color_mode == "ring" and extra_data is not None:
                # Velodyne ring data (usually single column)
                if extra_data.shape[1] >= 1:
                    point_cloud["ring"] = extra_data[:, 0]
                    plotter.add_mesh(point_cloud, scalars="ring", 
                                   point_size=point_size, render_points_as_spheres=render_as_spheres)
                else:
                    # Fall back to white
                    plotter.add_mesh(point_cloud, color="white", 
                                   point_size=point_size, render_points_as_spheres=render_as_spheres)
            else:
                # Default white coloring
                plotter.add_mesh(point_cloud, color="white", 
                               point_size=point_size, render_points_as_spheres=render_as_spheres)
            
            # Add coordinate axes and title
            plotter.add_axes()
            title = f"Raw Point Cloud: {os.path.basename(self.current_file)}"
            if self.current_file.endswith('.bag'):
                title += f" (Frame 0)"  # Default to frame 0 for bag files
            plotter.add_title(title)
            
            self.active_plotters.append(plotter)
            plotter.show(auto_close=False, interactive_update=True)
        
        def visualization_complete(result):
            self.root.after(0, lambda: create_visualization(result))
            self.log_message("Raw file visualization opened.")
        
        def visualization_error(error):
            self.log_message(f"Raw file visualization failed: {error}")
            messagebox.showerror("Visualization Error", f"Failed to visualize file: {error}")
        
        # Load data in worker thread, then create visualization in main thread
        worker = WorkerThread(load_and_visualize, callback=visualization_complete, error_callback=visualization_error)
        worker.start()
    
    def visualize_raw(self):
        """Visualize raw point cloud data from analysis results."""
        if not VTK_AVAILABLE or not self.analysis_results:
            messagebox.showwarning("Warning", "No analysis data available or VTK not available.")
            return
        
        self.log_message("Opening raw analysis data visualization...")
        
        def create_visualization():
            plotter = pv.Plotter(window_size=(1280, 800))
            plotter.set_background("black")
            
            if "points" in self.analysis_results:
                points = self.analysis_results["points"]
                point_cloud = pv.PolyData(points)
                
                # Add intensity or RGB coloring if available
                if "intensity" in self.analysis_results and self.analysis_results["intensity"] is not None:
                    point_cloud["intensity"] = self.analysis_results["intensity"]
                    plotter.add_mesh(point_cloud, scalars="intensity", point_size=self.point_size_var.get())
                elif "rgb" in self.analysis_results and self.analysis_results["rgb"] is not None:
                    point_cloud["rgb"] = self.analysis_results["rgb"]
                    plotter.add_mesh(point_cloud, scalars="rgb", point_size=self.point_size_var.get())
                else:
                    plotter.add_mesh(point_cloud, color="white", point_size=self.point_size_var.get())
            
            plotter.add_title("Raw Analysis Data")
            self.active_plotters.append(plotter)
            plotter.show(auto_close=False, interactive_update=True)
        
        # Run visualization in main thread
        self.root.after(0, create_visualization)
    
    def visualize_results(self):
        """Visualize analysis results."""
        if not VTK_AVAILABLE or not self.analysis_results:
            messagebox.showwarning("Warning", "No analysis results available or VTK not available.")
            return
        
        self.log_message("Opening analysis results visualization...")
        
        def create_visualization():
            plotter = pv.Plotter(window_size=(1280, 800))
            plotter.set_background("black")
            
            colors = ["gray", "green", "blue", "red"]
            labels = ["Ground", "Non-ground", "Vehicle", "Interior"]
            point_data = ["ground_points", "non_ground_points", "vehicle_points", "interior_points"]
            
            for key, color, label in zip(point_data, colors, labels):
                if key in self.analysis_results and self.analysis_results[key] is not None:
                    points = self.analysis_results[key]
                    if len(points) > 0:
                        point_cloud = pv.PolyData(points)
                        plotter.add_mesh(point_cloud, color=color, point_size=self.point_size_var.get(), 
                                       label=label)
            
            plotter.add_legend()
            self.active_plotters.append(plotter)
            plotter.show(auto_close=False, interactive_update=True)
        
        # Run visualization in main thread
        self.root.after(0, create_visualization)
    
    def visualize_output(self):
        """Visualize saved output files."""
        if not VTK_AVAILABLE:
            messagebox.showwarning("Warning", "VTK not available.")
            return
        
        self.log_message("Opening output files visualization...")
        
        def load_output_data():
            """Load output data in background thread (safe)."""
            try:
                visualizer = OutputVisualizer()
                visualizer.load_output_files()
                return visualizer
            except Exception as e:
                raise e
        
        def create_visualization(visualizer):
            """Create VTK visualization on main thread (required for macOS)."""
            try:
                # Create visualization with plotter tracking callback
                def plotter_callback(plotter):
                    """Track the plotter for proper cleanup."""
                    if plotter:
                        self.active_plotters.append(plotter)
                
                visualizer.create_3d_visualization(plotter_callback=plotter_callback)
                self.log_message("Output visualization completed.")
            except Exception as e:
                self.log_message(f"Output visualization failed: {e}")
                messagebox.showerror("Visualization Error", f"Failed to visualize output: {e}")
        
        def data_loaded(visualizer):
            """Callback when data loading is complete - schedule visualization on main thread."""
            # Schedule visualization to run on main thread
            self.root.after(0, lambda: create_visualization(visualizer))
        
        def load_error(error):
            self.log_message(f"Output data loading failed: {error}")
            messagebox.showerror("Load Error", f"Failed to load output data: {error}")
        
        # Load data in background thread, create visualization on main thread
        worker = WorkerThread(load_output_data, callback=data_loaded, error_callback=load_error)
        worker.start()
    
    def browse_cube_file(self):
        """Browse for a point cloud file for cube selection."""
        filetypes = [
            ("E57 files", "*.e57"),
            ("PCD files", "*.pcd"), 
            ("ROS Bag files", "*.bag"),
            ("NumPy files", "*.npy"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Select Point Cloud File for Cube Selection",
            filetypes=filetypes
        )
        
        if filename:
            self.cube_file_path_var.set(filename)
            self.log_message(f"Selected cube selection file: {filename}")
    
    def start_cube_selection(self):
        """Start the interactive cube selection process."""
        if not VTK_AVAILABLE:
            messagebox.showerror("Error", "VTK not available. Cannot run cube selection.")
            return
        
        # Determine point cloud source
        source = self.cube_file_source_var.get()
        points = None
        source_description = ""
        
        if source == "current":
            if not self.current_file:
                messagebox.showerror("Error", "No current file selected. Please select a file in the File Operations tab.")
                return
            source_description = f"current file: {os.path.basename(self.current_file)}"
            
        elif source == "new":
            file_path = self.cube_file_path_var.get()
            if not file_path or not os.path.exists(file_path):
                messagebox.showerror("Error", "Please select a valid point cloud file.")
                return
            source_description = f"selected file: {os.path.basename(file_path)}"
            
        elif source == "analysis":
            if not self.analysis_results or "points" not in self.analysis_results:
                messagebox.showerror("Error", "No analysis results available. Please run analysis first.")
                return
            points = self.analysis_results["points"]
            source_description = "analysis results"
        
        self.log_message(f"Starting cube selection from {source_description}")
        self.start_cube_btn.config(state=tk.DISABLED)
        
        def load_points():
            nonlocal points
            try:
                if points is None:
                    # Load points from file
                    if source == "current":
                        file_path = self.current_file
                    else:  # source == "new"
                        file_path = self.cube_file_path_var.get()
                    
                    # Check if it's a numpy file
                    if file_path.endswith('.npy'):
                        points = np.load(file_path)
                        if points.shape[1] != 3:
                            raise ValueError(f"Invalid point cloud shape: {points.shape}. Expected (N, 3)")
                    else:
                        # Use read_points function
                        points, _, _ = read_points(file_path)
                
                self.message_queue.put(("log", f"Loaded {len(points):,} points for cube selection"))
                return points
                
            except Exception as e:
                raise Exception(f"Failed to load points: {str(e)}")
        
        def run_cube_selection(loaded_points):
            try:
                # Get window size and output name
                window_size = (self.cube_window_width_var.get(), self.cube_window_height_var.get())
                output_name = self.cube_output_name_var.get().strip() or "interior_cockpit"
                
                # Create message callback for the cube selector
                def cube_message_callback(message):
                    self.message_queue.put(("log", message))
                
                # Create plotter callback for tracking (but don't use it here since we're in worker thread)
                def cube_plotter_callback(plotter):
                    """Track the plotter for proper cleanup."""
                    if plotter:
                        self.active_plotters.append(plotter)
                
                # Run cube selection
                manager = CubeSelectionManager()
                selected_points = manager.select_from_file(
                    loaded_points, 
                    filename=output_name, 
                    window_size=window_size,
                    message_callback=cube_message_callback,
                    plotter_callback=cube_plotter_callback
                )
                
                return selected_points, output_name
                
            except Exception as e:
                raise Exception(f"Cube selection failed: {str(e)}")
        
        def cube_selection_complete(result):
            selected_points, output_name = result
            self.start_cube_btn.config(state=tk.NORMAL)
            
            if selected_points is not None:
                # Update results text
                self.cube_results_text.config(state=tk.NORMAL)
                self.cube_results_text.delete(1.0, tk.END)
                
                result_text = f"""Cube Selection Results:

Source: {source_description}
Total Points: {len(points):,}
Selected Points: {len(selected_points):,}
Selection Ratio: {len(selected_points)/len(points)*100:.1f}%
Output File: {output_name}.npy

Cube selection completed successfully!
Selected points have been saved to the output directory.

You can now use "View Selection" to visualize the results."""
                
                self.cube_results_text.insert(tk.END, result_text)
                self.cube_results_text.config(state=tk.DISABLED)
                
                self.log_message("Cube selection completed successfully!")
            else:
                self.log_message("Cube selection was cancelled or failed.")
                self.cube_results_text.config(state=tk.NORMAL)
                self.cube_results_text.delete(1.0, tk.END)
                self.cube_results_text.insert(tk.END, "Cube selection was cancelled or failed.")
                self.cube_results_text.config(state=tk.DISABLED)
        
        def cube_selection_error(error):
            self.start_cube_btn.config(state=tk.NORMAL)
            self.log_message(f"Cube selection failed: {error}")
            messagebox.showerror("Cube Selection Error", f"Failed to run cube selection: {error}")
        
        # Load points in worker thread, then run cube selection on main thread
        def load_points_worker():
            try:
                return load_points()
            except Exception as e:
                cube_selection_error(e)
                return None
        
        def points_loaded(loaded_points):
            if loaded_points is None:
                return
            
            # Run cube selection on main thread (required for VTK on macOS)
            def run_on_main_thread():
                try:
                    result = run_cube_selection(loaded_points)
                    cube_selection_complete(result)
                except Exception as e:
                    cube_selection_error(e)
            
            # Schedule cube selection to run on main thread
            self.root.after(0, run_on_main_thread)
        
        worker = WorkerThread(load_points_worker, callback=points_loaded)
        worker.start()
    
    def load_cube_selection(self):
        """Load and display information about saved cube selections."""
        manager = CubeSelectionManager()
        saved_files = manager.list_saved_selections()
        
        if not saved_files:
            messagebox.showinfo("No Selections", "No saved cube selections found in the output directory.")
            return
        
        # Show list of saved selections
        selection_dialog = tk.Toplevel(self.root)
        selection_dialog.title("Saved Cube Selections")
        selection_dialog.geometry("500x400")
        
        ttk.Label(selection_dialog, text="Select a saved cube selection:", font=("Arial", 12)).pack(pady=10)
        
        # List of saved files
        listbox_frame = ttk.Frame(selection_dialog)
        listbox_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        listbox = tk.Listbox(listbox_frame, height=15)
        scrollbar = ttk.Scrollbar(listbox_frame, orient=tk.VERTICAL, command=listbox.yview)
        listbox.config(yscrollcommand=scrollbar.set)
        
        for file in saved_files:
            listbox.insert(tk.END, file)
        
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Buttons
        button_frame = ttk.Frame(selection_dialog)
        button_frame.pack(fill=tk.X, padx=20, pady=10)
        
        def load_selected():
            selection = listbox.curselection()
            if not selection:
                messagebox.showwarning("No Selection", "Please select a file to load.")
                return
            
            filename = listbox.get(selection[0])
            points = manager.load_previous_selection(filename)
            
            if points is not None:
                self.cube_results_text.config(state=tk.NORMAL)
                self.cube_results_text.delete(1.0, tk.END)
                
                result_text = f"""Loaded Cube Selection:

File: {filename}
Points: {len(points):,}

Selection loaded successfully!
Use "View Selection" to visualize the loaded points."""
                
                self.cube_results_text.insert(tk.END, result_text)
                self.cube_results_text.config(state=tk.DISABLED)
                
                self.log_message(f"Loaded cube selection: {filename}")
                selection_dialog.destroy()
            else:
                messagebox.showerror("Load Error", f"Failed to load selection from {filename}")
        
        ttk.Button(button_frame, text="Load Selected", command=load_selected).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=selection_dialog.destroy).pack(side=tk.RIGHT, padx=5)
    
    def view_cube_selection(self):
        """Visualize the current cube selection."""
        if not VTK_AVAILABLE:
            messagebox.showwarning("Warning", "VTK not available for visualization.")
            return
        
        # Try to load the most recent cube selection
        output_name = self.cube_output_name_var.get().strip() or "interior_cockpit"
        manager = CubeSelectionManager()
        
        # Try the specified output name first
        selected_points = manager.load_previous_selection(f"{output_name}.npy")
        
        # If not found, try default name
        if selected_points is None:
            selected_points = manager.load_previous_selection("interior_cockpit.npy")
        
        if selected_points is None:
            messagebox.showwarning("No Selection", "No cube selection found to visualize. Run cube selection first.")
            return
        
        self.log_message(f"Visualizing cube selection: {len(selected_points):,} points")
        
        def create_visualization():
            plotter = pv.Plotter(window_size=(1280, 800))
            plotter.set_background("black")
            
            # Add selected points
            point_cloud = pv.PolyData(selected_points)
            plotter.add_mesh(point_cloud, color="lime", point_size=4, 
                           label=f"Selected Points ({len(selected_points):,})")
            
            # Add axes and title
            plotter.add_axes()
            plotter.add_title(f"Cube Selection Visualization - {len(selected_points):,} points")
            
            self.active_plotters.append(plotter)
            plotter.show(auto_close=False, interactive_update=True)
        
        # Run visualization in main thread
        self.root.after(0, create_visualization)
    
    def position_human_model(self):
        """Open human model positioning interface."""
        if not VTK_AVAILABLE:
            messagebox.showwarning("Warning", "VTK not available.")
            return
        
        self.log_message("Opening human model positioner...")
        
        def load_positioner_data():
            """Load positioner data in background thread (safe)."""
            try:
                # Look for an existing interior cockpit file  
                output_name = self.cube_output_name_var.get().strip() or "interior_cockpit"
                cockpit_file = f"output/{output_name}.npy"
                
                if not os.path.exists(cockpit_file):
                    # Try default name
                    cockpit_file = "output/interior_cockpit.npy"
                    if not os.path.exists(cockpit_file):
                        raise FileNotFoundError("No interior cockpit file found. Run cube selection first.")
                
                # Create positioner but don't run visualization yet
                positioner = InteractiveHumanPositioner(cockpit_file)
                return positioner
            except Exception as e:
                raise e
        
        def run_positioning_visualization(positioner):
            """Run VTK visualization on main thread (required for macOS)."""
            try:
                # Create plotter callback for tracking
                def plotter_callback(plotter):
                    """Track the plotter for proper cleanup."""
                    if plotter:
                        self.active_plotters.append(plotter)
                
                result = positioner.run_interactive_positioning(plotter_callback=plotter_callback)
                self.log_message("Human model positioning completed.")
                self.human_info_text.config(state=tk.NORMAL)
                self.human_info_text.insert(tk.END, f"Human model positioned: {result}\n")
                self.human_info_text.config(state=tk.DISABLED)
            except Exception as e:
                self.log_message(f"Human model positioning failed: {e}")
                messagebox.showerror("Positioning Error", f"Failed to position human model: {e}")
        
        def data_loaded(positioner):
            """Callback when data loading is complete - schedule visualization on main thread."""
            # Schedule visualization to run on main thread
            self.root.after(0, lambda: run_positioning_visualization(positioner))
        
        def load_error(error):
            self.log_message(f"Human model data loading failed: {error}")
            messagebox.showerror("Load Error", f"Failed to load human model data: {error}")
        
        # Load data in background thread, run positioning on main thread
        worker = WorkerThread(load_positioner_data, callback=data_loaded, error_callback=load_error)
        worker.start()
    
    def add_human_to_scene(self):
        """Add human model to current scene."""
        self.log_message("Adding human model to scene...")
        messagebox.showinfo("Info", "Human model integration not yet implemented.")
    
    def remove_human(self):
        """Remove human model from scene."""
        self.log_message("Removing human model from scene...")
        messagebox.showinfo("Info", "Human model removal not yet implemented.")
    
    def log_message(self, message):
        """Add message to log."""
        self.message_queue.put(("log", message))
    
    def clear_log(self):
        """Clear the log."""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)
    
    def save_log(self):
        """Save log to file."""
        filename = filedialog.asksaveasfilename(
            title="Save Log",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filename:
            with open(filename, 'w') as f:
                f.write(self.log_text.get(1.0, tk.END))
            self.log_message(f"Log saved to: {filename}")
    
    def process_message_queue(self):
        """Process messages from worker threads."""
        try:
            while True:
                msg_type, msg_data = self.message_queue.get_nowait()
                
                if msg_type == "log":
                    self.log_text.config(state=tk.NORMAL)
                    self.log_text.insert(tk.END, f"{msg_data}\n")
                    self.log_text.see(tk.END)
                    self.log_text.config(state=tk.DISABLED)
                
                elif msg_type == "progress":
                    self.progress_var.set(msg_data)
                
                elif msg_type == "status":
                    self.status_bar.config(text=msg_data)
        
        except queue.Empty:
            pass
        
        # Schedule next check
        self.root.after(100, self.process_message_queue)

    def select_multiple_files(self):
        """Select multiple point cloud files for visualization."""
        filetypes = [
            ("Point Cloud files", "*.e57 *.pcd *.bag"),
            ("E57 files", "*.e57"),
            ("PCD files", "*.pcd"),
            ("ROS Bag files", "*.bag"),
            ("All files", "*.*")
        ]

        filenames = filedialog.askopenfilenames(
            title="Select Multiple Point Cloud Files",
            filetypes=filetypes
        )

        if filenames:
            # Add new files to the list (avoid duplicates)
            added_count = 0
            for filename in filenames:
                if filename not in self.multi_file_list:
                    self.multi_file_list.append(filename)
                    self.multi_files_listbox.insert(tk.END, os.path.basename(filename))
                    added_count += 1

            # Update button state and status
            if self.multi_file_list:
                self.visualize_all_btn.config(state=tk.NORMAL)

            self.update_multi_viz_status(f"Added {added_count} files. Total: {len(self.multi_file_list)} files selected.")
            self.log_message(f"Selected {added_count} files for multi-visualization. Total: {len(self.multi_file_list)} files.")

    def add_single_file(self):
        """Add a single point cloud file to the selection."""
        filetypes = [
            ("E57 files", "*.e57"),
            ("PCD files", "*.pcd"),
            ("ROS Bag files", "*.bag"),
            ("All files", "*.*")
        ]

        filename = filedialog.askopenfilename(
            title="Add Single Point Cloud File",
            filetypes=filetypes
        )

        if filename:
            if filename not in self.multi_file_list:
                self.multi_file_list.append(filename)
                self.multi_files_listbox.insert(tk.END, os.path.basename(filename))

                # Update button state
                if self.multi_file_list:
                    self.visualize_all_btn.config(state=tk.NORMAL)

                self.update_multi_viz_status(f"Added {os.path.basename(filename)}. Total: {len(self.multi_file_list)} files selected.")
                self.log_message(f"Added file to multi-visualization: {filename}")
            else:
                messagebox.showinfo("Duplicate File", f"File {os.path.basename(filename)} is already in the selection.")

    def clear_file_selection(self):
        """Clear all selected files."""
        self.multi_file_list.clear()
        self.multi_files_listbox.delete(0, tk.END)
        self.visualize_all_btn.config(state=tk.DISABLED)
        self.update_multi_viz_status("File selection cleared.")
        self.log_message("Cleared multi-file selection.")

    def remove_selected_file(self):
        """Remove the selected file from the list."""
        selection = self.multi_files_listbox.curselection()
        if selection:
            index = selection[0]
            filename = self.multi_file_list[index]

            # Remove from both list and listbox
            del self.multi_file_list[index]
            self.multi_files_listbox.delete(index)

            # Update button state
            if not self.multi_file_list:
                self.visualize_all_btn.config(state=tk.DISABLED)

            self.update_multi_viz_status(f"Removed {os.path.basename(filename)}. Total: {len(self.multi_file_list)} files selected.")
            self.log_message(f"Removed file from multi-visualization: {filename}")
        else:
            messagebox.showwarning("No Selection", "Please select a file to remove.")

    def view_file_info(self):
        """Display information about the selected file."""
        selection = self.multi_files_listbox.curselection()
        if selection:
            index = selection[0]
            filename = self.multi_file_list[index]

            try:
                file_path = Path(filename)
                file_size = file_path.stat().st_size

                info = f"File: {file_path.name}\n"
                info += f"Path: {file_path}\n"
                info += f"Size: {file_size / (1024*1024):.2f} MB\n"
                info += f"Extension: {file_path.suffix}\n"

                messagebox.showinfo("File Information", info)

            except Exception as e:
                messagebox.showerror("Error", f"Could not get file information: {e}")
        else:
            messagebox.showwarning("No Selection", "Please select a file to view information.")

    def visualize_selected_file(self):
        """Visualize the currently selected file in the listbox."""
        if not VTK_AVAILABLE:
            messagebox.showwarning("Warning", "VTK not available for visualization.")
            return

        selection = self.multi_files_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a file to visualize.")
            return

        index = selection[0]
        filename = self.multi_file_list[index]
        self.visualize_single_file(filename, f"File {index + 1}")

    def visualize_all_files(self):
        """Visualize all selected files in separate windows."""
        if not VTK_AVAILABLE:
            messagebox.showwarning("Warning", "VTK not available for visualization.")
            return

        if not self.multi_file_list:
            messagebox.showwarning("No Files", "No files selected for visualization.")
            return

        self.update_multi_viz_status(f"Starting visualization of {len(self.multi_file_list)} files...")
        self.log_message(f"Starting multi-file visualization for {len(self.multi_file_list)} files")

        # Get window arrangement preference
        arrangement = self.window_arrangement_var.get()

        # Visualize each file
        for i, filename in enumerate(self.multi_file_list):
            window_title = f"File {i + 1}: {os.path.basename(filename)}"
            self.visualize_single_file(filename, window_title, position_index=i, arrangement=arrangement)

        self.update_multi_viz_status(f"Opened {len(self.multi_file_list)} visualization windows.")

    def visualize_single_file(self, filename, window_title, position_index=0, arrangement="custom"):
        """Visualize a single file with independent window positioning."""
        self.log_message(f"Opening visualization for: {os.path.basename(filename)}")

        def load_and_visualize():
            try:
                # Load the point cloud data
                stride = max(1, self.multi_stride_var.get())

                self.message_queue.put(("log", f"Loading {os.path.basename(filename)} with stride={stride}"))

                points, intensity, extra_data = read_points(filename, stride=stride)

                self.message_queue.put(("log", f"Loaded {len(points):,} points from {os.path.basename(filename)}"))

                return points, intensity, extra_data, filename, window_title, position_index, arrangement

            except Exception as e:
                raise Exception(f"Failed to load {os.path.basename(filename)}: {str(e)}")

        def create_visualization(data):
            points, intensity, extra_data, filename, window_title, position_index, arrangement = data

            # Calculate window position based on arrangement
            window_size = (800, 600)  # Default window size for multi-file viz
            window_position = self.calculate_window_position(position_index, arrangement, window_size)

            plotter = pv.Plotter(window_size=window_size, off_screen=False)
            plotter.set_background("black")

            point_cloud = pv.PolyData(points)

            # Determine coloring based on user selection and available data
            color_mode = self.multi_color_mode_var.get()
            point_size = self.multi_point_size_var.get()
            render_as_spheres = self.multi_spheres_var.get()

            if color_mode == "intensity" and intensity is not None:
                point_cloud["intensity"] = intensity
                plotter.add_mesh(point_cloud, scalars="intensity",
                               point_size=point_size, render_points_as_spheres=render_as_spheres)

            elif color_mode == "elevation":
                elevation = points[:, 2]  # Z coordinate
                point_cloud["elevation"] = elevation
                plotter.add_mesh(point_cloud, scalars="elevation",
                               point_size=point_size, render_points_as_spheres=render_as_spheres)

            elif color_mode == "rgb" and extra_data is not None:
                # Check if extra_data looks like RGB (3 columns)
                if extra_data.shape[1] >= 3:
                    point_cloud["rgb"] = extra_data[:, :3]
                    plotter.add_mesh(point_cloud, scalars="rgb", rgb=True,
                                   point_size=point_size, render_points_as_spheres=render_as_spheres)
                else:
                    # Fall back to white
                    plotter.add_mesh(point_cloud, color="white",
                                   point_size=point_size, render_points_as_spheres=render_as_spheres)

            elif color_mode == "ring" and extra_data is not None:
                # Velodyne ring data (usually single column)
                if extra_data.shape[1] >= 1:
                    point_cloud["ring"] = extra_data[:, 0]
                    plotter.add_mesh(point_cloud, scalars="ring",
                                   point_size=point_size, render_points_as_spheres=render_as_spheres)
                else:
                    # Fall back to white
                    plotter.add_mesh(point_cloud, color="white",
                                   point_size=point_size, render_points_as_spheres=render_as_spheres)
            else:
                # Default white coloring
                plotter.add_mesh(point_cloud, color="white",
                               point_size=point_size, render_points_as_spheres=render_as_spheres)

            # Add coordinate axes and title
            plotter.add_axes()
            plotter.add_title(window_title)

            # Track plotter for cleanup
            self.active_plotters.append(plotter)

            # Show with position if specified
            if window_position and arrangement != "custom":
                plotter.show(window_size=window_size, auto_close=False, interactive_update=True)
                # Note: PyVista doesn't directly support window positioning, so we use custom arrangement conceptually
            else:
                plotter.show(auto_close=False, interactive_update=True)

        def visualization_complete(result):
            self.root.after(0, lambda: create_visualization(result))

        def visualization_error(error):
            self.log_message(f"Visualization failed for {os.path.basename(filename)}: {error}")
            self.update_multi_viz_status(f"Failed to visualize {os.path.basename(filename)}: {error}")

        # Load data in worker thread, then create visualization in main thread
        worker = WorkerThread(load_and_visualize, callback=visualization_complete, error_callback=visualization_error)
        worker.start()

    def calculate_window_position(self, index, arrangement, window_size):
        """Calculate window position based on arrangement strategy."""
        if arrangement == "cascade":
            # Cascade windows with 30px offset
            offset = 30 * index
            return (100 + offset, 100 + offset)
        elif arrangement == "grid":
            # Arrange in grid pattern (2 columns)
            cols = 2
            row = index // cols
            col = index % cols
            x = 100 + col * (window_size[0] + 50)
            y = 100 + row * (window_size[1] + 100)
            return (x, y)
        else:  # custom
            return None

    def close_all_multi_windows(self):
        """Close all visualization windows opened from multi-file tab."""
        closed_count = 0
        for plotter in self.active_plotters[:]:  # Create a copy to iterate over
            try:
                VTKSafetyManager.cleanup_vtk_plotter(plotter)
                self.active_plotters.remove(plotter)
                closed_count += 1
            except Exception as e:
                self.log_message(f"Error closing plotter: {e}")

        # Force garbage collection
        gc.collect()

        self.update_multi_viz_status(f"Closed {closed_count} visualization windows.")
        self.log_message(f"Closed {closed_count} visualization windows.")

    def update_multi_viz_status(self, message):
        """Update the status text in the multi-file visualization tab."""
        self.multi_viz_status_text.config(state=tk.NORMAL)
        self.multi_viz_status_text.insert(tk.END, f"{message}\n")
        self.multi_viz_status_text.see(tk.END)
        self.multi_viz_status_text.config(state=tk.DISABLED)

    def on_closing(self):
        """Handle application closing."""
        self.log_message("Cleaning up and closing application...")
        
        # Clean up all VTK plotters
        for plotter in self.active_plotters:
            VTKSafetyManager.cleanup_vtk_plotter(plotter)
        
        self.active_plotters.clear()
        
        # Force garbage collection
        gc.collect()
        
        # Destroy the window
        self.root.destroy()
    
    def run(self):
        """Run the application."""
        self.log_message("LIDAR Point Cloud Analysis Suite started.")
        self.root.mainloop()


def main():
    """Main entry point."""
    if not MODULES_AVAILABLE:
        print("Warning: Point cloud modules not available. Some functionality may be limited.")
    
    app = LidarGUIApp()
    app.run()


if __name__ == "__main__":
    main()