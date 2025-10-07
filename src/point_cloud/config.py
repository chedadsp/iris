#!/usr/bin/env python3
"""
Configuration Constants for LIDAR Point Cloud Processing

Centralizes all configuration values, thresholds, and parameters used throughout
the application to improve maintainability and allow easy tuning.

Author: Dimitrije Stojanovic
Date: September 2025
Created: Code Review Fixes
"""

from typing import Dict, Any
from pathlib import Path


class AnalysisConfig:
    """Configuration for point cloud analysis algorithms."""
    
    # Ground separation parameters
    GROUND_HEIGHT_THRESHOLD = 0.5  # meters above lowest point to consider for ground detection
    GROUND_PLANE_TOLERANCE = 0.1   # 10cm tolerance for RANSAC plane fitting
    GROUND_FINAL_THRESHOLD = 0.15  # 15cm above predicted ground for final separation
    SIMPLE_GROUND_THRESHOLD = 0.2  # 20cm above lowest point for simple method
    
    # RANSAC parameters for ground plane fitting
    RANSAC_MAX_TRIALS = 1000
    RANSAC_MIN_SAMPLES = 100  # Minimum points needed for ground fitting
    
    # Vehicle identification parameters
    DBSCAN_EPS = 0.3           # DBSCAN clustering epsilon (30cm)
    DBSCAN_MIN_SAMPLES = 50    # Minimum samples for DBSCAN cluster
    VEHICLE_MAX_HEIGHT = 3.0   # Maximum vehicle height in meters
    
    # Interior detection parameters
    SEAT_HEIGHT_OFFSET = 0.4   # Height above floor for seat level (40cm)
    HEAD_HEIGHT_OFFSET = 1.8   # Height above floor for head level when sitting (1.8m)
    COCKPIT_HEIGHT_MIN = 0.2   # Minimum cockpit height for fallback (20cm)
    COCKPIT_HEIGHT_MAX = 1.4   # Maximum cockpit height for fallback (1.4m)
    
    # 3D occupancy grid parameters
    GRID_RESOLUTION = 0.1      # 10cm voxel size for 3D grids
    INTERIOR_THRESHOLD_3D = 2  # Distance threshold for interior detection (20cm)
    
    # Dashboard detection parameters
    DASHBOARD_HEIGHT_MIN = 0.8  # Dashboard minimum height (80cm above floor)
    DASHBOARD_HEIGHT_MAX = 1.4  # Dashboard maximum height (1.4m above floor)
    
    # Cockpit geometry parameters
    COCKPIT_FRONT_RATIO = 0.33     # Front 2/3 of vehicle for cockpit (33% from front)
    COCKPIT_SIDE_TOLERANCE = 0.35  # Within 35% of center width for seats
    COCKPIT_SEAT_HEIGHT_MIN = 0.6  # Minimum seated passenger height (60cm)
    COCKPIT_SEAT_HEIGHT_MAX = 1.6  # Maximum seated passenger height (1.6m)
    
    # Distance and density thresholds
    DISTANCE_PERCENTILE_THRESHOLD = 95    # Keep 95% of points (remove outliers)
    INTERIOR_DENSITY_PERCENTILE = 50      # Percentile for density-based filtering
    FALLBACK_DENSITY_PERCENTILE = 65     # Higher percentile for fallback method
    FINAL_FALLBACK_PERCENTILE = 70       # Final fallback percentile
    
    # Nearest neighbors parameters
    DEFAULT_NEIGHBORS = 10      # Default number of neighbors for density calculation
    MIN_NEIGHBORS = 8          # Minimum neighbors for cockpit height filtered
    MAX_NEIGHBORS = 12         # Maximum neighbors for fallback method
    NEIGHBOR_RADIUS = 0.3      # Radius for neighbor search (30cm)
    FALLBACK_RADIUS = 0.4      # Larger radius for fallback (40cm)


class VTKConfig:
    """Configuration for VTK visualization and rendering."""
    
    # Window sizes
    DEFAULT_WINDOW_SIZE = (1600, 1200)
    SMALL_WINDOW_SIZE = (800, 600)
    LARGE_WINDOW_SIZE = (1920, 1080)
    
    # Point rendering
    DEFAULT_POINT_SIZE = 2.0
    LARGE_POINT_SIZE = 6.0
    INTERIOR_POINT_SIZE = 8.0
    GROUND_POINT_SIZE = 1.0
    
    # Colors (RGB tuples)
    COLORS = {
        'vehicle': 'lightgreen',
        'interior': 'red',
        'ground': 'gray',
        'selected': (0, 255, 0),    # Bright green
        'unselected': (128, 128, 128),  # Gray
        'cube_widget': 'red',
        'background': 'black'
    }
    
    # Opacity settings
    VEHICLE_OPACITY = 0.6
    INTERIOR_OPACITY = 1.0
    GROUND_OPACITY = 0.2
    OCCUPANCY_OPACITY = 0.4
    INTERIOR_GRID_OPACITY = 0.7
    DISTANCE_MAP_OPACITY = 0.5
    
    # VTK environment settings
    VTK_ENV_VARS = {
        'VTK_RENDER_WINDOW_MAIN_THREAD': '1',
        'VTK_USE_COCOA': '1',  # Will be set to '0' on non-macOS
        'VTK_SILENCE_GET_VOID_POINTER_WARNINGS': '1',
        'VTK_DEBUG_LEAKS': '0',
        'VTK_AUTO_INIT': '1',
        'VTK_RENDERING_BACKEND': 'OpenGL2',
        'PYVISTA_OFF_SCREEN': '0',
        'PYVISTA_USE_PANEL': '0',
    }
    
    # Retry settings
    MAX_RETRIES = 3
    RETRY_DELAY = 0.01  # 10ms delay between retries


class FileConfig:
    """Configuration for file handling and I/O operations."""
    
    # Supported file extensions
    SUPPORTED_EXTENSIONS = {
        'e57': ['.e57'],
        'pcd': ['.pcd'],
        'ros_bag': ['.bag'],
        'numpy': ['.npy'],
        'json': ['.json']
    }
    
    # Default directories
    DEFAULT_OUTPUT_DIR = "output"
    DEFAULT_LOG_DIR = "logs"
    
    # Default filenames
    DEFAULT_GROUND_POINTS = "ground_points.npy"
    DEFAULT_NON_GROUND_POINTS = "non_ground_points.npy"
    DEFAULT_VEHICLE_POINTS = "vehicle_points.npy"
    DEFAULT_INTERIOR_POINTS = "interior_points.npy"
    DEFAULT_INTERIOR_COCKPIT = "interior_cockpit.npy"
    DEFAULT_HUMAN_MODEL = "interior_cockpit_with_human.npy"
    
    # File size limits (in MB)
    MAX_FILE_SIZE = 500  # 500MB limit for safety
    WARN_FILE_SIZE = 100  # Warn if file is larger than 100MB
    
    # Stride parameters for large files
    LARGE_FILE_THRESHOLD = 1000000  # 1M points
    AUTO_STRIDE_FACTOR = 10         # Automatic stride for large files


class GUIConfig:
    """Configuration for GUI application."""
    
    # Window geometry
    MAIN_WINDOW_SIZE = "1000x700"
    DIALOG_WINDOW_SIZE = (500, 400)
    
    # Threading and UI update intervals
    MESSAGE_QUEUE_CHECK_INTERVAL = 100  # milliseconds
    PROGRESS_UPDATE_INTERVAL = 50       # milliseconds
    
    # File dialog settings
    MAX_RECENT_FILES = 10
    
    # Cube selection parameters
    DEFAULT_CUBE_WINDOW_WIDTH = 1600
    DEFAULT_CUBE_WINDOW_HEIGHT = 1200
    CUBE_SIZE_INCREMENT = 100
    
    # Cube bounds defaults (relative to point cloud center)
    DEFAULT_CUBE_X_SIZE = 4.0  # 2m in each direction
    DEFAULT_CUBE_Y_SIZE = 4.0
    DEFAULT_CUBE_Z_SIZE = 2.0
    
    # Human model parameters
    DEFAULT_HUMAN_SCALE = 1.0
    MIN_HUMAN_SCALE = 0.5
    MAX_HUMAN_SCALE = 2.0
    HUMAN_SCALE_INCREMENT = 0.1
    
    DEFAULT_HUMAN_ROTATION = 0.0
    HUMAN_ROTATION_INCREMENT = 15.0  # degrees


class HumanModelConfig:
    """Configuration for human model generation."""
    
    # Base point density
    BASE_DENSITY_MULTIPLIER = 100
    
    # Human body proportions (in meters)
    PROPORTIONS = {
        # Head
        'head_height_offset': 0.65,
        'head_radii': [0.10, 0.12, 0.11],
        
        # Neck
        'neck_height_offset': 0.55,
        'neck_radius': 0.06,
        'neck_height': 0.08,
        
        # Torso
        'torso_height_offset': 0.35,
        'torso_radii': [0.18, 0.12, 0.25],
        
        # Shoulders
        'shoulder_width': 0.20,
        'shoulder_height_offset': 0.50,
        'shoulder_radius': 0.08,
        
        # Arms
        'arm_length': 0.30,
        'upper_arm_radius': 0.05,
        'forearm_radius': 0.04,
        'forearm_length': 0.25,
        'hand_radii': [0.04, 0.08, 0.02],
        
        # Lower body
        'hip_radii': [0.15, 0.20, 0.10],
        'thigh_length': 0.35,
        'thigh_radius': 0.08,
        'knee_radius': 0.06,
        'lower_leg_height': 0.30,
        'lower_leg_radius': 0.05,
        'foot_radii': [0.05, 0.12, 0.04],
    }
    
    # Positioning offsets
    POSITIONING = {
        'seat_offset_x_ratio': 0.3,   # 30% from front
        'seat_offset_y_ratio': 0.4,   # 40% from left (driver side)
        'seat_offset_z': -0.1,        # Slightly below roof
    }
    
    # Noise parameters
    DEFAULT_NOISE_LEVEL = 0.005
    MAX_NOISE_LEVEL = 0.02


class LoggingConfig:
    """Configuration for logging system."""
    
    # Log levels
    DEFAULT_LEVEL = "INFO"
    CONSOLE_LEVEL = "INFO"
    FILE_LEVEL = "DEBUG"
    
    # Log format
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    
    # File settings
    MAX_LOG_SIZE = 10 * 1024 * 1024  # 10MB
    BACKUP_COUNT = 5


def get_config_dict() -> Dict[str, Any]:
    """
    Get all configuration as a dictionary for easy access.
    
    Returns:
        Dictionary containing all configuration values
    """
    return {
        'analysis': {attr: getattr(AnalysisConfig, attr) 
                    for attr in dir(AnalysisConfig) if not attr.startswith('_')},
        'vtk': {attr: getattr(VTKConfig, attr) 
               for attr in dir(VTKConfig) if not attr.startswith('_')},
        'file': {attr: getattr(FileConfig, attr) 
                for attr in dir(FileConfig) if not attr.startswith('_')},
        'gui': {attr: getattr(GUIConfig, attr) 
               for attr in dir(GUIConfig) if not attr.startswith('_')},
        'human_model': {attr: getattr(HumanModelConfig, attr) 
                       for attr in dir(HumanModelConfig) if not attr.startswith('_')},
        'logging': {attr: getattr(LoggingConfig, attr) 
                   for attr in dir(LoggingConfig) if not attr.startswith('_')},
    }


def validate_config() -> bool:
    """
    Validate configuration values for consistency.
    
    Returns:
        True if configuration is valid, False otherwise
    """
    # Check that thresholds are reasonable
    if AnalysisConfig.GROUND_HEIGHT_THRESHOLD <= 0:
        return False
    
    if AnalysisConfig.DBSCAN_EPS <= 0 or AnalysisConfig.DBSCAN_MIN_SAMPLES <= 0:
        return False
    
    if VTKConfig.DEFAULT_WINDOW_SIZE[0] <= 0 or VTKConfig.DEFAULT_WINDOW_SIZE[1] <= 0:
        return False
    
    # Check that file size limits are reasonable
    if FileConfig.MAX_FILE_SIZE <= 0 or FileConfig.WARN_FILE_SIZE <= 0:
        return False
    
    return True


if __name__ == "__main__":
    print("LIDAR Configuration Module")
    print("=" * 40)
    
    # Validate configuration
    if validate_config():
        print("✅ Configuration validation passed")
    else:
        print("❌ Configuration validation failed")
    
    # Show some key values
    print(f"DBSCAN eps: {AnalysisConfig.DBSCAN_EPS}")
    print(f"Grid resolution: {AnalysisConfig.GRID_RESOLUTION}")
    print(f"Default window size: {VTKConfig.DEFAULT_WINDOW_SIZE}")
    print(f"Default output directory: {FileConfig.DEFAULT_OUTPUT_DIR}")
    
    print("\nConfiguration module loaded successfully.")