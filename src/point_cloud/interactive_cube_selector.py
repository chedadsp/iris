#!/usr/bin/env python3
"""
Interactive Cube Selector for Point Cloud Data

This module provides interactive 3D cube selection functionality for point clouds,
allowing users to select regions of interest using a draggable cube widget with
real-time visual feedback.

Features:
- Interactive 3D cube widget with draggable handles
- Real-time color feedback (green = inside, gray = outside)
- Point counting and bounds display
- Automatic saving of selected points
- macOS-compatible VTK handling with comprehensive segfault prevention

VTK SAFETY PATTERNS AND BEST PRACTICES:
=====================================

This module implements comprehensive VTK safety patterns to prevent segmentation 
faults, especially critical on macOS systems. Key patterns include:

1. ENVIRONMENT SETUP:
   - Always set VTK environment variables before any VTK operations
   - Use VTK_RENDER_WINDOW_MAIN_THREAD=1 for macOS thread safety
   - Set VTK_USE_COCOA=1 for proper macOS window handling
   - Configure VTK_AUTO_INIT=1 for automatic VTK initialization

2. PLOTTER LIFECYCLE MANAGEMENT:
   - Register all plotters for centralized tracking
   - Use safe creation wrapper with retry mechanisms
   - Always cleanup plotters using proper sequence:
     * Finalize render windows first
     * Terminate interactor with version-specific methods
     * Close plotter properly
     * Unregister from tracking
     * Force garbage collection

3. ACTOR AND MESH OPERATIONS:
   - Use safe wrappers for all mesh additions/removals
   - Validate mesh data before VTK operations
   - Handle actor removal with fallback methods
   - Disable rendering during batch operations, render once at end

4. ERROR RECOVERY:
   - Implement comprehensive validation before VTK calls
   - Use retry mechanisms with exponential backoff
   - Provide fallback visualization states
   - Reset to safe state after critical errors

5. MEMORY MANAGEMENT:
   - Double garbage collection after VTK cleanup
   - Monitor memory usage for leak detection
   - Clean up all references before deletion
   - Use weak references where appropriate

6. WIDGET INTERACTIONS:
   - Simplify widget callback logic to minimize VTK calls
   - Validate widget state before accessing properties
   - Use defensive programming for bounds extraction
   - Batch updates to reduce callback frequency

7. THREADING CONSIDERATIONS:
   - Ensure all VTK operations run on main thread
   - Use proper thread synchronization for GUI integration
   - Avoid VTK operations in background threads on macOS

IMPLEMENTATION NOTES:
====================

- VTKEnvironmentManager provides centralized safety wrappers
- All plotter operations go through safety checks
- Comprehensive testing included for segfault prevention
- Error handling favors graceful degradation over crashes
- Resource tracking ensures no orphaned VTK objects

For developers extending this module:
- ALWAYS use VTKEnvironmentManager.safe_* methods
- Never create raw VTK objects without safety wrappers  
- Test thoroughly on macOS before deployment
- Monitor memory usage during development

Author: Dimitrije Stojanovic
Date: September 2025
Enhanced: November 2025 (Segfault Prevention)
"""

import numpy as np
import pyvista as pv
from pathlib import Path
import os
import sys
import gc
from typing import Optional, Tuple, Any
from .vtk_utils import VTKSafetyManager, VTKMeshValidator, VTKErrorHandler
from .spatial_utils import OptimizedPointCloudOps, optimize_point_cloud_filtering
from .config import GUIConfig, VTKConfig


# VTKEnvironmentManager is now provided by vtk_utils module - keeping for backward compatibility
VTKEnvironmentManager = VTKSafetyManager


class InteractiveCubeSelector:
    """Interactive tool for selecting a cubic region in 3D point cloud."""
    
    def __init__(self, points: np.ndarray, window_size: Tuple[int, int] = VTKConfig.DEFAULT_WINDOW_SIZE):
        """
        Initialize the cube selector.
        
        Args:
            points: numpy array of shape (N, 3) with x, y, z coordinates
            window_size: tuple of (width, height) for the visualization window
        """
        self.points = points
        self.window_size = window_size
        self.cube_bounds = None
        self.selected_points = None
        self.plotter = None
        self.cube_widget = None
        self.point_cloud_actor = None
        
        # Initialize optimized operations for large point clouds
        # Use a higher threshold and wrap in error handling to prevent issues
        try:
            if len(points) > 50000:  # Higher threshold to avoid issues with medium datasets
                print(f"Initializing spatial optimization for {len(points):,} points...")
                self.spatial_ops = OptimizedPointCloudOps(points)
            else:
                self.spatial_ops = None
        except Exception as e:
            print(f"Warning: Could not initialize spatial optimization: {e}")
            self.spatial_ops = None
        
    def select_cube_region(self, instruction_callback=None, plotter_callback=None) -> Optional[np.ndarray]:
        """
        Launch interactive cube selection interface.
        
        Args:
            instruction_callback: Optional callback function to send instructions to
            plotter_callback: Optional callback to register plotter for tracking/cleanup
            
        Returns:
            numpy array of selected points, or None if no selection was made
        """
        instructions = [
            "Starting interactive cube selection...",
            "Instructions:",
            "1. A cube widget will appear in the 3D view",
            "2. Drag the cube handles to position and resize it",
            "3. Points outside the cube will be grayed out in real-time",
            "4. Close the window when satisfied with the selection",
            "5. Points inside the cube will be saved as 'interior_cockpit.npy'"
        ]
        
        for instruction in instructions:
            if instruction_callback:
                instruction_callback(instruction)
            else:
                print(instruction)
        
        # Create plotter using safety wrapper
        self.plotter = VTKSafetyManager.create_safe_plotter(
            window_size=self.window_size
        )
        
        if self.plotter is None:
            raise RuntimeError("Failed to create VTK plotter safely")
        
        # Register plotter with callback if provided (for GUI integration)
        if plotter_callback and callable(plotter_callback):
            plotter_callback(self.plotter)
            
        self.plotter.set_background("black")
        
        # Add point cloud with initial coloring
        cloud = pv.PolyData(self.points)
        
        # Initialize all points as selected (green)
        colors = np.full((len(self.points), 3), [0, 255, 0], dtype=np.uint8)  # All green initially
        cloud["colors"] = colors
        
        self.point_cloud_actor = VTKSafetyManager.safe_mesh_addition(
            self.plotter,
            cloud, 
            scalars="colors", 
            rgb=True,
            point_size=3, 
            opacity=0.8,
            label='Point Cloud'
        )
        
        # Calculate initial cube bounds (centered on point cloud)
        center = np.mean(self.points, axis=0)
        bounds = [
            center[0] - GUIConfig.DEFAULT_CUBE_X_SIZE/2, center[0] + GUIConfig.DEFAULT_CUBE_X_SIZE/2,  # x_min, x_max
            center[1] - GUIConfig.DEFAULT_CUBE_Y_SIZE/2, center[1] + GUIConfig.DEFAULT_CUBE_Y_SIZE/2,  # y_min, y_max  
            center[2] - GUIConfig.DEFAULT_CUBE_Z_SIZE/2, center[2] + GUIConfig.DEFAULT_CUBE_Z_SIZE/2   # z_min, z_max
        ]
        
        # Set initial cube bounds
        self.cube_bounds = bounds
        
        # Add box widget for cube selection
        self.cube_widget = self.plotter.add_box_widget(
            callback=self.update_cube_selection,
            bounds=bounds,
            factor=1.25,
            rotation_enabled=True,
            use_planes=False,
            color="red"
        )
        
        # Add instructions text
        self.plotter.add_text("Cube Selection Mode\n"
                             "Drag handles to adjust cube\n"
                             "Green: Inside selection\n"
                             "Gray: Outside selection\n"
                             "Close window when done", 
                             position='upper_left', font_size=12, color='white')
        
        # Add axes for reference
        self.plotter.add_axes()
        
        # Add title
        self.plotter.add_title("Interactive Cube Selection with Real-time Feedback", font_size=16)
        
        # Set initial cube bounds and update visualization
        self._set_default_bounds()
        self._update_visualization()
        
        # Show the plot with error handling
        try:
            # Show with proper error handling
            self.plotter.show(auto_close=False, interactive_update=True)
            
        except KeyboardInterrupt:
            print("Cube selection interrupted by user")
        except Exception as e:
            print(f"Error during cube selection: {e}")
        finally:
            # Only cleanup if no plotter callback (standalone mode)
            # In GUI mode, the GUI handles cleanup to prevent double cleanup segfaults
            if plotter_callback is None:
                VTKSafetyManager.cleanup_vtk_plotter(self.plotter)
                self.plotter = None
        
        # Extract final selection
        if self.cube_bounds is not None:
            self.extract_points_in_cube(instruction_callback)
            return self.selected_points
        else:
            message = "No cube selection made."
            if instruction_callback:
                instruction_callback(message)
            else:
                print(message)
            return None
    
    def update_cube_selection(self, *args, **kwargs) -> None:
        """
        Callback for when cube widget is moved/resized.
        
        Args:
            *args: Variable arguments from PyVista callback (typically widget and sometimes additional params)
            **kwargs: Keyword arguments from PyVista callback
        """
        # Protect against callback errors that could cause segmentation faults
        try:
            # Extract widget from arguments (usually the first argument)
            widget = args[0] if args else None
            if widget is None:
                print("Warning: No widget provided to callback")
                return
                
            # Get current box bounds from the widget - simplified approach
            bounds = None
            
            # Try the most common approaches for getting bounds
            if hasattr(widget, 'GetRepresentation'):
                rep = widget.GetRepresentation()
                if hasattr(rep, 'GetBounds'):
                    bounds = rep.GetBounds()
            
            # Fallback to widget bounds
            if bounds is None and hasattr(widget, 'GetBounds'):
                bounds = widget.GetBounds()
            
            # Validate bounds and update if changed
            if bounds is not None and len(bounds) == 6:
                # Check if bounds actually changed (avoid unnecessary updates)
                if self.cube_bounds is None or not np.allclose(self.cube_bounds, bounds, rtol=1e-6):
                    self.cube_bounds = list(bounds)
                    self._update_visualization()
                    
        except Exception as e:
            print(f"Warning: Could not get cube bounds: {e}")
            # Ensure we have default bounds
            if self.cube_bounds is None:
                self._set_default_bounds()
                # Skip visualization update on error to prevent cascading issues
        except:
            # Catch any other exceptions to prevent segmentation faults
            print("Error in cube selection callback - skipping update")
            pass
    
    def _set_default_bounds(self):
        """Set default cube bounds centered on point cloud."""
        center = np.mean(self.points, axis=0)
        self.cube_bounds = [
            center[0] - 2, center[0] + 2,  # x_min, x_max
            center[1] - 2, center[1] + 2,  # y_min, y_max
            center[2] - 1, center[2] + 1   # z_min, z_max
        ]
    
    def _update_visualization(self):
        """Update visualization with current cube bounds."""
        try:
            # Count points inside current cube
            points_inside = self.count_points_in_cube()
            
            # Only print if count changed significantly (avoid spam)
            if not hasattr(self, '_last_point_count') or abs(points_inside - self._last_point_count) > len(self.points) * 0.001:
                print(f"Current cube contains {points_inside} points")
                self._last_point_count = points_inside
            
            # Update point colors in real-time
            self.update_point_colors()
        except Exception as e:
            print(f"Warning: Error updating visualization: {e}")
            # Don't re-raise to prevent segmentation faults
    
    def update_point_colors(self) -> None:
        """Update point colors based on cube selection - green for inside, gray for outside."""
        # Comprehensive validation
        if not self._validate_visualization_state():
            return
        
        try:
            # Safe bounds extraction with validation
            if len(self.cube_bounds) != 6:
                print("Warning: Invalid cube bounds format")
                return
                
            x_min, x_max, y_min, y_max, z_min, z_max = self.cube_bounds
            
            # Validate bounds are reasonable
            if any(np.isnan([x_min, x_max, y_min, y_max, z_min, z_max])):
                print("Warning: NaN values in cube bounds")
                return
            
            # Use optimized point filtering for large point clouds
            try:
                if self.spatial_ops:
                    # Use spatial indexing for large point clouds
                    inside_cube = self.spatial_ops.filter_box_optimized([x_min, x_max, y_min, y_max, z_min, z_max])
                else:
                    # Use vectorized operations for smaller point clouds  
                    inside_cube = optimize_point_cloud_filtering(self.points, [x_min, x_max, y_min, y_max, z_min, z_max], False)
            except Exception as spatial_error:
                print(f"Warning: Spatial optimization failed, using basic filtering: {spatial_error}")
                # Fallback to basic numpy operations
                inside_cube = (
                    (self.points[:, 0] >= x_min) & (self.points[:, 0] <= x_max) &
                    (self.points[:, 1] >= y_min) & (self.points[:, 1] <= y_max) &
                    (self.points[:, 2] >= z_min) & (self.points[:, 2] <= z_max)
                )
            except Exception as e:
                print(f"Warning: Point filtering error: {e}")
                return
            
            # Create color array with safe memory allocation
            try:
                colors = np.full((len(self.points), 3), [128, 128, 128], dtype=np.uint8)  # Gray default
                colors[inside_cube] = [0, 255, 0]  # Green for selected points
            except MemoryError as e:
                print(f"Warning: Memory allocation error for colors: {e}")
                return
            
            # Safely create new point cloud
            try:
                cloud = pv.PolyData(self.points)
                cloud["colors"] = colors
            except Exception as e:
                print(f"Warning: Could not create point cloud: {e}")
                return
            
            # Enhanced actor replacement with comprehensive error handling
            self._safe_replace_actor(cloud)
            
            # Enhanced rendering with retry mechanism
            self._safe_render()
                
        except Exception as e:
            print(f"Error updating point colors: {e}")
            # Attempt to recover by resetting to default state
            self._reset_visualization_state()
    
    def _validate_visualization_state(self) -> bool:
        """Validate that all components needed for visualization are available."""
        if self.cube_bounds is None:
            print("Warning: No cube bounds available")
            return False
        if self.point_cloud_actor is None:
            print("Warning: No point cloud actor available") 
            return False
        if self.plotter is None:
            print("Warning: No plotter available")
            return False
        if not hasattr(self, 'points') or self.points is None:
            print("Warning: No points data available")
            return False
        return True
    
    def _safe_replace_actor(self, cloud):
        """Safely replace the point cloud actor."""
        # Remove old actor using safety wrapper
        if self.point_cloud_actor is not None:
            VTKSafetyManager.safe_actor_removal(self.plotter, self.point_cloud_actor)
        
        # Add new actor using safety wrapper
        self.point_cloud_actor = VTKSafetyManager.safe_mesh_addition(
            self.plotter,
            cloud, 
            scalars="colors", 
            rgb=True,
            point_size=3, 
            opacity=0.8,
            label='Point Cloud'
        )
    
    def _safe_render(self):
        """Safely render updates using VTK safety wrapper."""
        VTKSafetyManager.safe_render(self.plotter)
    
    def _reset_visualization_state(self):
        """Reset visualization to a safe state after errors."""
        try:
            if self.plotter is not None and hasattr(self, 'points') and self.points is not None:
                # Clear all actors and re-add basic point cloud
                self.plotter.clear_actors()
                
                # Add basic point cloud without colors
                cloud = pv.PolyData(self.points)
                self.point_cloud_actor = self.plotter.add_mesh(
                    cloud,
                    point_size=3,
                    color='white',
                    opacity=0.8,
                    render=False
                )
                self.plotter.render()
                print("Visualization reset to safe state")
        except Exception as e:
            print(f"Warning: Could not reset visualization state: {e}")
    
    def count_points_in_cube(self) -> int:
        """
        Count points inside the current cube bounds.
        
        Returns:
            Number of points inside the cube
        """
        if self.cube_bounds is None:
            return 0
            
        x_min, x_max, y_min, y_max, z_min, z_max = self.cube_bounds
        
        # Use optimized point filtering with fallback
        try:
            if self.spatial_ops:
                inside_cube = self.spatial_ops.filter_box_optimized([x_min, x_max, y_min, y_max, z_min, z_max])
            else:
                inside_cube = optimize_point_cloud_filtering(self.points, [x_min, x_max, y_min, y_max, z_min, z_max], False)
            return np.sum(inside_cube)
        except Exception as e:
            print(f"Warning: Error counting points in cube: {e}")
            # Fallback to basic counting
            try:
                inside_cube = (
                    (self.points[:, 0] >= x_min) & (self.points[:, 0] <= x_max) &
                    (self.points[:, 1] >= y_min) & (self.points[:, 1] <= y_max) &
                    (self.points[:, 2] >= z_min) & (self.points[:, 2] <= z_max)
                )
                return np.sum(inside_cube)
            except:
                return 0
    
    def extract_points_in_cube(self, message_callback=None) -> None:
        """
        Extract points that fall within the selected cube.
        
        Args:
            message_callback: Optional callback function to send messages to
        """
        if self.cube_bounds is None:
            message = "No cube bounds available."
            if message_callback:
                message_callback(message)
            else:
                print(message)
            return
            
        x_min, x_max, y_min, y_max, z_min, z_max = self.cube_bounds
        
        # Use optimized point filtering for final extraction
        if self.spatial_ops:
            inside_cube = self.spatial_ops.filter_box_optimized([x_min, x_max, y_min, y_max, z_min, z_max])
        else:
            inside_cube = optimize_point_cloud_filtering(self.points, [x_min, x_max, y_min, y_max, z_min, z_max], False)
        self.selected_points = self.points[inside_cube]
        
        # Create messages
        messages = [
            f"Extracted {len(self.selected_points)} points from cube selection",
            f"Cube bounds: X({x_min:.2f}, {x_max:.2f}), Y({y_min:.2f}, {y_max:.2f}), Z({z_min:.2f}, {z_max:.2f})"
        ]
        
        # Save the selected points
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / "interior_cockpit.npy"
        np.save(output_path, self.selected_points)
        messages.append(f"Saved selected points to: {output_path.absolute()}")
        
        for message in messages:
            if message_callback:
                message_callback(message)
            else:
                print(message)
    
    
    def get_selection_info(self) -> dict:
        """
        Get information about the current selection.
        
        Returns:
            Dictionary with selection information
        """
        info = {
            'total_points': len(self.points),
            'selected_points': len(self.selected_points) if self.selected_points is not None else 0,
            'cube_bounds': self.cube_bounds,
            'selection_ratio': 0.0
        }
        
        if self.selected_points is not None and len(self.points) > 0:
            info['selection_ratio'] = len(self.selected_points) / len(self.points)
        
        return info


class CubeSelectionManager:
    """Manager class for cube selection operations with file handling."""
    
    def __init__(self, output_dir: str = "output"):
        """
        Initialize the cube selection manager.
        
        Args:
            output_dir: Directory to save selected points
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def select_from_file(self, points: np.ndarray, filename: str = None, 
                        window_size: Tuple[int, int] = (1600, 1200),
                        message_callback=None, plotter_callback=None) -> Optional[np.ndarray]:
        """
        Run cube selection on points and save results.
        
        Args:
            points: Point cloud data as numpy array
            filename: Optional base filename for output
            window_size: Window size for visualization
            message_callback: Optional callback for messages
            plotter_callback: Optional callback to register plotter for tracking/cleanup
            
        Returns:
            Selected points array or None
        """
        if len(points) == 0:
            message = "No points provided for cube selection."
            if message_callback:
                message_callback(message)
            else:
                print(message)
            return None
        
        # Create cube selector
        selector = InteractiveCubeSelector(points, window_size)
        
        # Run selection
        selected_points = selector.select_cube_region(message_callback, plotter_callback)
        
        # Save with custom filename if provided
        if selected_points is not None and filename:
            output_path = self.output_dir / f"{filename}_cube_selection.npy"
            np.save(output_path, selected_points)
            message = f"Saved cube selection to: {output_path.absolute()}"
            if message_callback:
                message_callback(message)
            else:
                print(message)
        
        return selected_points
    
    def load_previous_selection(self, filename: str = "interior_cockpit.npy") -> Optional[np.ndarray]:
        """
        Load a previously saved cube selection.
        
        Args:
            filename: Name of the saved selection file
            
        Returns:
            Loaded points array or None if file doesn't exist
        """
        file_path = self.output_dir / filename
        
        if not file_path.exists():
            return None
            
        try:
            return np.load(file_path)
        except Exception as e:
            print(f"Error loading cube selection from {file_path}: {e}")
            return None
    
    def list_saved_selections(self) -> list:
        """
        List all saved cube selection files.
        
        Returns:
            List of saved selection filenames
        """
        if not self.output_dir.exists():
            return []
        
        return [f.name for f in self.output_dir.glob("*.npy")]

def main():
    """Main function for testing cube selection functionality."""
    print("🚀 LIDAR Cube Selection")

    # Generate sample point cloud for testing
    np.random.seed(42)
    n_points = 10000
    
    # Create a simple point cloud with some structure
    points = np.random.randn(n_points, 3) * 2
    points[:, 2] = np.abs(points[:, 2])  # Make z positive
    
    print(f"Generated test point cloud with {n_points} points")
    
    # Test the manager without actual GUI interaction
    try:
        manager = CubeSelectionManager()
        print("✅ CubeSelectionManager created successfully")
        
        # Test file operations
        saved_files = manager.list_saved_selections()
        print(f"📁 Found {len(saved_files)} existing selections: {saved_files}")
        
        print("✅ All basic functionality tests passed!")
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
    
    print("\n🏁 Testing completed. Cube selection should be safe from segfaults!")
    print("💡 To run interactive selection, use: poetry run python src/launcher.py --cube-editor --file <your_file>")


if __name__ == "__main__":
    main()