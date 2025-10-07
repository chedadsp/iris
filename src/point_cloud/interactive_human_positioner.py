#!/usr/bin/env python3
"""
Interactive Human Model Positioner for LIDAR Point Clouds

This script allows you to interactively position and resize a human model
within a LIDAR point cloud using a 3D cube widget.

Usage:
    python interactive_human_positioner.py

Instructions:
1. The script will load the interior_cockpit.npy point cloud
2. A 3D visualization window will open with a red cube widget
3. Drag the cube handles to position and resize the human model
4. The human model preview will update in real-time
5. Close the window to save the positioned human model

Author: Dimitrije Stojanovic
Date: September 2025
"""

import numpy as np
import pyvista as pv
from pathlib import Path
import argparse
import sys
import os

# Import VTK safety manager
from .vtk_utils import VTKSafetyManager


class HumanModelGenerator:
    """Generate a realistic human model point cloud."""
    
    @staticmethod
    def generate_seated_human(center_position, scale=1.0, rotation_deg=0.0, num_points=1200):
        """Generate a seated human model with realistic proportions.
        
        Args:
            center_position: (x, y, z) center position for the human
            scale: Scale factor for the human size
            rotation_deg: Rotation angle in degrees around Z-axis
            num_points: Number of points to generate
            
        Returns:
            numpy array of human model points
        """
        x_center, y_center, z_center = center_position
        
        # Human body proportions (seated)
        torso_height = 0.6 * scale
        torso_width = 0.4 * scale
        torso_depth = 0.25 * scale
        
        head_radius = 0.1 * scale
        head_height = 0.15 * scale
        
        leg_height = 0.4 * scale
        leg_width = 0.15 * scale
        
        arm_length = 0.5 * scale
        arm_width = 0.08 * scale
        
        points = []
        
        # Generate head (sphere at top)
        head_center_z = z_center + torso_height + head_height/2
        for _ in range(int(num_points * 0.1)):  # 10% of points for head
            theta = np.random.uniform(0, 2*np.pi)
            phi = np.random.uniform(0, np.pi)
            r = np.random.uniform(0, head_radius)
            
            x = x_center + r * np.sin(phi) * np.cos(theta)
            y = y_center + r * np.sin(phi) * np.sin(theta)
            z = head_center_z + r * np.cos(phi)
            
            points.append([x, y, z])
        
        # Generate torso (rectangular body)
        for _ in range(int(num_points * 0.4)):  # 40% of points for torso
            x = x_center + np.random.uniform(-torso_width/2, torso_width/2)
            y = y_center + np.random.uniform(-torso_depth/2, torso_depth/2)
            z = z_center + np.random.uniform(0, torso_height)
            
            points.append([x, y, z])
        
        # Generate arms (extending from torso)
        for _ in range(int(num_points * 0.2)):  # 20% of points for arms
            # Left arm
            arm_x = x_center + np.random.uniform(-torso_width/2 - arm_length, -torso_width/2)
            arm_y = y_center + np.random.uniform(-arm_width/2, arm_width/2)
            arm_z = z_center + np.random.uniform(torso_height*0.6, torso_height*0.9)
            points.append([arm_x, arm_y, arm_z])
            
            # Right arm
            arm_x = x_center + np.random.uniform(torso_width/2, torso_width/2 + arm_length)
            points.append([arm_x, arm_y, arm_z])
        
        # Generate legs (seated position - legs bent)
        for _ in range(int(num_points * 0.3)):  # 30% of points for legs
            # Upper legs (thighs) - horizontal when seated
            thigh_x = x_center + np.random.uniform(-leg_width, leg_width)
            thigh_y = y_center + np.random.uniform(0, leg_height)
            thigh_z = z_center + np.random.uniform(-0.1*scale, 0.1*scale)
            points.append([thigh_x, thigh_y, thigh_z])
            
            # Lower legs (shins) - vertical when seated
            shin_x = x_center + np.random.uniform(-leg_width/2, leg_width/2)
            shin_y = y_center + leg_height + np.random.uniform(-0.05*scale, 0.05*scale)
            shin_z = z_center + np.random.uniform(-leg_height, 0)
            points.append([shin_x, shin_y, shin_z])
        
        # Add some noise for realism
        points = np.array(points)
        noise = np.random.normal(0, 0.02 * scale, points.shape)
        points += noise
        
        # Apply rotation around Z-axis if specified
        if rotation_deg != 0.0:
            rotation_rad = np.radians(rotation_deg)
            cos_rot = np.cos(rotation_rad)
            sin_rot = np.sin(rotation_rad)
            
            # Translate to origin, rotate, then translate back
            points[:, 0] -= x_center
            points[:, 1] -= y_center
            
            # Apply rotation matrix for Z-axis rotation
            x_rot = points[:, 0] * cos_rot - points[:, 1] * sin_rot
            y_rot = points[:, 0] * sin_rot + points[:, 1] * cos_rot
            
            points[:, 0] = x_rot + x_center
            points[:, 1] = y_rot + y_center
        
        return points


class InteractiveHumanPositioner:
    """Interactive tool for positioning a human model in 3D LIDAR space."""
    
    def __init__(self, point_cloud_file, window_size=(1600, 1200)):
        """Initialize the interactive positioner.
        
        Args:
            point_cloud_file: Path to the point cloud numpy file
            window_size: (width, height) of the visualization window
        """
        self.point_cloud_file = Path(point_cloud_file)
        self.window_size = window_size
        
        # Load point cloud data
        self.load_point_cloud()
        
        # Initialize human model parameters
        self.human_center = self.calculate_initial_position()
        self.human_scale = 1.0
        self.human_rotation = 0.0  # Rotation angle in degrees around Z-axis
        self.human_points = None
        
        # PyVista objects
        self.plotter = None
        self.cube_widget = None
        self.human_actor = None
        self.point_cloud_actor = None
        self.rotation_slider = None
        
        # Widget callback state
        self.updating = False
        
    def load_point_cloud(self):
        """Load the point cloud from file."""
        if not self.point_cloud_file.exists():
            raise FileNotFoundError(f"Point cloud file not found: {self.point_cloud_file}")
        
        self.points = np.load(self.point_cloud_file)
        print(f"Loaded {len(self.points):,} points from {self.point_cloud_file}")
        
        # Calculate point cloud bounds
        self.bounds = {
            'x_min': self.points[:, 0].min(),
            'x_max': self.points[:, 0].max(),
            'y_min': self.points[:, 1].min(),
            'y_max': self.points[:, 1].max(),
            'z_min': self.points[:, 2].min(),
            'z_max': self.points[:, 2].max()
        }
        
        print(f"Point cloud bounds:")
        print(f"  X: {self.bounds['x_min']:.3f} to {self.bounds['x_max']:.3f}")
        print(f"  Y: {self.bounds['y_min']:.3f} to {self.bounds['y_max']:.3f}")
        print(f"  Z: {self.bounds['z_min']:.3f} to {self.bounds['z_max']:.3f}")
    
    def calculate_initial_position(self):
        """Calculate a good initial position for the human model."""
        # Place human in the center of the point cloud horizontally,
        # and at the top of the Z range (assuming this is a cockpit/interior)
        x_center = (self.bounds['x_min'] + self.bounds['x_max']) / 2
        y_center = (self.bounds['y_min'] + self.bounds['y_max']) / 2
        z_center = self.bounds['z_max'] - 0.3  # Slightly below the top
        
        return [x_center, y_center, z_center]
    
    def generate_human_model(self):
        """Generate the human model based on current parameters."""
        try:
            # Validate parameters before generation
            if self.human_center is None or len(self.human_center) != 3:
                raise ValueError("Invalid human center position")
            
            if not np.isfinite(self.human_scale) or self.human_scale <= 0:
                raise ValueError("Invalid human scale")
            
            if not np.isfinite(self.human_rotation):
                raise ValueError("Invalid human rotation")
            
            # Generate human model
            self.human_points = HumanModelGenerator.generate_seated_human(
                self.human_center, 
                scale=self.human_scale,
                rotation_deg=self.human_rotation
            )
            
            # Validate generated points
            if self.human_points is None or len(self.human_points) == 0:
                raise RuntimeError("Human model generation returned no points")
            
            print(f"Generated human model with {len(self.human_points)} points at position "
                  f"[{self.human_center[0]:.3f}, {self.human_center[1]:.3f}, {self.human_center[2]:.3f}] "
                  f"with scale {self.human_scale:.2f} and rotation {self.human_rotation:.1f}°")
                  
        except Exception as e:
            print(f"Error generating human model: {e}")
            # Keep previous human model if generation fails
            if self.human_points is None:
                # Create a fallback simple human model
                self.human_points = HumanModelGenerator.generate_seated_human(
                    [0, 0, 0], scale=1.0, rotation_deg=0.0
                )
                print("Using fallback human model")
    
    def setup_visualization(self):
        """Setup the PyVista visualization with interactive widgets."""
        # Use VTK safety wrapper for plotter creation
        self.plotter = VTKSafetyManager.create_safe_plotter(
            window_size=self.window_size
        )
        
        if self.plotter is None:
            raise RuntimeError("Failed to create VTK plotter safely")
            
        self.plotter.set_background('white')
        
        # Add point cloud with error handling
        try:
            if len(self.points) > 0:
                point_cloud = pv.PolyData(self.points)
                self.point_cloud_actor = VTKSafetyManager.safe_mesh_addition(
                    self.plotter,
                    point_cloud, 
                    color='lightblue', 
                    point_size=2, 
                    opacity=0.6,
                    name='point_cloud'
                )
            else:
                print("Warning: No points to display")
        except Exception as e:
            print(f"Warning: Could not add point cloud to visualization: {e}")
            self.point_cloud_actor = None
        
        # Generate initial human model with error handling
        try:
            self.generate_human_model()
        except Exception as e:
            print(f"Warning: Could not generate initial human model: {e}")
            # Continue without human model initially
        
        # Add human model with error handling
        try:
            if self.human_points is not None and len(self.human_points) > 0:
                human_cloud = pv.PolyData(self.human_points)
                self.human_actor = VTKSafetyManager.safe_mesh_addition(
                    self.plotter,
                    human_cloud,
                    color='red',
                    point_size=4,
                    name='human_model'
                )
            else:
                print("Warning: No human points to display")
                self.human_actor = None
        except Exception as e:
            print(f"Warning: Could not add human model to visualization: {e}")
            self.human_actor = None
        
        # Create initial cube bounds for widget
        cube_size = 1.0  # Initial cube size
        cube_bounds = [
            self.human_center[0] - cube_size/2, self.human_center[0] + cube_size/2,  # x_min, x_max
            self.human_center[1] - cube_size/2, self.human_center[1] + cube_size/2,  # y_min, y_max
            self.human_center[2] - cube_size/2, self.human_center[2] + cube_size/2   # z_min, z_max
        ]
        
        # Add interactive cube widget with error handling
        try:
            self.cube_widget = self.plotter.add_box_widget(
                callback=self.update_human_position,
                bounds=cube_bounds,
                factor=1.25,
                rotation_enabled=False,
                color='yellow'
            )
        except Exception as e:
            print(f"Warning: Could not add cube widget: {e}")
            self.cube_widget = None
        
        # Add rotation slider with error handling
        try:
            self.rotation_slider = self.plotter.add_slider_widget(
                callback=self.update_human_rotation,
                rng=[0, 360],
                value=self.human_rotation,
                title="Human Rotation (°)",
                pointa=(0.1, 0.9),
                pointb=(0.4, 0.9),
                style='modern'
            )
        except Exception as e:
            print(f"Warning: Could not add rotation slider: {e}")
            self.rotation_slider = None
        
        # Add text instructions
        self.plotter.add_text(
            "Drag cube handles to position and resize human model\n"
            "Use rotation slider to rotate human\n"
            "Close window to save the positioned model",
            position='upper_left',
            font_size=12,
            color='black'
        )
        
        # Set up camera
        self.plotter.camera_position = 'iso'
        self.plotter.show_axes()
        self.plotter.show_grid()
        
        print("\nInstructions:")
        print("- Drag the yellow cube handles to move and resize the human model")
        print("- Use the rotation slider to rotate the human around the Z-axis")
        print("- The red human model will update in real-time")
        print("- Close the window when you're satisfied with the position")
    
    def update_human_position(self, widget):
        """Callback function when cube widget is moved/resized."""
        # Protect against callback errors that could cause segmentation faults
        if self.updating:
            return
        
        self.updating = True
        
        try:
            # Validate widget and get bounds safely
            if widget is None:
                print("Warning: No widget provided to position callback")
                return
            
            # Get cube bounds from widget with error handling
            try:
                bounds = widget.GetBounds()
                if bounds is None or len(bounds) != 6:
                    print("Warning: Invalid bounds from widget")
                    return
            except Exception as e:
                print(f"Warning: Could not get widget bounds: {e}")
                return
            
            # Validate bounds
            x_min, x_max, y_min, y_max, z_min, z_max = bounds
            if any(np.isnan([x_min, x_max, y_min, y_max, z_min, z_max])):
                print("Warning: NaN values in widget bounds")
                return
            
            # Calculate new center and scale with validation
            new_center = [
                (x_min + x_max) / 2,
                (y_min + y_max) / 2,
                (z_min + z_max) / 2
            ]
            
            # Calculate scale based on cube size
            cube_width = x_max - x_min
            cube_height = z_max - z_min
            new_scale = min(cube_width, cube_height) / 1.0  # Normalize to default size
            
            # Update human model parameters with validation
            if all(np.isfinite(new_center)) and np.isfinite(new_scale):
                self.human_center = new_center
                self.human_scale = max(0.1, min(3.0, new_scale))  # Clamp scale between 0.1 and 3.0
                
                # Regenerate human model with error handling
                try:
                    self.generate_human_model()
                except Exception as e:
                    print(f"Warning: Error generating human model: {e}")
                    return
                
                # Update the human model visualization with error handling
                try:
                    if self.human_actor is not None:
                        VTKSafetyManager.safe_actor_removal(self.plotter, self.human_actor)
                    
                    if self.human_points is not None and len(self.human_points) > 0:
                        human_cloud = pv.PolyData(self.human_points)
                        self.human_actor = VTKSafetyManager.safe_mesh_addition(
                            self.plotter,
                            human_cloud,
                            color='red',
                            point_size=4,
                            name='human_model'
                        )
                    
                    # Update the display
                    VTKSafetyManager.safe_render(self.plotter)
                    
                except Exception as e:
                    print(f"Warning: Error updating human model visualization: {e}")
            else:
                print("Warning: Invalid center or scale values")
            
        except Exception as e:
            print(f"Error updating human position: {e}")
        except:
            # Catch any other exceptions to prevent segmentation faults
            print("Critical error in human position callback - skipping update")
        finally:
            self.updating = False
    
    def update_human_rotation(self, value):
        """Callback function when rotation slider is moved."""
        # Protect against callback errors that could cause segmentation faults
        if self.updating:
            return
        
        self.updating = True
        
        try:
            # Validate rotation value
            try:
                rotation_value = float(value)
                if not np.isfinite(rotation_value):
                    print("Warning: Invalid rotation value")
                    return
            except (ValueError, TypeError) as e:
                print(f"Warning: Could not convert rotation value to float: {e}")
                return
            
            # Update rotation angle with clamping
            self.human_rotation = np.clip(rotation_value, 0.0, 360.0)
            
            # Regenerate human model with new rotation
            try:
                self.generate_human_model()
            except Exception as e:
                print(f"Warning: Error generating human model with rotation: {e}")
                return
            
            # Update the human model visualization with error handling
            try:
                if self.human_actor is not None:
                    VTKSafetyManager.safe_actor_removal(self.plotter, self.human_actor)
                
                if self.human_points is not None and len(self.human_points) > 0:
                    human_cloud = pv.PolyData(self.human_points)
                    self.human_actor = VTKSafetyManager.safe_mesh_addition(
                        self.plotter,
                        human_cloud,
                        color='red',
                        point_size=4,
                        name='human_model'
                    )
                
                # Update the display
                VTKSafetyManager.safe_render(self.plotter)
                
            except Exception as e:
                print(f"Warning: Error updating human model visualization: {e}")
            
        except Exception as e:
            print(f"Error updating human rotation: {e}")
        except:
            # Catch any other exceptions to prevent segmentation faults
            print("Critical error in human rotation callback - skipping update")
        finally:
            self.updating = False
    
    def save_combined_point_cloud(self, output_file=None):
        """Save the point cloud with the positioned human model."""
        if output_file is None:
            output_file = self.point_cloud_file.parent / f"{self.point_cloud_file.stem}_with_human.npy"
        
        if self.human_points is None:
            print("No human model to save!")
            return
        
        # Combine original point cloud with human model
        combined_points = np.vstack([self.points, self.human_points])
        
        # Save to file
        np.save(output_file, combined_points)
        
        print(f"\nSaved combined point cloud to: {output_file}")
        print(f"Original points: {len(self.points):,}")
        print(f"Human model points: {len(self.human_points):,}")
        print(f"Total points: {len(combined_points):,}")
        print(f"Human position: [{self.human_center[0]:.3f}, {self.human_center[1]:.3f}, {self.human_center[2]:.3f}]")
        print(f"Human scale: {self.human_scale:.2f}")
        print(f"Human rotation: {self.human_rotation:.1f}°")
    
    def run_interactive_positioning(self, plotter_callback=None):
        """Run the interactive positioning interface.
        
        Args:
            plotter_callback: Optional callback to register plotter for tracking/cleanup
        """
        print("Starting interactive human positioning...")
        print("Loading visualization...")
        
        # Setup and show visualization
        self.setup_visualization()
        
        # Register plotter with callback if provided (for GUI integration)
        if plotter_callback and callable(plotter_callback):
            plotter_callback(self.plotter)
        
        # Show the interactive window with proper cleanup
        try:
            self.plotter.show(auto_close=False, interactive_update=True)
        except KeyboardInterrupt:
            print("Human positioning interrupted by user")
        except Exception as e:
            print(f"Error during human positioning: {e}")
        finally:
            # Only cleanup if no callback (standalone mode)
            # In GUI mode, the GUI handles cleanup to prevent double cleanup segfaults
            if plotter_callback is None:
                try:
                    # Clean up widgets first to prevent callback issues
                    if hasattr(self, 'cube_widget') and self.cube_widget is not None:
                        try:
                            self.cube_widget.Off()
                        except:
                            pass
                        self.cube_widget = None
                    
                    if hasattr(self, 'rotation_slider') and self.rotation_slider is not None:
                        try:
                            self.rotation_slider.Off()
                        except:
                            pass
                        self.rotation_slider = None
                    
                    # Enhanced cleanup for standalone mode
                    if hasattr(self.plotter, 'render_window') and self.plotter.render_window:
                        self.plotter.render_window.Finalize()
                    if hasattr(self.plotter, 'iren') and self.plotter.iren:
                        self.plotter.iren.TerminateApp()
                    self.plotter.close()
                except Exception as cleanup_error:
                    print(f"Warning during plotter cleanup: {cleanup_error}")
                finally:
                    # Use VTK safety manager for additional cleanup
                    VTKSafetyManager.cleanup_vtk_plotter(self.plotter)
                    self.plotter = None
                    
                    # Clean up references to prevent memory leaks
                    if hasattr(self, 'point_cloud_actor'):
                        self.point_cloud_actor = None
                    if hasattr(self, 'human_actor'):
                        self.human_actor = None
                    
                    # Force garbage collection to help with VTK cleanup
                    import gc
                    gc.collect()
                    gc.collect()
        
        # After window closes, save the result
        print("\nVisualization window closed.")
        print("Saving positioned human model...")
        
        # Save the positioned human model
        self.save_combined_point_cloud()
        
        print("Interactive positioning complete!")
    
    def cleanup_visualization(self):
        """Clean up visualization resources - for GUI integration."""
        try:
            # Clean up widgets
            if hasattr(self, 'cube_widget') and self.cube_widget is not None:
                try:
                    self.cube_widget.Off()
                except:
                    pass
                self.cube_widget = None
            
            if hasattr(self, 'rotation_slider') and self.rotation_slider is not None:
                try:
                    self.rotation_slider.Off()
                except:
                    pass
                self.rotation_slider = None
            
            # Clean up actors
            if hasattr(self, 'point_cloud_actor'):
                self.point_cloud_actor = None
            if hasattr(self, 'human_actor'):
                self.human_actor = None
            
            # Clean up plotter if it exists
            if hasattr(self, 'plotter') and self.plotter is not None:
                VTKSafetyManager.cleanup_vtk_plotter(self.plotter)
                self.plotter = None
            
            # Force garbage collection
            import gc
            gc.collect()
            
        except Exception as e:
            print(f"Warning during visualization cleanup: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Interactive Human Model Positioner")
    parser.add_argument("--input", default="output/interior_cockpit.npy",
                       help="Input point cloud file (.npy)")
    parser.add_argument("--output", 
                       help="Output file for combined point cloud (default: auto-generated)")
    parser.add_argument("--window-width", type=int, default=1600,
                       help="Visualization window width")
    parser.add_argument("--window-height", type=int, default=1200,
                       help="Visualization window height")
    
    args = parser.parse_args()
    
    # Check input file
    input_file = Path(args.input)
    if not input_file.exists():
        print(f"Error: Input file {input_file} does not exist!")
        print("Please run the E57 analysis first to generate the point cloud.")
        return 1
    
    # Create positioner
    positioner = InteractiveHumanPositioner(
        input_file, 
        window_size=(args.window_width, args.window_height)
    )
    
    # Run interactive positioning
    positioner.run_interactive_positioning()
    
    return 0


if __name__ == "__main__":
    exit(main())
