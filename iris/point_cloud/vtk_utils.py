#!/usr/bin/env python3
"""
VTK Safety Utilities Module

Centralized VTK safety patterns and resource management to prevent segmentation 
faults, especially critical on macOS systems.

This module consolidates all VTK safety patterns that were previously duplicated
across multiple files, providing a consistent and reliable interface for VTK operations.

Author: Dimitrije Stojanovic
Date: September 2025
Enhanced: Code Review Fixes
"""

import os
import sys
import gc
import time
import numpy as np
import pyvista as pv
from typing import Optional, Tuple, Any, Callable, Iterator, Dict
from contextlib import contextmanager
from pathlib import Path


class VTKSafetyManager:
    """Centralized VTK safety manager with comprehensive resource tracking."""

    # Class-level plotter tracking for resource management
    _active_plotters = []
    _environment_initialized = False
    
    @classmethod
    def setup_vtk_environment(cls, force_reinit: bool = False):
        """
        Set up VTK environment variables for cross-platform stability.

        Args:
            force_reinit: Force re-initialization even if already initialized
        """
        # Avoid redundant initialization unless forced
        if cls._environment_initialized and not force_reinit:
            return

        vtk_env_vars = {
            'VTK_RENDER_WINDOW_MAIN_THREAD': '1',
            'VTK_USE_COCOA': '1' if sys.platform == "darwin" else '0',
            'VTK_SILENCE_GET_VOID_POINTER_WARNINGS': '1',
            'VTK_DEBUG_LEAKS': '0',
            'VTK_AUTO_INIT': '1',
            'VTK_RENDERING_BACKEND': 'OpenGL2',
        }

        # Platform-specific settings
        if sys.platform == "darwin":  # macOS
            vtk_env_vars.update({
                'PYVISTA_OFF_SCREEN': '0',
                'PYVISTA_USE_PANEL': '0',
                'VTK_USE_OFFSCREEN': '0'
            })

        # Apply environment variables
        for key, value in vtk_env_vars.items():
            os.environ[key] = value

        cls._environment_initialized = True
    
    @classmethod
    def register_plotter(cls, plotter):
        """Register a plotter for tracking and cleanup."""
        if plotter not in cls._active_plotters:
            cls._active_plotters.append(plotter)
    
    @classmethod 
    def unregister_plotter(cls, plotter):
        """Unregister a plotter from tracking."""
        if plotter in cls._active_plotters:
            cls._active_plotters.remove(plotter)
    
    @classmethod
    def cleanup_all_plotters(cls):
        """Clean up all registered plotters."""
        for plotter in cls._active_plotters[:]:  # Copy list to avoid modification during iteration
            cls.cleanup_vtk_plotter(plotter)
        cls._active_plotters.clear()
        
    @classmethod
    def cleanup_vtk_plotter(cls, plotter):
        """
        Safely cleanup VTK plotter to prevent segmentation faults.
        
        Args:
            plotter: PyVista plotter instance to cleanup
        """
        if plotter is not None:
            try:
                # Unregister from tracking
                cls.unregister_plotter(plotter)
                
                # Properly close all render windows
                if hasattr(plotter, 'render_window') and plotter.render_window:
                    try:
                        plotter.render_window.Finalize()
                    except Exception as rw_error:
                        print(f"Warning: Could not finalize render window: {rw_error}")
                
                # Handle interactor cleanup (different methods for different VTK versions)
                if hasattr(plotter, 'iren') and plotter.iren:
                    try:
                        if hasattr(plotter.iren, 'TerminateApp'):
                            plotter.iren.TerminateApp()
                        elif hasattr(plotter.iren, 'ExitCallback'):
                            plotter.iren.ExitCallback()
                        elif hasattr(plotter.iren, 'Close'):
                            plotter.iren.Close()
                    except Exception as iren_error:
                        print(f"Warning: Could not cleanup interactor: {iren_error}")
                
                # Close the plotter itself
                try:
                    plotter.close()
                except Exception as close_error:
                    print(f"Warning: Could not close plotter: {close_error}")
                    
            except Exception as e:
                print(f"Warning: Error during plotter cleanup: {e}")
            finally:
                # Force cleanup
                try:
                    del plotter
                except:
                    pass
                    
        # Enhanced garbage collection
        gc.collect()
        gc.collect()  # Double collection for better cleanup
    
    @staticmethod
    def safe_vtk_operation(operation: Callable, operation_name: str = "VTK operation", 
                          max_retries: int = 3, **kwargs) -> Any:
        """
        Safely execute VTK operations with retry mechanism and error handling.
        
        Args:
            operation: Callable VTK operation to execute
            operation_name: Human-readable name for logging
            max_retries: Maximum number of retry attempts
            **kwargs: Arguments to pass to the operation
            
        Returns:
            Result of the operation or None if failed
        """
        last_error = None
        
        for attempt in range(max_retries):
            try:
                result = operation(**kwargs)
                if attempt > 0:
                    print(f"✅ {operation_name} succeeded on retry {attempt + 1}")
                return result
                
            except Exception as e:
                last_error = e
                print(f"⚠️  {operation_name} attempt {attempt + 1} failed: {e}")
                
                if attempt < max_retries - 1:
                    # Brief pause and garbage collection before retry
                    time.sleep(0.01)
                    gc.collect()
                else:
                    print(f"❌ {operation_name} failed after {max_retries} attempts")
        
        return None
    
    @classmethod
    def create_safe_plotter(cls, **plotter_kwargs) -> Optional[pv.Plotter]:
        """
        Safely create a PyVista plotter with environment setup.
        
        Args:
            **plotter_kwargs: Arguments to pass to pv.Plotter()
            
        Returns:
            PyVista plotter instance or None if creation failed
        """
        def create_plotter():
            # Ensure environment is set up
            cls.setup_vtk_environment()
            
            # Create plotter with safe defaults
            safe_kwargs = {
                'window_size': (1600, 1200),
            }
            safe_kwargs.update(plotter_kwargs)
            
            plotter = pv.Plotter(**safe_kwargs)
            
            # Register for tracking
            cls.register_plotter(plotter)
            
            return plotter
            
        return cls.safe_vtk_operation(create_plotter, "Plotter creation")
    
    @classmethod 
    def safe_mesh_addition(cls, plotter, mesh, **mesh_kwargs) -> Any:
        """
        Safely add mesh to plotter.
        
        Args:
            plotter: PyVista plotter instance
            mesh: Mesh to add
            **mesh_kwargs: Arguments to pass to add_mesh()
            
        Returns:
            Actor object or None if failed
        """
        def add_mesh():
            return plotter.add_mesh(mesh, **mesh_kwargs)
            
        return cls.safe_vtk_operation(add_mesh, "Mesh addition")
    
    @classmethod
    def safe_actor_removal(cls, plotter, actor) -> bool:
        """
        Safely remove actor from plotter.
        
        Args:
            plotter: PyVista plotter instance
            actor: Actor to remove
            
        Returns:
            True if successful, False otherwise
        """
        def remove_actor():
            return plotter.remove_actor(actor, reset_camera=False, render=False)
            
        result = cls.safe_vtk_operation(remove_actor, "Actor removal")
        return result is not None
    
    @classmethod
    def safe_render(cls, plotter) -> bool:
        """
        Safely render plotter.
        
        Args:
            plotter: PyVista plotter instance
            
        Returns:
            True if successful, False otherwise
        """
        def render():
            return plotter.render()
            
        result = cls.safe_vtk_operation(render, "Rendering")
        return result is not None
    
    @classmethod
    def safe_show(cls, plotter, **show_kwargs) -> bool:
        """
        Safely show plotter visualization.
        
        Args:
            plotter: PyVista plotter instance  
            **show_kwargs: Arguments to pass to show()
            
        Returns:
            True if successful, False otherwise
        """
        def show():
            return plotter.show(**show_kwargs)
            
        try:
            result = cls.safe_vtk_operation(show, "Show visualization")
            return result is not None
        finally:
            # Always cleanup after show
            cls.cleanup_vtk_plotter(plotter)


class VTKMeshValidator:
    """Utility class for validating mesh data before VTK operations."""
    
    @staticmethod
    def validate_points(points: np.ndarray) -> Tuple[bool, str]:
        """
        Validate point cloud data for VTK compatibility.
        
        Args:
            points: Point cloud array to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if points is None:
            return False, "Points array is None"
        
        if not isinstance(points, np.ndarray):
            return False, "Points must be numpy array"
        
        if len(points.shape) != 2:
            return False, f"Points must be 2D array, got shape {points.shape}"
        
        if points.shape[1] != 3:
            return False, f"Points must have 3 columns (x,y,z), got {points.shape[1]}"
        
        if len(points) == 0:
            return False, "Points array is empty"
        
        if not np.isfinite(points).all():
            return False, "Points contain non-finite values (NaN or inf)"
        
        return True, "Valid"
    
    @staticmethod
    def validate_scalars(scalars: np.ndarray, num_points: int) -> Tuple[bool, str]:
        """
        Validate scalar data for VTK compatibility.
        
        Args:
            scalars: Scalar array to validate
            num_points: Expected number of points
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if scalars is None:
            return True, "No scalars (valid)"  # Scalars are optional
        
        if not isinstance(scalars, np.ndarray):
            return False, "Scalars must be numpy array"
        
        if len(scalars) != num_points:
            return False, f"Scalars length {len(scalars)} doesn't match points {num_points}"
        
        if not np.isfinite(scalars).all():
            return False, "Scalars contain non-finite values"
        
        return True, "Valid"
    
    @staticmethod
    def create_safe_polydata(points: np.ndarray, scalars: Optional[np.ndarray] = None,
                           scalar_name: str = "values") -> Optional[pv.PolyData]:
        """
        Create PyVista PolyData with validation.
        
        Args:
            points: Point cloud array
            scalars: Optional scalar data
            scalar_name: Name for scalar field
            
        Returns:
            PyVista PolyData or None if validation failed
        """
        # Validate points
        points_valid, points_error = VTKMeshValidator.validate_points(points)
        if not points_valid:
            print(f"Point validation failed: {points_error}")
            return None
        
        # Validate scalars if provided
        if scalars is not None:
            scalars_valid, scalars_error = VTKMeshValidator.validate_scalars(scalars, len(points))
            if not scalars_valid:
                print(f"Scalar validation failed: {scalars_error}")
                return None
        
        try:
            cloud = pv.PolyData(points)
            if scalars is not None:
                cloud[scalar_name] = scalars
            return cloud
        except Exception as e:
            print(f"Failed to create PolyData: {e}")
            return None


class VTKErrorHandler:
    """Centralized VTK error handling and recovery."""
    
    @staticmethod
    def handle_vtk_error(error: Exception, context: str = "VTK operation") -> None:
        """
        Handle VTK-specific errors with appropriate logging and recovery suggestions.
        
        Args:
            error: Exception that occurred
            context: Context where the error occurred
        """
        error_msg = str(error).lower()
        
        if "segmentation fault" in error_msg or "sigsegv" in error_msg:
            print(f"🚨 CRITICAL VTK Error in {context}: Segmentation fault detected!")
            print("Recommendations:")
            print("  1. Ensure VTK environment variables are set")
            print("  2. Check that all VTK objects are properly cleaned up")
            print("  3. Verify operations run on main thread (macOS)")
            
        elif "render window" in error_msg:
            print(f"🖼️  VTK Render Window Error in {context}: {error}")
            print("Try setting VTK_RENDER_WINDOW_MAIN_THREAD=1")
            
        elif "interactor" in error_msg:
            print(f"🖱️  VTK Interactor Error in {context}: {error}")
            print("Interactor cleanup may have failed - this is usually non-critical")
            
        elif "memory" in error_msg or "allocation" in error_msg:
            print(f"💾 VTK Memory Error in {context}: {error}")
            print("Try reducing point cloud size or use stride parameter")
            
        else:
            print(f"⚠️  VTK Error in {context}: {error}")
    
    @staticmethod
    def safe_vtk_call(func: Callable, context: str = "VTK operation", 
                     fallback_result: Any = None) -> Any:
        """
        Execute VTK function call with error handling.
        
        Args:
            func: Function to call
            context: Context description for error messages
            fallback_result: Value to return if function fails
            
        Returns:
            Function result or fallback_result if failed
        """
        try:
            return func()
        except Exception as e:
            VTKErrorHandler.handle_vtk_error(e, context)
            return fallback_result


    @classmethod
    @contextmanager
    def vtk_plotter(cls, **plotter_kwargs) -> Iterator[pv.Plotter]:
        """
        Context manager for VTK plotter with automatic cleanup.

        Usage:
            with VTKSafetyManager.vtk_plotter(window_size=(800, 600)) as plotter:
                plotter.add_mesh(mesh)
                plotter.show()
        """
        plotter = None
        try:
            # Ensure environment is set up
            cls.setup_vtk_environment()

            # Create plotter with safety checks
            plotter = cls.create_safe_plotter(**plotter_kwargs)
            if plotter is None:
                raise RuntimeError("Failed to create VTK plotter")

            yield plotter

        except Exception as e:
            VTKErrorHandler.handle_vtk_error(e, "vtk_plotter context manager")
            raise
        finally:
            # Always cleanup
            if plotter is not None:
                cls.cleanup_vtk_plotter(plotter)

    @classmethod
    @contextmanager
    def vtk_environment(cls, force_reinit: bool = False) -> Iterator[None]:
        """
        Context manager for VTK environment setup.

        Usage:
            with VTKSafetyManager.vtk_environment():
                # VTK operations here
                pass
        """
        try:
            cls.setup_vtk_environment(force_reinit=force_reinit)
            yield
        except Exception as e:
            VTKErrorHandler.handle_vtk_error(e, "vtk_environment context manager")
            raise
        finally:
            # Perform cleanup if needed
            cls._perform_environment_cleanup()

    @classmethod
    def _perform_environment_cleanup(cls):
        """Clean up environment-specific resources."""
        # Force garbage collection for VTK objects
        for _ in range(2):
            gc.collect()

    @classmethod
    def vtk_safe_operation(cls, operation: Callable, *args, **kwargs) -> Any:
        """
        Execute a VTK operation with comprehensive error handling.

        Args:
            operation: Function to execute safely
            *args: Arguments for the operation
            **kwargs: Keyword arguments for the operation

        Returns:
            Result of the operation or None if it fails
        """
        try:
            cls.setup_vtk_environment()
            return operation(*args, **kwargs)
        except Exception as e:
            operation_name = getattr(operation, '__name__', 'unknown_operation')
            VTKErrorHandler.handle_vtk_error(e, f"vtk_safe_operation({operation_name})")
            return None

    @classmethod
    def get_platform_optimized_settings(cls) -> Dict[str, Any]:
        """
        Get platform-optimized VTK settings.

        Returns:
            Dictionary with optimized settings for current platform
        """
        settings = {
            'window_size': (1200, 900),
            'point_size': 2.0,
            'background_color': 'black',
            'show_axes': True,
        }

        if sys.platform == "darwin":  # macOS
            settings.update({
                'window_size': (1600, 1200),  # Better for Retina displays
                'point_size': 3.0,           # More visible on high-DPI
                'multi_samples': 4,          # Anti-aliasing
                'use_depth_buffer': True,    # Better rendering
            })
        elif sys.platform == "win32":  # Windows
            settings.update({
                'window_size': (1400, 1000),
                'point_size': 2.5,
            })

        return settings


# Convenience functions for backward compatibility
def setup_macos_environment():
    """Legacy function - use VTKSafetyManager.setup_vtk_environment() instead."""
    VTKSafetyManager.setup_vtk_environment()


def cleanup_vtk_plotter(plotter):
    """Legacy function - use VTKSafetyManager.cleanup_vtk_plotter() instead."""
    VTKSafetyManager.cleanup_vtk_plotter(plotter)


# Module initialization
if __name__ == "__main__":
    print("VTK Safety Utilities Module")
    print("This module provides centralized VTK safety patterns.")
    print("Import and use VTKSafetyManager for safe VTK operations.")