#!/usr/bin/env python3
"""
macOS-Specific Optimizations

Contains macOS-specific optimizations that were scattered across
the original launcher implementations. This consolidates all macOS
VTK stability fixes and performance optimizations.
"""

import os
import sys
import gc
import atexit
from typing import Dict, Any, Optional


class MacOSOptimizer:
    """macOS-specific optimizations for VTK stability and performance."""

    def __init__(self):
        """Initialize macOS optimizer."""
        self.is_macos = sys.platform == "darwin"
        self.optimizations_applied = False

    def apply_vtk_optimizations(self) -> None:
        """
        Apply macOS-specific VTK optimizations to prevent segmentation faults.
        This consolidates VTK setup from the original launchers.
        """
        if not self.is_macos:
            print("⚠️  MacOS optimizations called on non-macOS platform")
            return

        # macOS VTK environment variables for stability
        vtk_env_vars = {
            'VTK_RENDER_WINDOW_MAIN_THREAD': '1',
            'VTK_USE_COCOA': '1',
            'VTK_SILENCE_GET_VOID_POINTER_WARNINGS': '1',
            'VTK_DEBUG_LEAKS': '0',
            'VTK_USE_OFFSCREEN': '0',
            'VTK_AUTO_INIT': '1',

            # PyVista-specific settings for macOS
            'PYVISTA_OFF_SCREEN': '0',
            'PYVISTA_USE_PANEL': '0',

            # Additional stability settings
            'VTK_RENDERING_BACKEND': 'OpenGL2',
            'VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN': '0',
        }

        # Apply environment variables
        for key, value in vtk_env_vars.items():
            os.environ[key] = value
            print(f"Set {key}={value}")

        self.optimizations_applied = True
        print("✅ Applied macOS VTK optimizations")

    def setup_memory_management(self) -> None:
        """Set up enhanced memory management for macOS."""
        if not self.is_macos:
            return

        # Register cleanup function for application exit
        atexit.register(self.cleanup_on_exit)

        # Set garbage collection thresholds for better VTK cleanup
        gc.set_threshold(700, 10, 10)  # More aggressive GC for VTK objects

        print("✅ Enhanced memory management configured for macOS")

    def setup_gui_optimizations(self) -> None:
        """Set up GUI-specific optimizations for macOS."""
        if not self.is_macos:
            return

        # macOS GUI environment settings
        gui_settings = {
            'MACOSX_DEPLOYMENT_TARGET': '10.9',  # Minimum macOS version
            'NSHighResolutionCapable': 'YES',    # Support high-DPI displays
        }

        for key, value in gui_settings.items():
            if key not in os.environ:  # Don't override existing settings
                os.environ[key] = value

        print("✅ GUI optimizations configured for macOS")

    def cleanup_on_exit(self) -> None:
        """Enhanced cleanup for macOS to prevent resource leaks."""
        print("🧹 Running macOS-specific cleanup...")

        # Multiple garbage collection passes (important for VTK on macOS)
        for i in range(3):
            collected = gc.collect()
            if collected > 0:
                print(f"GC pass {i+1}: collected {collected} objects")

        print("✅ macOS cleanup completed")

    def get_recommended_settings(self) -> Dict[str, Any]:
        """
        Get macOS-specific recommended settings.

        Returns:
            Dictionary with optimized settings for macOS
        """
        if not self.is_macos:
            return {}

        return {
            'vtk_settings': {
                'window_size': (1600, 1200),  # Good for Retina displays
                'point_size': 2.0,           # Appropriate for high-DPI
                'anti_aliasing': True,       # Better quality on Retina
                'multi_sampling': 4,         # Reduce aliasing
            },
            'memory_settings': {
                'gc_frequency': 2,           # More frequent GC
                'cleanup_threshold': 1000,   # Clean up after processing fewer points
                'vtk_cleanup_passes': 3,     # Multiple VTK cleanup attempts
            },
            'performance_settings': {
                'use_metal_backend': False,  # OpenGL2 is more stable
                'enable_transparency': False, # Can cause issues on some macOS versions
                'vsync': True,               # Better display synchronization
            }
        }

    def validate_macos_environment(self) -> bool:
        """
        Validate that macOS environment is properly configured.

        Returns:
            True if environment is valid, False otherwise
        """
        if not self.is_macos:
            print("⚠️  Not running on macOS")
            return True  # Not an error, just not applicable

        issues = []

        # Check essential VTK variables
        required_vars = [
            'VTK_RENDER_WINDOW_MAIN_THREAD',
            'VTK_USE_COCOA',
        ]

        for var in required_vars:
            if os.environ.get(var) != '1':
                issues.append(f"VTK variable {var} not set correctly")

        # Check macOS version compatibility
        try:
            import platform
            macos_version = platform.mac_ver()[0]
            if macos_version:
                # Parse version (e.g., "10.15.7" -> (10, 15, 7))
                version_parts = [int(x) for x in macos_version.split('.')]
                if len(version_parts) >= 2:
                    major, minor = version_parts[:2]
                    if major < 10 or (major == 10 and minor < 9):
                        issues.append(f"macOS version {macos_version} may not be supported")
        except Exception as e:
            issues.append(f"Could not determine macOS version: {e}")

        # Check for common problematic environment variables
        problematic_vars = {
            'VTK_DEBUG_LEAKS': '1',  # Should be '0'
            'PYVISTA_OFF_SCREEN': '1',  # Should be '0' for GUI
        }

        for var, bad_value in problematic_vars.items():
            if os.environ.get(var) == bad_value:
                issues.append(f"Environment variable {var} set to problematic value: {bad_value}")

        # Report issues
        if issues:
            print("❌ macOS environment validation failed:")
            for issue in issues:
                print(f"  - {issue}")
            return False
        else:
            print("✅ macOS environment validation passed")
            return True

    def print_macos_info(self) -> None:
        """Print macOS-specific information."""
        if not self.is_macos:
            print("ℹ️  Not running on macOS")
            return

        try:
            import platform
            macos_version = platform.mac_ver()[0]
            print(f"🍎 macOS Information:")
            print(f"  Version: {macos_version}")
            print(f"  Optimizations applied: {'Yes' if self.optimizations_applied else 'No'}")

            # Check if running on Apple Silicon
            machine = platform.machine()
            if machine == 'arm64':
                print(f"  Architecture: Apple Silicon (M1/M2)")
            elif machine == 'x86_64':
                print(f"  Architecture: Intel")
            else:
                print(f"  Architecture: {machine}")

            # Check display information
            try:
                # This is a rough way to detect Retina displays
                import tkinter as tk
                root = tk.Tk()
                dpi = root.winfo_fpixels('1i')
                root.destroy()
                if dpi > 120:
                    print(f"  Display: High-DPI detected (DPI: {dpi:.0f})")
                else:
                    print(f"  Display: Standard DPI ({dpi:.0f})")
            except:
                print(f"  Display: Could not determine DPI")

        except Exception as e:
            print(f"❌ Could not get macOS information: {e}")

    def apply_all_optimizations(self) -> bool:
        """
        Apply all macOS optimizations.

        Returns:
            True if optimizations were applied successfully
        """
        if not self.is_macos:
            print("ℹ️  Skipping macOS optimizations (not running on macOS)")
            return True

        try:
            print("🍎 Applying macOS optimizations...")

            self.apply_vtk_optimizations()
            self.setup_memory_management()
            self.setup_gui_optimizations()

            # Validate the setup
            is_valid = self.validate_macos_environment()

            if is_valid:
                print("✅ All macOS optimizations applied successfully")
                return True
            else:
                print("❌ macOS optimizations applied but validation failed")
                return False

        except Exception as e:
            print(f"❌ Failed to apply macOS optimizations: {e}")
            return False