#!/usr/bin/env python3
"""
Environment Setup

Consolidated environment setup logic extracted from the original
launcher implementations. Handles Python path setup, platform-specific
optimizations, and runtime configuration.
"""

import os
import sys
import platform
from pathlib import Path
from typing import Dict, Optional


class EnvironmentSetup:
    """Centralized environment setup for all launcher modes."""

    def __init__(self):
        """Initialize environment setup."""
        self.platform = platform.system()
        self.is_macos = self.platform == "Darwin"
        self.is_windows = self.platform == "Windows"
        self.is_linux = self.platform == "Linux"

    def setup_python_path(self, additional_paths: Optional[list] = None) -> None:
        """
        Set up Python path to include necessary modules.

        Args:
            additional_paths: Optional list of additional paths to add
        """
        # Get the current directory (where the launcher is located)
        current_dir = Path(__file__).parent.parent  # Go up from src/core/ to src/

        # Add the src directory to Python path
        if str(current_dir) not in sys.path:
            sys.path.insert(0, str(current_dir))
            print(f"✅ Added to Python path: {current_dir}")

        # Add any additional paths
        if additional_paths:
            for path in additional_paths:
                path_obj = Path(path)
                if path_obj.exists() and str(path_obj) not in sys.path:
                    sys.path.insert(0, str(path_obj))
                    print(f"✅ Added to Python path: {path_obj}")

    def setup_base_environment(self) -> None:
        """Set up basic environment variables common to all modes."""
        # Set up Python environment
        os.environ['PYTHONPATH'] = os.pathsep.join(sys.path)

        # Ensure UTF-8 encoding
        if 'PYTHONIOENCODING' not in os.environ:
            os.environ['PYTHONIOENCODING'] = 'utf-8'

        print(f"✅ Base environment configured for {self.platform}")

    def setup_vtk_environment(self) -> None:
        """
        Set up VTK environment variables for cross-platform stability.
        This consolidates VTK setup that was duplicated across launcher files.
        """
        # Base VTK environment variables
        vtk_env_vars = {
            'VTK_RENDER_WINDOW_MAIN_THREAD': '1',
            'VTK_SILENCE_GET_VOID_POINTER_WARNINGS': '1',
            'VTK_DEBUG_LEAKS': '0',
            'VTK_AUTO_INIT': '1',
            'VTK_RENDERING_BACKEND': 'OpenGL2',
        }

        # Platform-specific VTK settings
        if self.is_macos:
            # macOS-specific VTK optimizations
            vtk_env_vars.update({
                'VTK_USE_COCOA': '1',
                'VTK_USE_OFFSCREEN': '0',
                'PYVISTA_OFF_SCREEN': '0',
                'PYVISTA_USE_PANEL': '0'
            })
        else:
            # Non-macOS settings
            vtk_env_vars.update({
                'VTK_USE_COCOA': '0',
            })

        # Apply environment variables
        for key, value in vtk_env_vars.items():
            os.environ[key] = value

        print(f"✅ VTK environment configured for {self.platform}")

        # Print applied VTK settings for debugging
        if os.environ.get('DEBUG', '').lower() == 'true':
            print("VTK Environment Variables:")
            for key, value in vtk_env_vars.items():
                print(f"  {key}={value}")

    def setup_gui_environment(self) -> None:
        """Set up environment for GUI mode."""
        self.setup_base_environment()
        self.setup_vtk_environment()

        # GUI-specific settings
        if self.is_macos:
            # macOS GUI optimizations
            os.environ['MACOSX_DEPLOYMENT_TARGET'] = '10.9'

        print("✅ GUI environment configured")

    def setup_cli_environment(self) -> None:
        """Set up environment for CLI mode."""
        self.setup_base_environment()

        # CLI may use VTK for visualization, so set it up
        self.setup_vtk_environment()

        print("✅ CLI environment configured")

    def setup_headless_environment(self) -> None:
        """Set up environment for headless/analysis-only mode."""
        self.setup_base_environment()

        # Headless mode - disable GUI features
        os.environ['PYVISTA_OFF_SCREEN'] = '1'
        os.environ['DISPLAY'] = ''  # Force headless on Linux

        # May still need VTK for some analysis tasks
        self.setup_vtk_environment()

        print("✅ Headless environment configured")

    def get_platform_info(self) -> Dict[str, str]:
        """
        Get platform information for debugging and logging.

        Returns:
            Dictionary with platform details
        """
        return {
            'system': self.platform,
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'python_implementation': platform.python_implementation(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'architecture': str(platform.architecture()),
        }

    def print_platform_info(self) -> None:
        """Print platform information."""
        info = self.get_platform_info()
        print("📋 Platform Information:")
        for key, value in info.items():
            if value:  # Only print non-empty values
                print(f"  {key.replace('_', ' ').title()}: {value}")

    def validate_environment(self) -> bool:
        """
        Validate that the environment is properly set up.

        Returns:
            True if environment is valid, False otherwise
        """
        try:
            # Check Python path
            if len(sys.path) == 0:
                print("❌ Python path is empty")
                return False

            # Check essential environment variables
            essential_vars = ['PYTHONPATH']
            for var in essential_vars:
                if var not in os.environ:
                    print(f"⚠️  Environment variable {var} not set")

            # Platform-specific checks
            if self.is_macos:
                # Check macOS VTK variables
                macos_vars = ['VTK_USE_COCOA', 'VTK_RENDER_WINDOW_MAIN_THREAD']
                for var in macos_vars:
                    if os.environ.get(var) != '1':
                        print(f"⚠️  macOS VTK variable {var} not set correctly")

            print("✅ Environment validation passed")
            return True

        except Exception as e:
            print(f"❌ Environment validation failed: {e}")
            return False