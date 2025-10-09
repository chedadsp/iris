#!/usr/bin/env python3
"""
Platform Detection Utilities

Provides cross-platform compatibility detection and configuration.
"""

import platform
import sys
import os
from typing import Dict, Any, Optional


class PlatformDetector:
    """Utility class for platform detection and configuration."""

    def __init__(self):
        """Initialize platform detector."""
        self._system = platform.system()
        self._cache: Dict[str, Any] = {}

    @property
    def is_macos(self) -> bool:
        """Check if running on macOS."""
        return self._system == "Darwin"

    @property
    def is_windows(self) -> bool:
        """Check if running on Windows."""
        return self._system == "Windows"

    @property
    def is_linux(self) -> bool:
        """Check if running on Linux."""
        return self._system == "Linux"

    @property
    def system_name(self) -> str:
        """Get the system name."""
        return self._system

    def get_python_version(self) -> tuple:
        """Get Python version as tuple."""
        return sys.version_info[:3]

    def is_python_compatible(self, min_version: tuple = (3, 8)) -> bool:
        """
        Check if Python version is compatible.

        Args:
            min_version: Minimum required Python version as tuple

        Returns:
            True if compatible, False otherwise
        """
        return self.get_python_version() >= min_version

    def get_architecture(self) -> str:
        """Get system architecture."""
        return platform.machine()

    def is_64bit(self) -> bool:
        """Check if running on 64-bit architecture."""
        return '64' in self.get_architecture() or sys.maxsize > 2**32

    def get_platform_config(self) -> Dict[str, Any]:
        """
        Get platform-specific configuration.

        Returns:
            Dictionary with platform configuration
        """
        if 'platform_config' not in self._cache:
            config = {
                'system': self.system_name,
                'is_macos': self.is_macos,
                'is_windows': self.is_windows,
                'is_linux': self.is_linux,
                'python_version': self.get_python_version(),
                'architecture': self.get_architecture(),
                'is_64bit': self.is_64bit(),
                'supports_gui': self._check_gui_support(),
                'vtk_requirements': self._get_vtk_requirements(),
                'path_separator': os.pathsep,
                'line_ending': os.linesep,
            }
            self._cache['platform_config'] = config

        return self._cache['platform_config']

    def _check_gui_support(self) -> bool:
        """Check if GUI support is available on this platform."""
        if self.is_windows or self.is_macos:
            return True

        if self.is_linux:
            # Check for X11 display
            return 'DISPLAY' in os.environ and os.environ['DISPLAY'] != ''

        return False

    def _get_vtk_requirements(self) -> Dict[str, str]:
        """Get VTK requirements for this platform."""
        requirements = {
            'VTK_RENDER_WINDOW_MAIN_THREAD': '1',
            'VTK_SILENCE_GET_VOID_POINTER_WARNINGS': '1',
            'VTK_DEBUG_LEAKS': '0',
            'VTK_AUTO_INIT': '1',
            'VTK_RENDERING_BACKEND': 'OpenGL2',
        }

        if self.is_macos:
            requirements.update({
                'VTK_USE_COCOA': '1',
                'VTK_USE_OFFSCREEN': '0',
                'PYVISTA_OFF_SCREEN': '0',
                'PYVISTA_USE_PANEL': '0'
            })
        elif self.is_windows:
            requirements.update({
                'VTK_USE_WIN32': '1',
            })
        elif self.is_linux:
            requirements.update({
                'VTK_USE_X': '1',
            })

        return requirements

    def get_recommended_window_size(self) -> tuple:
        """
        Get recommended window size for this platform.

        Returns:
            Tuple of (width, height)
        """
        if self.is_macos:
            # macOS typically has high-DPI screens
            return (1600, 1200)
        elif self.is_windows:
            # Windows default
            return (1400, 1000)
        else:
            # Linux/other
            return (1280, 960)

    def get_memory_info(self) -> Optional[Dict[str, int]]:
        """
        Get system memory information if available.

        Returns:
            Dictionary with memory info or None if not available
        """
        try:
            import psutil
            memory = psutil.virtual_memory()
            return {
                'total': memory.total,
                'available': memory.available,
                'percent_used': memory.percent,
            }
        except ImportError:
            # psutil not available
            return None

    def print_platform_summary(self) -> None:
        """Print a summary of the platform configuration."""
        config = self.get_platform_config()

        print("🖥️  Platform Summary:")
        print(f"  System: {config['system']}")
        print(f"  Architecture: {config['architecture']} ({'64-bit' if config['is_64bit'] else '32-bit'})")
        print(f"  Python: {'.'.join(map(str, config['python_version']))}")
        print(f"  GUI Support: {'Yes' if config['supports_gui'] else 'No'}")

        # Memory info if available
        memory_info = self.get_memory_info()
        if memory_info:
            total_gb = memory_info['total'] / (1024**3)
            available_gb = memory_info['available'] / (1024**3)
            print(f"  Memory: {total_gb:.1f}GB total, {available_gb:.1f}GB available")

    def warn_about_limitations(self) -> None:
        """Print warnings about platform-specific limitations."""
        warnings = []

        if not self.is_python_compatible():
            warnings.append("Python version may be too old for optimal performance")

        if not self.is_64bit():
            warnings.append("32-bit architecture may limit memory usage")

        if self.is_macos and not self.get_platform_config()['supports_gui']:
            warnings.append("GUI may not work properly without proper macOS setup")

        if self.is_linux and not self.get_platform_config()['supports_gui']:
            warnings.append("X11 display not available - GUI mode will not work")

        memory_info = self.get_memory_info()
        if memory_info and memory_info['available'] < 2 * 1024**3:  # Less than 2GB
            warnings.append("Low available memory - large point clouds may cause issues")

        if warnings:
            print("⚠️  Platform Warnings:")
            for warning in warnings:
                print(f"  - {warning}")
        else:
            print("✅ No platform-specific warnings")

    def get_optimal_settings(self) -> Dict[str, Any]:
        """
        Get optimal settings for this platform.

        Returns:
            Dictionary with optimal configuration settings
        """
        settings = {
            'window_size': self.get_recommended_window_size(),
            'vtk_environment': self._get_vtk_requirements(),
            'memory_management': {
                'gc_threshold': 10000 if self.is_macos else 5000,
                'cleanup_frequency': 2 if self.is_macos else 1,
            }
        }

        # Adjust based on available memory
        memory_info = self.get_memory_info()
        if memory_info:
            if memory_info['available'] < 4 * 1024**3:  # Less than 4GB
                settings['memory_management']['cleanup_frequency'] = 3
                settings['point_stride'] = 2  # Skip every other point for large files

        return settings