#!/usr/bin/env python3
"""
GUI Launcher Mode

Handles launching the application in GUI mode, extracted from the
original launcher implementations with improved error handling and
platform-specific optimizations.
"""

from typing import Tuple, Optional
from pathlib import Path
import sys

from .launcher_base import BaseLauncher, LauncherValidationError, LauncherExecutionError
from .dependency_checker import DependencyChecker
from .environment_setup import EnvironmentSetup
import sys
from pathlib import Path
# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from platforms import PlatformDetector, MacOSOptimizer


class GUILauncher(BaseLauncher):
    """Launcher for GUI mode."""

    def __init__(self, **kwargs):
        """Initialize GUI launcher."""
        super().__init__(**kwargs)
        self.dependency_checker = DependencyChecker()
        self.environment_setup = EnvironmentSetup()
        self.platform_detector = PlatformDetector()
        self.macos_optimizer = MacOSOptimizer() if self.platform_detector.is_macos else None

    def validate_requirements(self) -> Tuple[bool, str]:
        """
        Validate requirements for GUI mode.

        Returns:
            Tuple of (success, error_message)
        """
        try:
            # Check GUI-specific dependencies
            success, missing_required, missing_optional = self.dependency_checker.check_mode_dependencies('gui')

            if not success:
                error_msg = f"Missing required dependencies for GUI mode: {', '.join(missing_required)}"
                suggestions = self.dependency_checker.get_installation_suggestions(missing_required)
                if suggestions:
                    error_msg += "\n" + "\n".join(suggestions)
                return False, error_msg

            # Check platform GUI support
            if not self.platform_detector.get_platform_config()['supports_gui']:
                return False, "GUI support not available on this platform (no display detected)"

            # Warn about missing optional dependencies
            if missing_optional:
                print(f"⚠️  Missing optional dependencies: {', '.join(missing_optional)}")
                print("Some GUI features may be disabled.")

            return True, ""

        except Exception as e:
            return False, f"GUI validation error: {e}"

    def launch(self, **kwargs) -> int:
        """
        Launch GUI mode.

        Args:
            **kwargs: Additional parameters (unused in GUI mode)

        Returns:
            Exit code
        """
        try:
            self.print_mode_info()

            # Validate requirements
            valid, error_msg = self.validate_requirements()
            if not valid:
                print(f"❌ {error_msg}")
                return 1

            # Set up environment
            print("📋 Setting up GUI environment...")
            self.environment_setup.setup_gui_environment()

            # Apply platform-specific optimizations
            if self.macos_optimizer:
                print("🍎 Applying macOS optimizations...")
                if not self.macos_optimizer.apply_all_optimizations():
                    print("⚠️  macOS optimizations failed, continuing anyway...")

            # Print platform information
            self.platform_detector.print_platform_summary()
            self.platform_detector.warn_about_limitations()

            # Import and launch GUI application
            print("🚀 Launching GUI application...")
            return self._launch_gui_application()

        except Exception as e:
            return self.handle_error(e, "GUI launch")

    def _launch_gui_application(self) -> int:
        """
        Launch the actual GUI application.

        Returns:
            Exit code from GUI application
        """
        try:
            # Import GUI application (done here to ensure environment is set up first)
            from lidar_gui_app import LidarGUIApp

            print("✨ Starting LIDAR GUI Application...")

            # Create and run the application
            app = LidarGUIApp()
            app.run()

            print("✅ GUI application completed successfully")
            return 0

        except ImportError as e:
            error_msg = f"GUI application not available: {e}"
            print(f"❌ {error_msg}")
            print("Make sure lidar_gui_app.py is in the correct location.")
            return 1

        except KeyboardInterrupt:
            print("\n⏹️  GUI application interrupted by user")
            return 0

        except Exception as e:
            raise LauncherExecutionError(f"GUI application failed: {e}") from e

    def cleanup(self) -> None:
        """Clean up GUI-specific resources."""
        try:
            # GUI-specific cleanup
            if self.macos_optimizer:
                self.macos_optimizer.cleanup_on_exit()

            # Call parent cleanup
            super().cleanup()

            print("🧹 GUI cleanup completed")

        except Exception as e:
            print(f"⚠️  GUI cleanup error: {e}")

    def get_mode_description(self) -> str:
        """Get description of GUI mode."""
        return ("GUI mode provides a comprehensive graphical interface for LIDAR point cloud "
                "processing and visualization with interactive tools and real-time feedback.")

    def print_mode_help(self) -> None:
        """Print help information for GUI mode."""
        print("GUI Mode Help:")
        print("=" * 40)
        print(self.get_mode_description())
        print()
        print("Features:")
        print("- File loading for E57, PCD, and ROS bag formats")
        print("- Vehicle analysis with progress tracking")
        print("- Interactive cube selection for interior extraction")
        print("- Human model positioning and visualization")
        print("- Results visualization with multiple view modes")
        print("- Comprehensive logging and status reporting")
        print()
        print("Requirements:")
        print("- tkinter (GUI framework)")
        print("- PyVista/VTK (3D visualization)")
        print("- Display/windowing system")
        print()
        print("Usage:")
        print("  python launcher.py --gui")
        print("  python launcher.py  # GUI is default mode")

    def is_default_mode(self) -> bool:
        """Check if this is the default launcher mode."""
        return True  # GUI is the default mode