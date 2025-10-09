#!/usr/bin/env python3
"""
Universal Launcher for LIDAR Point Cloud Analysis Suite

This is the unified launcher that replaces the original 3 separate launchers
(launcher.py, macos_launcher.py, lidar_gui_app.py) with a single, well-architected
dispatch system.

Features:
- GUI mode (default): Full Tkinter interface
- CLI mode: Interactive command-line interface
- Headless mode: Automated processing pipeline
- Platform-aware optimizations (especially macOS VTK stability)
- Comprehensive error handling and dependency validation
- Backward compatibility with all original command-line arguments

Author: Dimitrije Stojanovic
Date: September 2025
Refactored: December 2025 (Consolidated Architecture)
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any

# Ensure core modules are importable
sys.path.insert(0, str(Path(__file__).parent))

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    from core import (
        GUILauncher,
        CLILauncher,
        HeadlessLauncher,
        DependencyChecker,
        EnvironmentSetup,
        LauncherError,
        LauncherValidationError,
        LauncherExecutionError
    )
    from platforms import PlatformDetector
    CORE_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"❌ Core launcher modules not available: {e}")
    print("Please ensure the refactored launcher architecture is properly installed.")
    print(f"Current directory: {current_dir}")
    print(f"Python path: {sys.path[:3]}...")  # Show first few paths
    CORE_MODULES_AVAILABLE = False


class UniversalLauncher:
    """
    Universal launcher that dispatches to appropriate launcher modes.

    This class replaces the functionality of the original 3 separate launchers
    with a unified, well-architected system.
    """

    def __init__(self):
        """Initialize the universal launcher."""
        self.platform_detector = PlatformDetector()
        self.dependency_checker = DependencyChecker()
        self.environment_setup = EnvironmentSetup()

        # Available launcher modes
        self.launchers = {
            'gui': GUILauncher,
            'cli': CLILauncher,
            'headless': HeadlessLauncher,
            'analysis-only': HeadlessLauncher,  # Alias for headless
        }

        # Default mode is GUI
        self.default_mode = 'gui'

    def detect_mode(self, args: argparse.Namespace) -> str:
        """
        Detect which launcher mode to use based on arguments.

        Args:
            args: Parsed command-line arguments

        Returns:
            Mode string ('gui', 'cli', 'headless')
        """
        # Explicit mode selection
        if hasattr(args, 'cli') and args.cli:
            return 'cli'
        if hasattr(args, 'analysis_only') and args.analysis_only:
            return 'headless'
        if hasattr(args, 'mode') and args.mode:
            return args.mode.lower()

        # Component-specific modes (backward compatibility)
        if hasattr(args, 'cube_editor') and args.cube_editor:
            return 'headless'  # Cube editor is handled within headless mode
        if hasattr(args, 'human_positioner') and args.human_positioner:
            return 'headless'
        if hasattr(args, 'visualize_results') and args.visualize_results:
            return 'headless'

        # GUI availability check
        if not self.platform_detector.get_platform_config()['supports_gui']:
            print("⚠️  GUI not available, falling back to CLI mode")
            return 'cli'

        # Default to GUI
        return self.default_mode

    def create_launcher(self, mode: str) -> Optional[object]:
        """
        Create launcher instance for the specified mode.

        Args:
            mode: Launcher mode

        Returns:
            Launcher instance or None if mode not supported
        """
        if mode not in self.launchers:
            print(f"❌ Unsupported launcher mode: {mode}")
            print(f"Available modes: {', '.join(self.launchers.keys())}")
            return None

        try:
            launcher_class = self.launchers[mode]
            return launcher_class()
        except Exception as e:
            print(f"❌ Failed to create {mode} launcher: {e}")
            return None

    def launch(self, args: argparse.Namespace) -> int:
        """
        Launch the application in the appropriate mode.

        Args:
            args: Parsed command-line arguments

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        try:
            # Print application header
            self._print_header()

            # Detect launcher mode
            mode = self.detect_mode(args)
            print(f"🎯 Mode: {mode}")

            # Create launcher
            launcher = self.create_launcher(mode)
            if launcher is None:
                return 1

            # Prepare launch parameters based on mode and arguments
            launch_params = self._prepare_launch_params(args, mode)

            # Launch the application
            try:
                exit_code = launcher.launch(**launch_params)

                # Cleanup regardless of success/failure
                launcher.cleanup()

                return exit_code

            except KeyboardInterrupt:
                print("\n⏹️  Application interrupted by user")
                launcher.cleanup()
                return 130  # Standard exit code for SIGINT

        except Exception as e:
            print(f"❌ Universal launcher failed: {e}")
            return 1

    def _prepare_launch_params(self, args: argparse.Namespace, mode: str) -> Dict[str, Any]:
        """
        Prepare launch parameters based on arguments and mode.

        Args:
            args: Command-line arguments
            mode: Launcher mode

        Returns:
            Dictionary of launch parameters
        """
        params = {}

        # Common parameters
        if hasattr(args, 'file') and args.file:
            params['file_path'] = args.file

        if hasattr(args, 'output_dir') and args.output_dir:
            params['output_dir'] = args.output_dir

        # Window size parameters
        if hasattr(args, 'window_width') and args.window_width:
            params['window_width'] = args.window_width
        if hasattr(args, 'window_height') and args.window_height:
            params['window_height'] = args.window_height

        # Analysis-specific parameters
        if hasattr(args, 'enable_cube') and args.enable_cube:
            params['enable_cube_selection'] = args.enable_cube
        if hasattr(args, 'disable_ptv3') and args.disable_ptv3:
            params['disable_ptv3'] = args.disable_ptv3
        if hasattr(args, 'no_save') and args.no_save:
            params['save_output'] = not args.no_save

        # Visualization parameters
        if hasattr(args, 'visualization') and args.visualization:
            params['visualization_mode'] = args.visualization

        return params

    def _print_header(self) -> None:
        """Print application header."""
        print("=" * 70)
        print("LIDAR Point Cloud Analysis Suite")
        print("Universal Launcher (Refactored Architecture)")
        print("=" * 70)

        # Show platform information
        platform_info = self.platform_detector.get_platform_config()
        print(f"🖥️  Platform: {platform_info['system']} ({platform_info['architecture']})")

        if self.platform_detector.is_macos:
            print("🍎 macOS optimizations will be applied")


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create the argument parser with all supported options.
    Maintains backward compatibility with original launchers.

    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description="LIDAR Point Cloud Analysis Suite - Universal Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Modes:
  GUI (default)     - Full graphical interface with interactive tools
  CLI               - Interactive command-line interface
  Headless          - Automated processing without GUI (requires --file)

Mode Selection:
  --gui             - Launch GUI mode (default)
  --cli             - Launch interactive CLI mode
  --analysis-only   - Launch headless mode (requires --file)
  --mode MODE       - Explicitly specify mode (gui|cli|headless)

Component-Specific Modes (Legacy Compatibility):
  --cube-editor     - Interactive cube selection (requires --file)
  --human-positioner - Human model positioning
  --visualize-results - View saved analysis results

Examples:
  %(prog)s                                    # Launch GUI (default)
  %(prog)s --cli                              # Launch CLI
  %(prog)s --analysis-only --file data.e57   # Headless analysis
  %(prog)s --cube-editor --file data.e57     # Cube selection
  %(prog)s --human-positioner                # Position human model
  %(prog)s --visualize-results               # View results
        """
    )

    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--gui", action="store_true",
                           help="Launch GUI mode (default)")
    mode_group.add_argument("--cli", action="store_true",
                           help="Launch interactive command-line interface")
    mode_group.add_argument("--analysis-only", action="store_true",
                           help="Run analysis without GUI (requires --file)")
    mode_group.add_argument("--mode", choices=['gui', 'cli', 'headless'],
                           help="Explicitly specify launcher mode")

    # Component-specific modes (backward compatibility)
    mode_group.add_argument("--cube-editor", action="store_true",
                           help="Launch cube editor for interior selection")
    mode_group.add_argument("--human-positioner", action="store_true",
                           help="Launch human model positioner")
    mode_group.add_argument("--visualize-results", action="store_true",
                           help="Launch results visualization")

    # File input/output
    parser.add_argument("--file", type=str,
                       help="Path to point cloud file (E57, PCD, or ROS bag)")
    parser.add_argument("--output-dir", default="output",
                       help="Output directory for results (default: output)")

    # Window configuration
    parser.add_argument("--window-width", type=int, default=1600,
                       help="Visualization window width in pixels (default: 1600)")
    parser.add_argument("--window-height", type=int, default=1200,
                       help="Visualization window height in pixels (default: 1200)")

    # Analysis options
    parser.add_argument("--visualization",
                       choices=["combined", "detailed", "focused", "all", "none"],
                       default="combined",
                       help="Visualization mode (default: combined)")
    parser.add_argument("--enable-cube", action="store_true",
                       help="Enable interactive cube selection")
    parser.add_argument("--disable-ptv3", action="store_true",
                       help="Disable PointTransformerV3 enhanced analysis")
    parser.add_argument("--no-save", action="store_true",
                       help="Don't save output files")

    # Help and information
    parser.add_argument("--version", action="version", version="LIDAR Suite v2.0 (Refactored)")

    return parser


def main() -> int:
    """
    Main entry point for the universal launcher.

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    # Check if core modules are available
    if not CORE_MODULES_AVAILABLE:
        print("💡 Fallback: Try running the original launcher:")
        print("  python src/launcher_original_backup.py")
        return 1

    try:
        # Parse arguments
        parser = create_argument_parser()
        args = parser.parse_args()

        # Create and run universal launcher
        launcher = UniversalLauncher()
        return launcher.launch(args)

    except KeyboardInterrupt:
        print("\n⏹️  Launcher interrupted by user")
        return 130
    except Exception as e:
        print(f"❌ Launcher failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())