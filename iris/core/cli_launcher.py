#!/usr/bin/env python3
"""
CLI Launcher Mode

Handles launching the application in CLI (Command Line Interface) mode,
providing an interactive command-line interface for LIDAR processing.
"""

from typing import Tuple, Optional, Dict, Any
from pathlib import Path
import sys

from .launcher_base import BaseLauncher, LauncherValidationError, LauncherExecutionError
from .dependency_checker import DependencyChecker
from .environment_setup import EnvironmentSetup
import sys
from pathlib import Path
# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from platforms import PlatformDetector


class CLILauncher(BaseLauncher):
    """Launcher for CLI mode."""

    def __init__(self, **kwargs):
        """Initialize CLI launcher."""
        super().__init__(**kwargs)
        self.dependency_checker = DependencyChecker()
        self.environment_setup = EnvironmentSetup()
        self.platform_detector = PlatformDetector()

    def validate_requirements(self) -> Tuple[bool, str]:
        """
        Validate requirements for CLI mode.

        Returns:
            Tuple of (success, error_message)
        """
        try:
            # Check CLI-specific dependencies
            success, missing_required, missing_optional = self.dependency_checker.check_mode_dependencies('cli')

            if not success:
                error_msg = f"Missing required dependencies for CLI mode: {', '.join(missing_required)}"
                suggestions = self.dependency_checker.get_installation_suggestions(missing_required)
                if suggestions:
                    error_msg += "\n" + "\n".join(suggestions)
                return False, error_msg

            # Warn about missing optional dependencies
            if missing_optional:
                print(f"⚠️  Missing optional dependencies: {', '.join(missing_optional)}")
                print("Some CLI features may be disabled.")

            return True, ""

        except Exception as e:
            return False, f"CLI validation error: {e}"

    def launch(self, **kwargs) -> int:
        """
        Launch CLI mode.

        Args:
            **kwargs: Additional parameters for CLI mode

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
            print("📋 Setting up CLI environment...")
            self.environment_setup.setup_cli_environment()

            # Print platform information
            self.platform_detector.print_platform_summary()

            # Launch CLI interface
            print("🖥️  Launching CLI interface...")
            return self._launch_cli_interface()

        except Exception as e:
            return self.handle_error(e, "CLI launch")

    def _launch_cli_interface(self) -> int:
        """
        Launch the interactive CLI interface.

        Returns:
            Exit code from CLI interface
        """
        try:
            print("=" * 60)
            print("LIDAR Point Cloud Analysis Suite - Interactive CLI")
            print("=" * 60)
            print()

            # Show available commands
            self._show_welcome_message()

            # Start interactive loop
            return self._interactive_loop()

        except KeyboardInterrupt:
            print("\n⏹️  CLI session interrupted by user")
            return 0
        except Exception as e:
            raise LauncherExecutionError(f"CLI interface failed: {e}") from e

    def _show_welcome_message(self) -> None:
        """Show welcome message and available commands."""
        print("Welcome to the LIDAR CLI interface!")
        print()
        print("Available commands:")
        print("  analyze <file>     - Analyze point cloud file (E57, PCD, bag)")
        print("  visualize <file>   - Visualize point cloud file")
        print("  cube-edit <file>   - Interactive cube selection")
        print("  human-pos          - Human model positioning")
        print("  results            - View saved analysis results")
        print("  info               - Show system information")
        print("  help               - Show this help message")
        print("  exit               - Exit CLI")
        print()
        print("Type a command or 'help' for more information.")
        print()

    def _interactive_loop(self) -> int:
        """
        Main interactive command loop.

        Returns:
            Exit code
        """
        while True:
            try:
                # Get user input
                command = input("lidar> ").strip().lower()

                if not command:
                    continue

                # Parse command
                parts = command.split()
                cmd = parts[0]
                args = parts[1:] if len(parts) > 1 else []

                # Execute command
                if cmd in ['exit', 'quit', 'q']:
                    print("👋 Goodbye!")
                    return 0
                elif cmd in ['help', 'h', '?']:
                    self._show_help()
                elif cmd == 'info':
                    self._show_system_info()
                elif cmd == 'analyze':
                    self._handle_analyze_command(args)
                elif cmd == 'visualize':
                    self._handle_visualize_command(args)
                elif cmd == 'cube-edit':
                    self._handle_cube_edit_command(args)
                elif cmd == 'human-pos':
                    self._handle_human_pos_command(args)
                elif cmd == 'results':
                    self._handle_results_command(args)
                else:
                    print(f"❌ Unknown command: {cmd}")
                    print("Type 'help' for available commands.")

            except KeyboardInterrupt:
                print("\n⏹️  Use 'exit' to quit")
                continue
            except EOFError:
                print("\n👋 Goodbye!")
                return 0
            except Exception as e:
                print(f"❌ Command error: {e}")
                continue

    def _show_help(self) -> None:
        """Show detailed help information."""
        print()
        print("LIDAR CLI Commands:")
        print("=" * 40)
        print()
        print("analyze <file>")
        print("  Perform complete vehicle analysis on point cloud file")
        print("  Supports: E57, PCD, ROS bag files")
        print("  Example: analyze data/scan.e57")
        print()
        print("visualize <file>")
        print("  Visualize raw point cloud file")
        print("  Example: visualize data/scan.e57")
        print()
        print("cube-edit <file>")
        print("  Interactive cube selection for interior extraction")
        print("  Example: cube-edit data/scan.e57")
        print()
        print("human-pos")
        print("  Position human model in interior points")
        print("  Requires: output/interior_cockpit.npy")
        print()
        print("results")
        print("  View and visualize saved analysis results")
        print()
        print("info")
        print("  Show system and platform information")
        print()
        print("help, exit")
        print("  Show this help or exit CLI")
        print()

    def _show_system_info(self) -> None:
        """Show system information."""
        print()
        print("System Information:")
        print("=" * 30)
        self.platform_detector.print_platform_summary()
        self.platform_detector.warn_about_limitations()

        print()
        print("Dependency Status:")
        print("-" * 20)
        self.dependency_checker.print_dependency_report('cli')

    def _handle_analyze_command(self, args: list) -> None:
        """Handle analyze command."""
        if not args:
            print("❌ Please specify a file to analyze")
            print("Usage: analyze <file>")
            return

        file_path = Path(args[0])
        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            return

        try:
            print(f"🔬 Starting analysis of: {file_path}")

            # Import and use the headless analyzer
            from point_cloud.e57_vehicle_analysis import E57VehicleAnalyzer

            analyzer = E57VehicleAnalyzer(str(file_path))
            results = analyzer.run_analysis(
                save_output=True,
                visualization_mode=None,  # No visualization in CLI
                enable_cube_selection=False
            )

            # Print summary
            summary = analyzer.get_results_summary()
            print("\n📊 Analysis Summary:")
            for key, value in summary.items():
                print(f"  {key}: {value}")

            print("✅ Analysis completed successfully")

        except Exception as e:
            print(f"❌ Analysis failed: {e}")

    def _handle_visualize_command(self, args: list) -> None:
        """Handle visualize command."""
        if not args:
            print("❌ Please specify a file to visualize")
            print("Usage: visualize <file>")
            return

        file_path = Path(args[0])
        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            return

        try:
            print(f"👁️  Visualizing: {file_path}")

            # Import visualization module
            from point_cloud.visualise_raw_lidar_files import main as visualize_main

            # Call the visualization function (this will open a GUI window)
            # Note: This switches to GUI mode temporarily
            import sys
            old_argv = sys.argv
            sys.argv = ['visualize', str(file_path)]

            try:
                visualize_main()
                print("✅ Visualization completed")
            finally:
                sys.argv = old_argv

        except Exception as e:
            print(f"❌ Visualization failed: {e}")

    def _handle_cube_edit_command(self, args: list) -> None:
        """Handle cube-edit command."""
        if not args:
            print("❌ Please specify a file for cube editing")
            print("Usage: cube-edit <file>")
            return

        file_path = Path(args[0])
        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            return

        try:
            print(f"🎯 Starting cube editor for: {file_path}")
            print("Note: This will open a 3D interface")

            # Import cube selector
            from point_cloud.interactive_cube_selector import CubeSelectionManager
            from point_cloud.visualise_raw_lidar_files import read_points

            # Load points
            points, _, _ = read_points(str(file_path))
            print(f"Loaded {len(points):,} points")

            # Run cube selection
            manager = CubeSelectionManager()
            selected_points = manager.select_from_file(
                points,
                f"cli_{file_path.stem}"
            )

            if selected_points is not None:
                print(f"✅ Cube selection completed: {len(selected_points):,} points selected")
            else:
                print("❌ Cube selection was cancelled")

        except Exception as e:
            print(f"❌ Cube editing failed: {e}")

    def _handle_human_pos_command(self, args: list) -> None:
        """Handle human-pos command."""
        try:
            print("👤 Starting human model positioner...")

            # Import human positioner
            from point_cloud.interactive_human_positioner import InteractiveHumanPositioner

            # Default to interior_cockpit.npy
            input_file = Path("output/interior_cockpit.npy")
            if not input_file.exists():
                print(f"❌ Required file not found: {input_file}")
                print("Please run 'analyze' or 'cube-edit' first to generate interior points.")
                return

            # Create and run positioner
            positioner = InteractiveHumanPositioner(str(input_file))
            positioner.run_interactive_positioning()

            print("✅ Human positioning completed")

        except Exception as e:
            print(f"❌ Human positioning failed: {e}")

    def _handle_results_command(self, args: list) -> None:
        """Handle results command."""
        try:
            print("📊 Loading saved results...")

            # Import results visualizer
            from point_cloud.visualize_output import OutputVisualizer

            visualizer = OutputVisualizer()
            visualizer.load_data()

            print("🎨 Opening results visualization...")
            visualizer.visualize_3d()

            print("✅ Results visualization completed")

        except Exception as e:
            print(f"❌ Results visualization failed: {e}")

    def get_mode_description(self) -> str:
        """Get description of CLI mode."""
        return ("CLI mode provides an interactive command-line interface for LIDAR point cloud "
                "processing with text-based commands and optional 3D visualization.")

    def print_mode_help(self) -> None:
        """Print help information for CLI mode."""
        print("CLI Mode Help:")
        print("=" * 40)
        print(self.get_mode_description())
        print()
        print("Features:")
        print("- Interactive command-line interface")
        print("- File analysis and processing commands")
        print("- System information and status")
        print("- Optional 3D visualization")
        print("- Results viewing and management")
        print()
        print("Usage:")
        print("  python launcher.py --cli")
        print("  python launcher.py --mode cli")