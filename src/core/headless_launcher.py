#!/usr/bin/env python3
"""
Headless Launcher Mode

Handles launching the application in headless/analysis-only mode for
automated processing without GUI or interactive components.
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


class HeadlessLauncher(BaseLauncher):
    """Launcher for headless/analysis-only mode."""

    def __init__(self, **kwargs):
        """Initialize headless launcher."""
        super().__init__(**kwargs)
        self.dependency_checker = DependencyChecker()
        self.environment_setup = EnvironmentSetup()
        self.platform_detector = PlatformDetector()

    def validate_requirements(self) -> Tuple[bool, str]:
        """
        Validate requirements for headless mode.

        Returns:
            Tuple of (success, error_message)
        """
        try:
            # Check headless-specific dependencies
            success, missing_required, missing_optional = self.dependency_checker.check_mode_dependencies('headless')

            if not success:
                error_msg = f"Missing required dependencies for headless mode: {', '.join(missing_required)}"
                suggestions = self.dependency_checker.get_installation_suggestions(missing_required)
                if suggestions:
                    error_msg += "\n" + "\n".join(suggestions)
                return False, error_msg

            # Warn about missing optional dependencies
            if missing_optional:
                print(f"⚠️  Missing optional dependencies: {', '.join(missing_optional)}")
                print("Some analysis features may be disabled.")

            return True, ""

        except Exception as e:
            return False, f"Headless validation error: {e}"

    def launch(self, file_path: str = None, output_dir: str = "output", **kwargs) -> int:
        """
        Launch headless mode.

        Args:
            file_path: Path to the point cloud file to analyze
            output_dir: Directory to save output files
            **kwargs: Additional analysis parameters

        Returns:
            Exit code
        """
        try:
            self.print_mode_info()

            # Validate file path
            if not file_path:
                print("❌ File path is required for headless mode")
                return 1

            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                print(f"❌ File not found: {file_path}")
                return 1

            # Validate requirements
            valid, error_msg = self.validate_requirements()
            if not valid:
                print(f"❌ {error_msg}")
                return 1

            # Set up environment
            print("📋 Setting up headless environment...")
            self.environment_setup.setup_headless_environment()

            # Print platform information (minimal for headless)
            print(f"🖥️  Platform: {self.platform_detector.system_name}")

            # Launch analysis
            print(f"🔬 Starting headless analysis of: {file_path}")
            return self._run_headless_analysis(file_path_obj, output_dir, **kwargs)

        except Exception as e:
            return self.handle_error(e, "headless launch")

    def _run_headless_analysis(self, file_path: Path, output_dir: str, **kwargs) -> int:
        """
        Run headless analysis on the provided file.

        Args:
            file_path: Path to the point cloud file
            output_dir: Output directory
            **kwargs: Additional analysis parameters

        Returns:
            Exit code
        """
        try:
            # Import analyzer
            from point_cloud.e57_vehicle_analysis import E57VehicleAnalyzer

            # Get analysis parameters from kwargs
            window_width = kwargs.get('window_width', 1600)
            window_height = kwargs.get('window_height', 1200)
            enable_cube_selection = kwargs.get('enable_cube_selection', False)
            disable_ptv3 = kwargs.get('disable_ptv3', False)
            save_output = kwargs.get('save_output', True)

            # Create analyzer
            analyzer = E57VehicleAnalyzer(
                str(file_path),
                window_size=(window_width, window_height),
                enable_ptv3=not disable_ptv3
            )

            # Run analysis (no visualization in headless mode)
            print("⚙️  Running analysis pipeline...")
            results = analyzer.run_analysis(
                save_output=save_output,
                visualization_mode=None,  # No visualization
                enable_cube_selection=enable_cube_selection,
                output_dir=output_dir
            )

            # Print comprehensive summary
            self._print_analysis_summary(analyzer, results, file_path, output_dir)

            print("✅ Headless analysis completed successfully")
            return 0

        except KeyboardInterrupt:
            print("\n⏹️  Analysis interrupted by user")
            return 130  # Standard exit code for SIGINT
        except Exception as e:
            raise LauncherExecutionError(f"Headless analysis failed: {e}") from e

    def _print_analysis_summary(self, analyzer, results, file_path: Path, output_dir: str) -> None:
        """
        Print comprehensive analysis summary.

        Args:
            analyzer: The analyzer instance
            results: Analysis results
            file_path: Input file path
            output_dir: Output directory
        """
        print("\n📊 Analysis Summary:")
        print("=" * 50)

        # Input file information
        print(f"Input file: {file_path}")
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        print(f"File size: {file_size_mb:.1f} MB")

        # Analysis results
        summary = analyzer.get_results_summary()
        for key, value in summary.items():
            print(f"{key}: {value}")

        # Output files
        output_path = Path(output_dir)
        if output_path.exists():
            output_files = list(output_path.glob("*.npy"))
            if output_files:
                print(f"\nOutput files ({len(output_files)} files):")
                for file in sorted(output_files):
                    file_size_mb = file.stat().st_size / (1024 * 1024)
                    print(f"  {file.name} ({file_size_mb:.1f} MB)")
            else:
                print("\nNo output files generated.")

        print("=" * 50)

    def get_mode_description(self) -> str:
        """Get description of headless mode."""
        return ("Headless mode performs automated point cloud analysis without GUI or "
                "interactive components, suitable for batch processing and automation.")

    def print_mode_help(self) -> None:
        """Print help information for headless mode."""
        print("Headless Mode Help:")
        print("=" * 40)
        print(self.get_mode_description())
        print()
        print("Features:")
        print("- Automated analysis pipeline")
        print("- No GUI or interactive components")
        print("- Suitable for batch processing")
        print("- Comprehensive result reporting")
        print("- Configurable output directory")
        print()
        print("Required Arguments:")
        print("  --file PATH        Path to point cloud file")
        print()
        print("Optional Arguments:")
        print("  --output-dir DIR   Output directory (default: output)")
        print("  --window-width N   Window width for processing (default: 1600)")
        print("  --window-height N  Window height for processing (default: 1200)")
        print("  --enable-cube      Enable cube selection (default: false)")
        print("  --disable-ptv3     Disable AI enhancement (default: false)")
        print("  --no-save          Don't save output files")
        print()
        print("Usage Examples:")
        print("  python launcher.py --headless --file data/scan.e57")
        print("  python launcher.py --analysis-only --file data/scan.e57 --output-dir results")
        print("  python launcher.py --headless --file scan.pcd --enable-cube")

    def validate_analysis_parameters(self, **kwargs) -> Tuple[bool, str]:
        """
        Validate analysis parameters for headless mode.

        Args:
            **kwargs: Analysis parameters to validate

        Returns:
            Tuple of (valid, error_message)
        """
        try:
            # Check window size parameters
            window_width = kwargs.get('window_width', 1600)
            window_height = kwargs.get('window_height', 1200)

            if not isinstance(window_width, int) or window_width < 100:
                return False, f"Invalid window width: {window_width}"

            if not isinstance(window_height, int) or window_height < 100:
                return False, f"Invalid window height: {window_height}"

            # Check output directory
            output_dir = kwargs.get('output_dir', 'output')
            try:
                output_path = Path(output_dir)
                output_path.mkdir(exist_ok=True)
            except Exception as e:
                return False, f"Cannot create output directory '{output_dir}': {e}"

            return True, ""

        except Exception as e:
            return False, f"Parameter validation error: {e}"