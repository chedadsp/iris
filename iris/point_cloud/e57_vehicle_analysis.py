#!/usr/bin/env python3
"""
E57 Vehicle Analysis Tool

Clean, modular implementation using pipeline architecture.
Analyzes E57 LIDAR files to identify and extract vehicle interior points.

Author: Dimitrije Stojanovic
Date: September 2025
"""

import argparse
from pathlib import Path
from typing import Optional, Dict, Any

from .config import AnalysisConfig, VTKConfig
from .models.point_cloud_data import AnalysisResults
from .pipeline.analysis_pipeline import AnalysisPipeline
from .processors.preprocessor import PointCloudPreprocessor
from .processors.ground_separator import GroundSeparator
from .processors.vehicle_identifier import VehicleIdentifier
from .processors.interior_detector import InteriorDetector
from .processors.cockpit_refiner import CockpitRefiner
from .processors.ai_analyzer import AIAnalyzer
from .error_handling import PointCloudError, validate_file_path


class E57VehicleAnalyzer:
    """
    Refactored E57 Vehicle Analyzer using modular pipeline architecture.

    This class orchestrates the analysis pipeline and provides a clean interface
    for vehicle analysis while maintaining backward compatibility with the
    original API.
    """

    def __init__(self,
                 file_path: str,
                 config: Optional[AnalysisConfig] = None,
                 window_size: tuple = VTKConfig.DEFAULT_WINDOW_SIZE,
                 enable_ptv3: bool = True):
        """
        Initialize the analyzer.

        Args:
            file_path: Path to the point cloud file
            config: Analysis configuration (uses defaults if None)
            window_size: PyVista window size for visualization
            enable_ptv3: Whether to enable PointTransformerV3 AI analysis
        """
        # Validate file path
        self.file_path = validate_file_path(
            file_path,
            must_exist=True,
            allowed_extensions=['.e57', '.pcd', '.bag']
        )

        self.config = config or AnalysisConfig()
        self.window_size = window_size
        self.enable_ptv3 = enable_ptv3

        # Create analysis pipeline
        self.pipeline = self._create_pipeline()

        # Results storage
        self.results: Optional[AnalysisResults] = None

        print(f"Initialized E57VehicleAnalyzer for: {self.file_path}")
        print(f"Pipeline steps: {', '.join(self.pipeline.get_step_names())}")

    def _create_pipeline(self) -> AnalysisPipeline:
        """Create the analysis pipeline with all processing steps."""
        pipeline = AnalysisPipeline(self.config)

        # Add core processing steps
        pipeline.add_step(PointCloudPreprocessor(self.config))
        pipeline.add_step(GroundSeparator(self.config))
        pipeline.add_step(VehicleIdentifier(self.config))
        pipeline.add_step(InteriorDetector(self.config))
        pipeline.add_step(CockpitRefiner(self.config))

        # Add AI analysis if enabled
        if self.enable_ptv3:
            pipeline.add_step(AIAnalyzer(self.config, enable_ptv3=True))

        return pipeline

    def run_analysis(self,
                     save_output: bool = True,
                     visualization_mode: str = "combined",
                     enable_cube_selection: bool = False,
                     output_dir: str = "output") -> AnalysisResults:
        """
        Run the complete analysis pipeline.

        Args:
            save_output: Whether to save results to files
            visualization_mode: Visualization type ('combined', 'detailed', 'focused', 'all')
            enable_cube_selection: Whether to enable interactive cube selection
            output_dir: Directory to save output files

        Returns:
            Complete analysis results
        """
        try:
            print(f"Starting analysis of: {self.file_path}")

            # Handle cube selection if enabled
            if enable_cube_selection:
                self._handle_cube_selection()

            # Run the analysis pipeline
            self.results = self.pipeline.run(self.file_path)

            # Save results if requested
            if save_output:
                self._save_results(output_dir)

            # Show visualization if requested
            if visualization_mode:
                self._visualize_results(visualization_mode)

            print("✅ Analysis completed successfully")
            return self.results

        except Exception as e:
            raise PointCloudError(f"Analysis failed: {e}") from e

    def get_results_summary(self) -> Dict[str, Any]:
        """Get a summary of analysis results."""
        if self.results is None:
            return {"status": "No analysis performed"}

        return self.results.get_summary()

    def set_window_size(self, width: int, height: int) -> None:
        """Set the PyVista window size."""
        self.window_size = (width, height)
        print(f"PyVista window size set to: {width}x{height} pixels")

    def _handle_cube_selection(self) -> None:
        """Handle interactive cube selection if enabled."""
        try:
            from .interactive_cube_selector import InteractiveCubeSelector

            print("Starting with interactive cube selection...")

            # Load points first for cube selection
            from .loaders.file_loader_factory import load_point_cloud_file
            data = load_point_cloud_file(str(self.file_path))

            cube_selector = InteractiveCubeSelector(data.points, self.window_size)
            selected_points = cube_selector.select_cube_region()

            if selected_points is not None:
                print(f"Cube selection completed: {len(selected_points):,} points selected")
                print("Saved to 'interior_cockpit.npy'")
            else:
                print("No cube selection made, continuing with standard analysis...")

        except ImportError:
            print("⚠️  Interactive cube selector not available")
        except Exception as e:
            print(f"⚠️  Cube selection failed: {e}")

    def _save_results(self, output_dir: str) -> None:
        """Save analysis results to files."""
        if self.results is None:
            return

        import numpy as np
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        saved_files = []

        # Save each result type
        if self.results.ground_points is not None:
            file_path = output_path / "ground_points.npy"
            np.save(file_path, self.results.ground_points)
            saved_files.append(file_path)

        if self.results.non_ground_points is not None:
            file_path = output_path / "non_ground_points.npy"
            np.save(file_path, self.results.non_ground_points)
            saved_files.append(file_path)

        if self.results.vehicle_points is not None:
            file_path = output_path / "vehicle_points.npy"
            np.save(file_path, self.results.vehicle_points)
            saved_files.append(file_path)

        if self.results.interior_points is not None:
            file_path = output_path / "interior_points.npy"
            np.save(file_path, self.results.interior_points)
            saved_files.append(file_path)

        if self.results.human_points is not None:
            file_path = output_path / "human_points.npy"
            np.save(file_path, self.results.human_points)
            saved_files.append(file_path)

        print(f"Saved {len(saved_files)} result files to {output_dir}/")

    def _visualize_results(self, mode: str) -> None:
        """Create visualization of results."""
        if self.results is None:
            print("No results available for visualization")
            return

        try:
            # Import visualization components (would be implemented in Phase 7)
            print(f"Visualization mode '{mode}' requested")
            print("⚠️  Visualization components not yet implemented in this refactor")
            # TODO: Implement visualization manager

        except ImportError:
            print("⚠️  Visualization components not available")
        except Exception as e:
            print(f"⚠️  Visualization failed: {e}")

    # Backward compatibility methods
    def load_point_cloud_file(self):
        """Backward compatibility: load point cloud file."""
        if self.results is None:
            # Load data using the pipeline's file loader
            from .loaders.file_loader_factory import load_point_cloud_file
            data = load_point_cloud_file(str(self.file_path))
            self.results = AnalysisResults(raw_data=data)
        return self.results.raw_data

    def preprocess_points(self):
        """Backward compatibility: preprocess points."""
        if self.results is None:
            self.load_point_cloud_file()

        from .processors.preprocessor import PointCloudPreprocessor
        preprocessor = PointCloudPreprocessor(self.config)
        self.results = preprocessor.analyze(self.results)

    def separate_ground_points(self):
        """Backward compatibility: separate ground points."""
        if self.results is None or not hasattr(self.results.raw_data, 'points'):
            self.preprocess_points()

        from .processors.ground_separator import GroundSeparator
        separator = GroundSeparator(self.config)
        self.results = separator.analyze(self.results)

    def identify_vehicle_points(self):
        """Backward compatibility: identify vehicle points."""
        if self.results is None or self.results.ground_points is None:
            self.separate_ground_points()

        from .processors.vehicle_identifier import VehicleIdentifier
        identifier = VehicleIdentifier(self.config)
        self.results = identifier.analyze(self.results)

    def find_vehicle_interior(self):
        """Backward compatibility: find vehicle interior."""
        if self.results is None or self.results.vehicle_points is None:
            self.identify_vehicle_points()

        from .processors.interior_detector import InteriorDetector
        from .processors.cockpit_refiner import CockpitRefiner

        detector = InteriorDetector(self.config)
        self.results = detector.analyze(self.results)

        refiner = CockpitRefiner(self.config)
        self.results = refiner.analyze(self.results)

    # Backward compatibility properties
    @property
    def points(self):
        """Backward compatibility: raw points."""
        return self.results.raw_data.points if self.results else None

    @property
    def colors(self):
        """Backward compatibility: colors."""
        return self.results.raw_data.colors if self.results else None

    @property
    def intensity(self):
        """Backward compatibility: intensity."""
        return self.results.raw_data.intensity if self.results else None

    @property
    def ground_points(self):
        """Backward compatibility: ground points."""
        return self.results.ground_points if self.results else None

    @property
    def non_ground_points(self):
        """Backward compatibility: non-ground points."""
        return self.results.non_ground_points if self.results else None

    @property
    def vehicle_points(self):
        """Backward compatibility: vehicle points."""
        return self.results.vehicle_points if self.results else None

    @property
    def interior_points(self):
        """Backward compatibility: interior points."""
        return self.results.interior_points if self.results else None


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Refactored E57 Vehicle Analysis Tool"
    )
    parser.add_argument("file_path", help="Path to the point cloud file")
    parser.add_argument("--no-save", action="store_true",
                       help="Don't save output files")
    parser.add_argument("--output-dir", default="output",
                       help="Output directory for results")
    parser.add_argument("--window-width", type=int, default=2560,
                       help="PyVista window width in pixels")
    parser.add_argument("--window-height", type=int, default=1440,
                       help="PyVista window height in pixels")
    parser.add_argument("--visualization", choices=["combined", "detailed", "focused", "all"],
                       default="combined", help="Visualization mode")
    parser.add_argument("--enable-cube", action="store_true",
                       help="Enable interactive cube selection")
    parser.add_argument("--disable-ptv3", action="store_true",
                       help="Disable PointTransformerV3 enhanced analysis")

    args = parser.parse_args()

    # Parse window size
    window_size = (args.window_width, args.window_height)

    # Validate file exists
    if not Path(args.file_path).exists():
        print(f"Error: File {args.file_path} does not exist")
        return 1

    print(f"Using PyVista window size: {window_size[0]}x{window_size[1]} pixels")

    try:
        # Create and run analyzer
        analyzer = E57VehicleAnalyzer(
            args.file_path,
            window_size=window_size,
            enable_ptv3=not args.disable_ptv3
        )

        results = analyzer.run_analysis(
            save_output=not args.no_save,
            visualization_mode=args.visualization,
            enable_cube_selection=args.enable_cube,
            output_dir=args.output_dir
        )

        # Print summary
        summary = analyzer.get_results_summary()
        print("\n📊 Analysis Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")

        return 0

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())