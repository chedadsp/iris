#!/usr/bin/env python3
"""
Output Visualization Tool

This script visualizes all the output files from the E57 vehicle analysis,
including ground points, non-ground points, vehicle points, and interior points.

Author: Dimitrije Stojanovic
Date: September 2025
"""

import argparse
import numpy as np
import pyvista as pv
from pathlib import Path
import matplotlib.pyplot as plt


class OutputVisualizer:
    """Visualizes analysis results from saved numpy files.

    The loader avoids hardcoded filenames and detects files by keywords.
    You can also override detections via explicit filenames.
    """

    def __init__(
        self,
        output_dir: str = "output",
        ground_file: str | None = None,
        non_ground_file: str | None = None,
        vehicle_file: str | None = None,
        interior_file: str | None = None,
    ):
        self.output_dir = Path(output_dir)
        self._override = {
            "ground": Path(ground_file) if ground_file else None,
            "non_ground": Path(non_ground_file) if non_ground_file else None,
            "vehicle": Path(vehicle_file) if vehicle_file else None,
            "interior": Path(interior_file) if interior_file else None,
        }

        self.ground_points = None
        self.non_ground_points = None
        self.vehicle_points = None
        self.interior_points = None

    def _detect_file(self, files: list[Path], *, include: list[str], exclude: list[str] | None = None) -> Path | None:
        """Return first file whose name contains all include keywords and none of the exclude keywords."""
        exclude = exclude or []
        for f in files:
            name = f.name.lower()
            if all(k in name for k in include) and all(k not in name for k in exclude):
                return f
        return None

    def _pick_largest(self, files: list[Path]) -> Path | None:
        return max(files, key=lambda p: p.stat().st_size) if files else None

    def load_output_files(self):
        """Load all available output files using robust name detection."""
        print(f"Loading output files from: {self.output_dir}")

        if not self.output_dir.exists():
            print(f"Output directory {self.output_dir} does not exist!")
            return False

        # List all .npy files
        npy_files = list(self.output_dir.glob("*.npy"))
        if not npy_files:
            print(f"No .npy files found in {self.output_dir}")
            return False

        print(f"Found {len(npy_files)} output files:")
        for file in npy_files:
            print(f"  - {file.name}")

        # Resolve files either via overrides or keyword detection
        resolved: dict[str, Path | None] = {
            "ground": self._override["ground"],
            "non_ground": self._override["non_ground"],
            "vehicle": self._override["vehicle"],
            "interior": self._override["interior"],
        }

        # Detection rules (case-insensitive)
        if resolved["interior"] is None:
            resolved["interior"] = (
                self._detect_file(npy_files, include=["interior"], exclude=None)
                or self._detect_file(npy_files, include=["cockpit"], exclude=None)
                or self._detect_file(npy_files, include=["inside"], exclude=None)
                or self._detect_file(npy_files, include=["passenger"], exclude=None)
            )

        if resolved["vehicle"] is None:
            resolved["vehicle"] = self._detect_file(npy_files, include=["vehicle"])  # e.g., vehicle_points.npy

        if resolved["non_ground"] is None:
            resolved["non_ground"] = (
                self._detect_file(npy_files, include=["non", "ground"])  # non_ground_points.npy
                or self._detect_file(npy_files, include=["nonground"])    # nonground.npy
            )

        if resolved["ground"] is None:
            # ground but not non-ground
            resolved["ground"] = self._detect_file(npy_files, include=["ground"], exclude=["non"])  # ground_points.npy

        # As a final fallback for "ground" (original/all points), pick the largest file not already used
        used = {p.resolve() for p in resolved.values() if p is not None}
        remaining = [p for p in npy_files if p.resolve() not in used]
        if resolved["ground"] is None and remaining:
            fallback = self._pick_largest(remaining)
            if fallback:
                print(f"No explicit ground file found. Using largest file as fallback: {fallback.name}")
                resolved["ground"] = fallback

        # Load arrays if files resolved
        if resolved["ground"] and resolved["ground"].exists():
            self.ground_points = np.load(resolved["ground"])
            print(f"Loaded {len(self.ground_points)} points from '{resolved['ground'].name}' as Ground/All points")

        if resolved["non_ground"] and resolved["non_ground"].exists():
            self.non_ground_points = np.load(resolved["non_ground"]) 
            print(f"Loaded {len(self.non_ground_points)} points from '{resolved['non_ground'].name}' as Non-ground points")

        if resolved["vehicle"] and resolved["vehicle"].exists():
            self.vehicle_points = np.load(resolved["vehicle"]) 
            print(f"Loaded {len(self.vehicle_points)} points from '{resolved['vehicle'].name}' as Vehicle points")

        if resolved["interior"] and resolved["interior"].exists():
            self.interior_points = np.load(resolved["interior"]) 
            print(f"Loaded {len(self.interior_points)} points from '{resolved['interior'].name}' as Interior points")
        else:
            print("No interior points file found")

        return True
    
    def print_statistics(self):
        """Print detailed statistics about the loaded data."""
        print("\n" + "="*50)
        print("ANALYSIS RESULTS STATISTICS")
        print("="*50)
        
        def print_point_stats(points, name):
            if points is not None and len(points) > 0:
                print(f"\n{name}:")
                print(f"  Count: {len(points):,} points")
                print(f"  X range: {points[:, 0].min():.2f} to {points[:, 0].max():.2f} m")
                print(f"  Y range: {points[:, 1].min():.2f} to {points[:, 1].max():.2f} m")
                print(f"  Z range: {points[:, 2].min():.2f} to {points[:, 2].max():.2f} m")
                print(f"  Dimensions: {points[:, 0].max() - points[:, 0].min():.2f} x "
                      f"{points[:, 1].max() - points[:, 1].min():.2f} x "
                      f"{points[:, 2].max() - points[:, 2].min():.2f} m")
            else:
                print(f"\n{name}: No data")
        
        print_point_stats(self.ground_points, "Ground Points")
        print_point_stats(self.non_ground_points, "Non-Ground Points")
        print_point_stats(self.vehicle_points, "Vehicle Points")
        print_point_stats(self.interior_points, "Interior/Cockpit Points")
        
        # Additional analysis
        if self.vehicle_points is not None and self.interior_points is not None:
            if len(self.interior_points) > 0:
                interior_ratio = len(self.interior_points) / len(self.vehicle_points) * 100
                print(f"\nInterior Detection Ratio: {interior_ratio:.1f}% of vehicle points")
            else:
                print(f"\nInterior Detection: No interior points found")
    
    def create_3d_visualization(self, plotter_callback=None):
        """Create interactive 3D visualization using PyVista.
        
        Args:
            plotter_callback: Optional callback function to register the plotter for tracking/cleanup
        """
        print("Creating 3D visualization...")
        
        plotter = None
        try:
            # Create plotter
            plotter = pv.Plotter(shape=(2, 2))
            
            # Register plotter with callback if provided (for GUI integration)
            if plotter_callback and callable(plotter_callback):
                plotter_callback(plotter)
            
            # Plot 1: Ground points
            plotter.subplot(0, 0)
            if self.ground_points is not None:
                ground_cloud = pv.PolyData(self.ground_points)
                plotter.add_mesh(ground_cloud, color='gray', point_size=1, opacity=0.6)
            plotter.add_title("Original Points", font_size=8)
            
            # Plot 2: Vehicle structure
            plotter.subplot(0, 1)
            if self.vehicle_points is not None:
                vehicle_cloud = pv.PolyData(self.vehicle_points)
                plotter.add_mesh(vehicle_cloud, color='lightblue', point_size=2)
            plotter.add_title("Vehicle", font_size=8)
            
            # Plot 3: Interior/Cockpit
            plotter.subplot(1, 0)
            if self.interior_points is not None and len(self.interior_points) > 0:
                interior_cloud = pv.PolyData(self.interior_points)
                plotter.add_mesh(interior_cloud, color='red', point_size=2)
                plotter.add_title("Cockpit/Passenger Compartment", font_size=8)
            else:
                plotter.add_title("Cockpit/Passenger Compartment (None Found)")
            
            # Plot 4: Complete scene
            plotter.subplot(1, 1)
            if self.ground_points is not None:
                ground_cloud = pv.PolyData(self.ground_points)
                plotter.add_mesh(ground_cloud, color='grey', point_size=1, opacity=0.3)
            
            if self.vehicle_points is not None:
                vehicle_cloud = pv.PolyData(self.vehicle_points)
                plotter.add_mesh(vehicle_cloud, color='lightblue', point_size=1.5, opacity=0.7)

            if self.interior_points is not None and len(self.interior_points) > 0:
                interior_cloud = pv.PolyData(self.interior_points)
                plotter.add_mesh(interior_cloud, color='red', point_size=3)
            
            plotter.add_title("Complete Analysis", font_size=8)
            
            # Show the visualization
            plotter.show(auto_close=False, interactive_update=True)
            
        except Exception as e:
            # If we have a plotter and no callback (standalone mode), clean it up
            if plotter is not None and plotter_callback is None:
                try:
                    plotter.close()
                except:
                    pass
                finally:
                    del plotter
            # Re-raise the exception for proper error handling
            raise e
    
    def run_visualization(self, show_2d=True, show_height=True, show_3d=True):
        """Run the complete visualization pipeline."""
        if not self.load_output_files():
            return
        
        self.print_statistics()
                
        if show_3d:
            print("\nShowing 3D visualization...")
            self.create_3d_visualization()


def main():
    """Main function to run the output visualization."""
    parser = argparse.ArgumentParser(description="Visualize E57 vehicle analysis output files")
    parser.add_argument("--output-dir", default="output", help="Output directory to read from")
    # Optional explicit file overrides (filenames within output dir or full paths)
    parser.add_argument("--ground-file", default=None, help="Override ground/all points file name")
    parser.add_argument("--non-ground-file", default=None, help="Override non-ground points file name")
    parser.add_argument("--vehicle-file", default=None, help="Override vehicle points file name")
    parser.add_argument("--interior-file", default=None, help="Override interior/cockpit points file name")
    parser.add_argument("--no-3d", action="store_true", help="Skip 3D visualization")
    
    args = parser.parse_args()
    
    # Resolve potential full paths vs relative names for overrides
    def resolve_override(path_like: str | None):
        if not path_like:
            return None
        p = Path(path_like)
        return p if p.is_absolute() else (Path(args.output_dir) / p)

    visualizer = OutputVisualizer(
        args.output_dir,
        ground_file=resolve_override(args.ground_file),
        non_ground_file=resolve_override(args.non_ground_file),
        vehicle_file=resolve_override(args.vehicle_file),
        interior_file=resolve_override(args.interior_file),
    )
    
    # Run visualization
    visualizer.run_visualization(
        show_3d=not args.no_3d
    )


if __name__ == "__main__":
    main()
