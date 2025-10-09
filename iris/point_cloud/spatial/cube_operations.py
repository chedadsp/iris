#!/usr/bin/env python3
"""
Cube Operations - Pure spatial algorithms for cube-based point selection.

Contains business logic for cube-based spatial operations without GUI dependencies.
Extracted from interactive_cube_selector.py as part of GUI/business logic separation.

Author: Dimitrije Stojanovic
Date: September 2025
Refactored: November 2025 (GUI/Business Logic Separation)
"""

import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass
from ..config import InteractionConfig
from ..spatial_utils import OptimizedPointCloudOps, optimize_point_cloud_filtering


@dataclass
class CubeBounds:
    """Data class for cube bounds with validation."""
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    def __post_init__(self):
        """Validate cube bounds after initialization."""
        if any(np.isnan([self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max])):
            raise ValueError("Cube bounds cannot contain NaN values")

        if self.x_min >= self.x_max or self.y_min >= self.y_max or self.z_min >= self.z_max:
            raise ValueError("Invalid cube bounds: min values must be less than max values")

    def to_list(self) -> List[float]:
        """Convert to list format for compatibility."""
        return [self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max]

    @classmethod
    def from_list(cls, bounds: List[float]) -> 'CubeBounds':
        """Create CubeBounds from list format."""
        if len(bounds) != 6:
            raise ValueError("Bounds list must have exactly 6 elements")
        return cls(*bounds)

    def get_center(self) -> Tuple[float, float, float]:
        """Get the center point of the cube."""
        return (
            (self.x_min + self.x_max) / 2,
            (self.y_min + self.y_max) / 2,
            (self.z_min + self.z_max) / 2
        )

    def get_size(self) -> Tuple[float, float, float]:
        """Get the size of the cube in each dimension."""
        return (
            self.x_max - self.x_min,
            self.y_max - self.y_min,
            self.z_max - self.z_min
        )


@dataclass
class CubeSelectionResult:
    """Result of cube selection operation."""
    selected_points: np.ndarray
    cube_bounds: CubeBounds
    total_points: int
    selected_count: int
    selection_ratio: float


class CubeSelector:
    """Pure business logic for cube-based point selection."""

    def __init__(self, points: np.ndarray):
        """
        Initialize cube selector with point cloud data.

        Args:
            points: Point cloud array of shape (N, 3)
        """
        if points.shape[1] != 3:
            raise ValueError("Points must be Nx3 array with x, y, z coordinates")

        self.points = points
        self.spatial_ops = None

        # Initialize optimized operations for large point clouds
        try:
            if len(points) > 50000:
                print(f"Initializing spatial optimization for {len(points):,} points...")
                self.spatial_ops = OptimizedPointCloudOps(points)
        except Exception as e:
            print(f"Warning: Could not initialize spatial optimization: {e}")

    def calculate_default_cube_bounds(self) -> CubeBounds:
        """Calculate default cube bounds centered on point cloud."""
        center = np.mean(self.points, axis=0)

        return CubeBounds(
            x_min=center[0] - InteractionConfig.DEFAULT_CUBE_X_SIZE/2,
            x_max=center[0] + InteractionConfig.DEFAULT_CUBE_X_SIZE/2,
            y_min=center[1] - InteractionConfig.DEFAULT_CUBE_Y_SIZE/2,
            y_max=center[1] + InteractionConfig.DEFAULT_CUBE_Y_SIZE/2,
            z_min=center[2] - InteractionConfig.DEFAULT_CUBE_Z_SIZE/2,
            z_max=center[2] + InteractionConfig.DEFAULT_CUBE_Z_SIZE/2
        )

    def filter_points_in_cube(self, cube_bounds: CubeBounds) -> np.ndarray:
        """
        Filter points that fall within cube bounds.

        Args:
            cube_bounds: Cube bounds for filtering

        Returns:
            Boolean array indicating points inside cube
        """
        try:
            bounds_list = cube_bounds.to_list()

            if self.spatial_ops:
                # Use spatial indexing for large point clouds
                return self.spatial_ops.filter_box_optimized(bounds_list)
            else:
                # Use vectorized operations for smaller point clouds
                return optimize_point_cloud_filtering(self.points, bounds_list, False)

        except Exception as e:
            print(f"Warning: Spatial optimization failed, using basic filtering: {e}")
            # Fallback to basic numpy operations
            return (
                (self.points[:, 0] >= cube_bounds.x_min) &
                (self.points[:, 0] <= cube_bounds.x_max) &
                (self.points[:, 1] >= cube_bounds.y_min) &
                (self.points[:, 1] <= cube_bounds.y_max) &
                (self.points[:, 2] >= cube_bounds.z_min) &
                (self.points[:, 2] <= cube_bounds.z_max)
            )

    def count_points_in_cube(self, cube_bounds: CubeBounds) -> int:
        """
        Count points inside cube bounds.

        Args:
            cube_bounds: Cube bounds for counting

        Returns:
            Number of points inside cube
        """
        try:
            inside_cube = self.filter_points_in_cube(cube_bounds)
            return np.sum(inside_cube)
        except Exception as e:
            print(f"Warning: Error counting points in cube: {e}")
            return 0

    def extract_points_in_cube(self, cube_bounds: CubeBounds) -> CubeSelectionResult:
        """
        Extract points that fall within cube bounds.

        Args:
            cube_bounds: Cube bounds for extraction

        Returns:
            CubeSelectionResult with selected points and metadata
        """
        inside_cube = self.filter_points_in_cube(cube_bounds)
        selected_points = self.points[inside_cube]
        selected_count = len(selected_points)
        total_points = len(self.points)

        return CubeSelectionResult(
            selected_points=selected_points,
            cube_bounds=cube_bounds,
            total_points=total_points,
            selected_count=selected_count,
            selection_ratio=selected_count / total_points if total_points > 0 else 0.0
        )

    def validate_cube_bounds(self, bounds: List[float]) -> Optional[CubeBounds]:
        """
        Validate and convert bounds list to CubeBounds object.

        Args:
            bounds: List of 6 floats [x_min, x_max, y_min, y_max, z_min, z_max]

        Returns:
            CubeBounds object if valid, None if invalid
        """
        try:
            if len(bounds) != 6:
                return None

            if any(np.isnan(bounds)):
                return None

            return CubeBounds.from_list(bounds)

        except (ValueError, TypeError):
            return None


class CubeVisualizationHelper:
    """Helper for generating visualization data from cube operations."""

    @staticmethod
    def generate_point_colors(points: np.ndarray, inside_cube: np.ndarray) -> np.ndarray:
        """
        Generate color array for point visualization.

        Args:
            points: Point cloud array
            inside_cube: Boolean array indicating points inside cube

        Returns:
            RGB color array (N, 3) with uint8 values
        """
        try:
            # Gray default, green for selected
            colors = np.full((len(points), 3), [128, 128, 128], dtype=np.uint8)
            colors[inside_cube] = [0, 255, 0]
            return colors

        except MemoryError as e:
            print(f"Warning: Memory allocation error for colors: {e}")
            # Return minimal color array
            return np.array([[128, 128, 128]], dtype=np.uint8)

    @staticmethod
    def should_update_visualization(current_count: int, last_count: int, total_points: int) -> bool:
        """
        Determine if visualization should be updated based on point count change.

        Args:
            current_count: Current number of selected points
            last_count: Previous number of selected points
            total_points: Total number of points

        Returns:
            True if visualization should be updated
        """
        if total_points == 0:
            return False

        # Update if count changed significantly (> 0.1% of total)
        change_threshold = total_points * 0.001
        return abs(current_count - last_count) > change_threshold