#!/usr/bin/env python3
"""
Data Models for Point Cloud Analysis

Defines the core data structures used throughout the analysis pipeline.
These models provide type safety and clear contracts between components.

Author: Dimitrije Stojanovic
Date: September 2025
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple
import numpy as np


@dataclass
class PointCloudData:
    """Container for raw point cloud data with optional attributes."""

    points: np.ndarray  # Nx3 array of XYZ coordinates
    colors: Optional[np.ndarray] = None  # Nx3 array of RGB values
    intensity: Optional[np.ndarray] = None  # Nx1 array of intensity values
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate data consistency."""
        if self.points is None or len(self.points) == 0:
            raise ValueError("Points array cannot be empty")

        if self.points.shape[1] != 3:
            raise ValueError("Points must have 3 coordinates (X, Y, Z)")

        n_points = len(self.points)

        if self.colors is not None:
            if len(self.colors) != n_points:
                raise ValueError("Colors array must match points array length")
            if self.colors.shape[1] != 3:
                raise ValueError("Colors must have 3 values (R, G, B)")

        if self.intensity is not None:
            if len(self.intensity) != n_points:
                raise ValueError("Intensity array must match points array length")

    @property
    def size(self) -> int:
        """Number of points in the cloud."""
        return len(self.points)

    @property
    def bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        """Get min/max bounds for each axis."""
        return (
            (self.points[:, 0].min(), self.points[:, 0].max()),  # X bounds
            (self.points[:, 1].min(), self.points[:, 1].max()),  # Y bounds
            (self.points[:, 2].min(), self.points[:, 2].max())   # Z bounds
        )


@dataclass
class GridAnalysisData:
    """Container for 3D grid analysis results."""

    occupancy_grid_3d: np.ndarray  # 3D boolean array of occupied voxels
    interior_grid_3d: np.ndarray   # 3D boolean array of interior voxels
    distance_map_3d: np.ndarray    # 3D float array of distance values
    bounds_3d: Tuple[float, float, float, float, float, float, float]  # (x_min, x_max, y_min, y_max, z_min, z_max, resolution)

    def __post_init__(self):
        """Validate grid data consistency."""
        if self.occupancy_grid_3d.shape != self.interior_grid_3d.shape:
            raise ValueError("Occupancy and interior grids must have same shape")

        if self.occupancy_grid_3d.shape != self.distance_map_3d.shape:
            raise ValueError("Occupancy and distance map grids must have same shape")

        if len(self.bounds_3d) != 7:
            raise ValueError("Bounds must contain (x_min, x_max, y_min, y_max, z_min, z_max, resolution)")

    @property
    def grid_shape(self) -> Tuple[int, int, int]:
        """Shape of the 3D grids (depth, height, width)."""
        return self.occupancy_grid_3d.shape

    @property
    def resolution(self) -> float:
        """Grid resolution in meters."""
        return self.bounds_3d[6]


@dataclass
class AnalysisResults:
    """Container for all analysis results from the processing pipeline."""

    raw_data: PointCloudData
    ground_points: Optional[np.ndarray] = None
    non_ground_points: Optional[np.ndarray] = None
    vehicle_points: Optional[np.ndarray] = None
    interior_points: Optional[np.ndarray] = None
    human_points: Optional[np.ndarray] = None
    seat_points: Optional[np.ndarray] = None
    grid_data: Optional[GridAnalysisData] = None

    # AI analysis results
    ptv3_results: Optional[Dict[str, Any]] = None
    human_detection_results: Optional[Dict[str, Any]] = None
    human_pose_analysis: Optional[list] = None

    # Enhanced vehicle points from AI
    vehicle_points_ptv3: Optional[np.ndarray] = None
    interior_points_ptv3: Optional[np.ndarray] = None
    ground_points_ptv3: Optional[np.ndarray] = None

    def __post_init__(self):
        """Validate analysis results."""
        if self.raw_data is None:
            raise ValueError("Raw data is required")

    @property
    def has_ground_separation(self) -> bool:
        """Check if ground separation was performed."""
        return self.ground_points is not None and self.non_ground_points is not None

    @property
    def has_vehicle_detection(self) -> bool:
        """Check if vehicle detection was performed."""
        return self.vehicle_points is not None

    @property
    def has_interior_detection(self) -> bool:
        """Check if interior detection was performed."""
        return self.interior_points is not None

    @property
    def has_human_detection(self) -> bool:
        """Check if human detection was performed."""
        return self.human_detection_results is not None

    @property
    def has_ai_analysis(self) -> bool:
        """Check if AI analysis was performed."""
        return self.ptv3_results is not None

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of analysis results."""
        summary = {
            'total_points': self.raw_data.size,
            'ground_points': len(self.ground_points) if self.ground_points is not None else 0,
            'non_ground_points': len(self.non_ground_points) if self.non_ground_points is not None else 0,
            'vehicle_points': len(self.vehicle_points) if self.vehicle_points is not None else 0,
            'interior_points': len(self.interior_points) if self.interior_points is not None else 0,
            'human_points': len(self.human_points) if self.human_points is not None else 0,
            'has_ai_analysis': self.has_ai_analysis,
            'has_human_detection': self.has_human_detection,
        }

        if self.has_human_detection and self.human_detection_results:
            summary.update({
                'humans_detected': self.human_detection_results.get('human_detected', False),
                'estimated_human_count': self.human_detection_results.get('estimated_human_count', 0),
                'human_confidence': self.human_detection_results.get('average_confidence', 0.0)
            })

        return summary