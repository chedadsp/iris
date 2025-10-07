#!/usr/bin/env python3
"""
Geometric Transformations - Pure spatial algorithms for coordinate transformations.

Contains business logic for geometric operations including rotation, scaling, and positioning.
Extracted from interactive_human_positioner.py as part of GUI/business logic separation.

Author: Dimitrije Stojanovic
Date: September 2025
Refactored: November 2025 (GUI/Business Logic Separation)
"""

import numpy as np
from typing import Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class Transform3D:
    """3D transformation parameters."""
    center: Tuple[float, float, float]
    scale: float = 1.0
    rotation_deg: float = 0.0

    def __post_init__(self):
        """Validate transformation parameters."""
        if len(self.center) != 3:
            raise ValueError("Center must be a 3-element tuple (x, y, z)")

        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("Scale must be a positive finite number")

        if not np.isfinite(self.rotation_deg):
            raise ValueError("Rotation must be a finite number")

        # Normalize rotation to [0, 360)
        self.rotation_deg = self.rotation_deg % 360

    @classmethod
    def from_cube_bounds(cls, cube_bounds: list) -> 'Transform3D':
        """
        Create transform from cube widget bounds.

        Args:
            cube_bounds: List of [x_min, x_max, y_min, y_max, z_min, z_max]

        Returns:
            Transform3D object
        """
        if len(cube_bounds) != 6:
            raise ValueError("Cube bounds must have 6 elements")

        x_min, x_max, y_min, y_max, z_min, z_max = cube_bounds

        # Calculate center
        center = (
            (x_min + x_max) / 2,
            (y_min + y_max) / 2,
            (z_min + z_max) / 2
        )

        # Calculate scale based on cube size
        cube_width = x_max - x_min
        cube_height = z_max - z_min
        scale = min(cube_width, cube_height) / 1.0  # Normalize to default size

        # Clamp scale to reasonable range
        scale = max(0.1, min(3.0, scale))

        return cls(center=center, scale=scale)


class GeometricTransformer:
    """Pure business logic for geometric transformations."""

    @staticmethod
    def rotate_points_z_axis(points: np.ndarray, center: Tuple[float, float, float],
                           rotation_deg: float) -> np.ndarray:
        """
        Rotate points around Z-axis.

        Args:
            points: Point array of shape (N, 3)
            center: Center of rotation (x, y, z)
            rotation_deg: Rotation angle in degrees

        Returns:
            Rotated points array
        """
        if rotation_deg == 0.0:
            return points.copy()

        rotation_rad = np.radians(rotation_deg)
        cos_rot = np.cos(rotation_rad)
        sin_rot = np.sin(rotation_rad)

        # Work with a copy
        rotated_points = points.copy()

        # Translate to origin
        rotated_points[:, 0] -= center[0]
        rotated_points[:, 1] -= center[1]

        # Apply rotation matrix for Z-axis rotation
        x_rot = rotated_points[:, 0] * cos_rot - rotated_points[:, 1] * sin_rot
        y_rot = rotated_points[:, 0] * sin_rot + rotated_points[:, 1] * cos_rot

        # Update coordinates and translate back
        rotated_points[:, 0] = x_rot + center[0]
        rotated_points[:, 1] = y_rot + center[1]

        return rotated_points

    @staticmethod
    def scale_points(points: np.ndarray, center: Tuple[float, float, float],
                    scale: float) -> np.ndarray:
        """
        Scale points around a center point.

        Args:
            points: Point array of shape (N, 3)
            center: Center of scaling (x, y, z)
            scale: Scale factor

        Returns:
            Scaled points array
        """
        if scale == 1.0:
            return points.copy()

        scaled_points = points.copy()

        # Translate to origin, scale, then translate back
        scaled_points[:, 0] = (scaled_points[:, 0] - center[0]) * scale + center[0]
        scaled_points[:, 1] = (scaled_points[:, 1] - center[1]) * scale + center[1]
        scaled_points[:, 2] = (scaled_points[:, 2] - center[2]) * scale + center[2]

        return scaled_points

    @staticmethod
    def apply_full_transform(points: np.ndarray, transform: Transform3D) -> np.ndarray:
        """
        Apply complete 3D transformation (scale + rotate).

        Args:
            points: Point array of shape (N, 3)
            transform: Transform3D parameters

        Returns:
            Transformed points array
        """
        # Apply scaling first
        transformed = GeometricTransformer.scale_points(points, transform.center, transform.scale)

        # Then apply rotation
        transformed = GeometricTransformer.rotate_points_z_axis(
            transformed, transform.center, transform.rotation_deg
        )

        return transformed

    @staticmethod
    def add_noise_to_points(points: np.ndarray, noise_scale: float = 0.02) -> np.ndarray:
        """
        Add Gaussian noise to points for realism.

        Args:
            points: Point array of shape (N, 3)
            noise_scale: Scale of the noise relative to the model

        Returns:
            Points with added noise
        """
        noise = np.random.normal(0, noise_scale, points.shape)
        return points + noise


class PositionCalculator:
    """Helper for calculating positions within point cloud bounds."""

    @staticmethod
    def calculate_point_cloud_bounds(points: np.ndarray) -> Dict[str, float]:
        """
        Calculate bounding box of point cloud.

        Args:
            points: Point cloud array

        Returns:
            Dictionary with min/max bounds for each axis
        """
        return {
            'x_min': float(points[:, 0].min()),
            'x_max': float(points[:, 0].max()),
            'y_min': float(points[:, 1].min()),
            'y_max': float(points[:, 1].max()),
            'z_min': float(points[:, 2].min()),
            'z_max': float(points[:, 2].max())
        }

    @staticmethod
    def calculate_center_position(bounds: Dict[str, float]) -> Tuple[float, float, float]:
        """
        Calculate center position from bounds.

        Args:
            bounds: Bounds dictionary from calculate_point_cloud_bounds

        Returns:
            Center position (x, y, z)
        """
        x_center = (bounds['x_min'] + bounds['x_max']) / 2
        y_center = (bounds['y_min'] + bounds['y_max']) / 2
        z_center = (bounds['z_min'] + bounds['z_max']) / 2

        return (x_center, y_center, z_center)

    @staticmethod
    def calculate_optimal_human_position(bounds: Dict[str, float]) -> Tuple[float, float, float]:
        """
        Calculate optimal position for human model placement.

        Args:
            bounds: Point cloud bounds

        Returns:
            Optimal human position (x, y, z)
        """
        # Place human in center horizontally, slightly below top vertically
        x_center = (bounds['x_min'] + bounds['x_max']) / 2
        y_center = (bounds['y_min'] + bounds['y_max']) / 2
        z_center = bounds['z_max'] - 0.3  # Slightly below the top

        return (x_center, y_center, z_center)

    @staticmethod
    def combine_point_clouds(original_points: np.ndarray,
                           additional_points: np.ndarray) -> np.ndarray:
        """
        Combine two point clouds into one.

        Args:
            original_points: Original point cloud
            additional_points: Points to add

        Returns:
            Combined point cloud
        """
        return np.vstack([original_points, additional_points])


class TransformationValidator:
    """Validator for transformation parameters."""

    @staticmethod
    def validate_transform_parameters(center: Tuple[float, float, float],
                                    scale: float, rotation_deg: float) -> bool:
        """
        Validate transformation parameters.

        Args:
            center: Center position
            scale: Scale factor
            rotation_deg: Rotation angle

        Returns:
            True if all parameters are valid
        """
        # Validate center
        if len(center) != 3 or not all(np.isfinite(center)):
            return False

        # Validate scale
        if not np.isfinite(scale) or scale <= 0:
            return False

        # Validate rotation
        if not np.isfinite(rotation_deg):
            return False

        return True

    @staticmethod
    def clamp_scale(scale: float, min_scale: float = 0.1, max_scale: float = 3.0) -> float:
        """
        Clamp scale to reasonable bounds.

        Args:
            scale: Input scale
            min_scale: Minimum allowed scale
            max_scale: Maximum allowed scale

        Returns:
            Clamped scale value
        """
        return max(min_scale, min(max_scale, scale))

    @staticmethod
    def normalize_rotation(rotation_deg: float) -> float:
        """
        Normalize rotation angle to [0, 360) range.

        Args:
            rotation_deg: Input rotation angle

        Returns:
            Normalized rotation angle
        """
        return rotation_deg % 360