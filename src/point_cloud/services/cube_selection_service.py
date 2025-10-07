#!/usr/bin/env python3
"""
Cube Selection Service

Pure business logic for cube-based point cloud selection without GUI dependencies.
This service handles all cube selection algorithms, point filtering, and
geometric calculations.

Author: Dimitrije Stojanovic
Date: December 2025
Extracted: From interactive_cube_selector.py (Phase 3 Refactoring)
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from pathlib import Path
import json

from ..spatial_utils import OptimizedPointCloudOps, optimize_point_cloud_filtering
from ..config import GUIConfig
from ..error_handling import with_error_handling, validate_point_cloud_data
from .base_service import BaseService, ServiceResult


class CubeSelectionResult:
    """Data class for cube selection results."""

    def __init__(self, selected_points: Optional[np.ndarray] = None,
                 cube_bounds: Optional[Tuple[float, float, float, float, float, float]] = None,
                 selection_info: Optional[Dict[str, Any]] = None):
        self.selected_points = selected_points
        self.cube_bounds = cube_bounds
        self.selection_info = selection_info or {}

    @property
    def has_selection(self) -> bool:
        """Check if there's a valid selection."""
        return self.selected_points is not None and len(self.selected_points) > 0

    @property
    def point_count(self) -> int:
        """Get number of selected points."""
        return len(self.selected_points) if self.selected_points is not None else 0


class CubeSelectionService(BaseService):
    """
    Service for cube-based point cloud selection operations.

    This service handles all business logic for cube selection without any
    GUI dependencies, making it fully testable and reusable.
    """

    def __init__(self, points: Optional[np.ndarray] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the cube selection service.

        Args:
            points: numpy array of shape (N, 3) with x, y, z coordinates
            config: Optional configuration dictionary
        """
        super().__init__(config)

        if points is not None:
            validate_point_cloud_data(points)  # This validates but doesn't return
            self.points = points  # Set the validated points
        else:
            self.points = np.array([]).reshape(0, 3)  # Empty point cloud

        self.current_selection = CubeSelectionResult()

    def calculate_default_bounds(self) -> Tuple[float, float, float, float, float, float]:
        """
        Calculate default cube bounds based on point cloud statistics.

        Returns:
            Tuple of (xmin, xmax, ymin, ymax, zmin, zmax)
        """
        if self.points is None or len(self.points) == 0:
            return (0, 1, 0, 1, 0, 1)

        # Calculate bounds with some padding
        bounds = [
            np.min(self.points[:, 0]), np.max(self.points[:, 0]),  # x bounds
            np.min(self.points[:, 1]), np.max(self.points[:, 1]),  # y bounds
            np.min(self.points[:, 2]), np.max(self.points[:, 2])   # z bounds
        ]

        # Add 10% padding on all sides
        ranges = [bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]]
        padding = [r * 0.1 for r in ranges]

        default_bounds = (
            bounds[0] + padding[0], bounds[1] - padding[0],  # x
            bounds[2] + padding[1], bounds[3] - padding[1],  # y
            bounds[4] + padding[2], bounds[5] - padding[2]   # z
        )

        return default_bounds

    @with_error_handling("select_points_in_cube")
    def select_points_in_cube(self, cube_bounds: Tuple[float, float, float, float, float, float]) -> CubeSelectionResult:
        """
        Select points within the specified cube bounds.

        Args:
            cube_bounds: Tuple of (xmin, xmax, ymin, ymax, zmin, zmax)

        Returns:
            CubeSelectionResult with selected points and metadata
        """
        xmin, xmax, ymin, ymax, zmin, zmax = cube_bounds

        # Validate bounds
        if xmin >= xmax or ymin >= ymax or zmin >= zmax:
            return CubeSelectionResult(selection_info={'error': 'Invalid cube bounds'})

        # Use optimized point cloud operations if available
        try:
            mask = optimize_point_cloud_filtering(
                self.points,
                bounds=cube_bounds,
                operation='cube_selection'
            )
        except:
            # Fallback to basic numpy operations
            mask = (
                (self.points[:, 0] >= xmin) & (self.points[:, 0] <= xmax) &
                (self.points[:, 1] >= ymin) & (self.points[:, 1] <= ymax) &
                (self.points[:, 2] >= zmin) & (self.points[:, 2] <= zmax)
            )

        selected_points = self.points[mask]

        # Calculate selection statistics
        selection_info = {
            'total_points': len(self.points),
            'selected_points': len(selected_points),
            'selection_percentage': (len(selected_points) / len(self.points)) * 100,
            'cube_bounds': cube_bounds,
            'cube_volume': (xmax - xmin) * (ymax - ymin) * (zmax - zmin),
        }

        result = CubeSelectionResult(
            selected_points=selected_points,
            cube_bounds=cube_bounds,
            selection_info=selection_info
        )

        self.current_selection = result
        return result

    def count_points_in_cube(self, cube_bounds: Tuple[float, float, float, float, float, float]) -> int:
        """
        Count points within the cube bounds without creating a selection.

        Args:
            cube_bounds: Tuple of (xmin, xmax, ymin, ymax, zmin, zmax)

        Returns:
            Number of points within the cube
        """
        xmin, xmax, ymin, ymax, zmin, zmax = cube_bounds

        mask = (
            (self.points[:, 0] >= xmin) & (self.points[:, 0] <= xmax) &
            (self.points[:, 1] >= ymin) & (self.points[:, 1] <= ymax) &
            (self.points[:, 2] >= zmin) & (self.points[:, 2] <= zmax)
        )

        return np.sum(mask)

    def get_point_colors_for_selection(self, cube_bounds: Tuple[float, float, float, float, float, float]) -> np.ndarray:
        """
        Generate colors for points based on cube selection (green=inside, gray=outside).

        Args:
            cube_bounds: Tuple of (xmin, xmax, ymin, ymax, zmin, zmax)

        Returns:
            Array of RGB colors for each point
        """
        xmin, xmax, ymin, ymax, zmin, zmax = cube_bounds

        mask = (
            (self.points[:, 0] >= xmin) & (self.points[:, 0] <= xmax) &
            (self.points[:, 1] >= ymin) & (self.points[:, 1] <= ymax) &
            (self.points[:, 2] >= zmin) & (self.points[:, 2] <= zmax)
        )

        # Create color array (RGB values 0-255)
        colors = np.full((len(self.points), 3), [128, 128, 128], dtype=np.uint8)  # Gray default
        colors[mask] = [0, 255, 0]  # Green for selected points

        return colors

    def validate_cube_bounds(self, cube_bounds: Tuple[float, float, float, float, float, float]) -> Dict[str, Any]:
        """
        Validate cube bounds and return validation result.

        Args:
            cube_bounds: Tuple of (xmin, xmax, ymin, ymax, zmin, zmax)

        Returns:
            Dictionary with validation results
        """
        xmin, xmax, ymin, ymax, zmin, zmax = cube_bounds

        validation = {
            'valid': True,
            'errors': [],
            'warnings': []
        }

        # Check for invalid bounds
        if xmin >= xmax:
            validation['valid'] = False
            validation['errors'].append('X minimum must be less than X maximum')

        if ymin >= ymax:
            validation['valid'] = False
            validation['errors'].append('Y minimum must be less than Y maximum')

        if zmin >= zmax:
            validation['valid'] = False
            validation['errors'].append('Z minimum must be less than Z maximum')

        # Check if bounds are reasonable
        point_bounds = self.get_point_cloud_bounds()

        # Check if cube is outside point cloud bounds
        if xmax < point_bounds[0] or xmin > point_bounds[1]:
            validation['warnings'].append('Cube X range is outside point cloud bounds')

        if ymax < point_bounds[2] or ymin > point_bounds[3]:
            validation['warnings'].append('Cube Y range is outside point cloud bounds')

        if zmax < point_bounds[4] or zmin > point_bounds[5]:
            validation['warnings'].append('Cube Z range is outside point cloud bounds')

        return validation

    def get_point_cloud_bounds(self) -> Tuple[float, float, float, float, float, float]:
        """
        Get the bounds of the entire point cloud.

        Returns:
            Tuple of (xmin, xmax, ymin, ymax, zmin, zmax)
        """
        if self.points is None or len(self.points) == 0:
            return (0, 0, 0, 0, 0, 0)

        return (
            float(np.min(self.points[:, 0])), float(np.max(self.points[:, 0])),
            float(np.min(self.points[:, 1])), float(np.max(self.points[:, 1])),
            float(np.min(self.points[:, 2])), float(np.max(self.points[:, 2]))
        )

    def get_selection_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the current selection.

        Returns:
            Dictionary with selection statistics
        """
        if not self.current_selection.has_selection:
            return {'error': 'No current selection'}

        points = self.current_selection.selected_points

        stats = {
            'total_points': len(self.points),
            'selected_points': len(points),
            'selection_percentage': (len(points) / len(self.points)) * 100,
            'cube_bounds': self.current_selection.cube_bounds,
            'point_density': len(points) / self.current_selection.selection_info.get('cube_volume', 1),
        }

        if len(points) > 0:
            stats.update({
                'centroid': [float(np.mean(points[:, i])) for i in range(3)],
                'std_deviation': [float(np.std(points[:, i])) for i in range(3)],
                'bounds': {
                    'x': [float(np.min(points[:, 0])), float(np.max(points[:, 0]))],
                    'y': [float(np.min(points[:, 1])), float(np.max(points[:, 1]))],
                    'z': [float(np.min(points[:, 2])), float(np.max(points[:, 2]))],
                }
            })

        return stats

    def save_selection(self, output_path: Path, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Save the current selection to file.

        Args:
            output_path: Path to save the selection
            metadata: Optional metadata to include

        Returns:
            True if saved successfully, False otherwise
        """
        if not self.current_selection.has_selection:
            return False

        try:
            # Save points as numpy array
            np.save(output_path, self.current_selection.selected_points)

            # Save metadata as JSON if provided
            if metadata:
                metadata_path = output_path.with_suffix('.json')
                combined_metadata = {
                    **self.get_selection_statistics(),
                    **metadata
                }

                with open(metadata_path, 'w') as f:
                    json.dump(combined_metadata, f, indent=2)

            return True

        except Exception as e:
            print(f"Error saving selection: {e}")
            return False

    def load_selection_from_file(self, file_path: Path) -> Optional[CubeSelectionResult]:
        """
        Load a previously saved selection.

        Args:
            file_path: Path to the saved selection

        Returns:
            CubeSelectionResult if loaded successfully, None otherwise
        """
        try:
            points = np.load(file_path)

            # Try to load metadata
            metadata_path = file_path.with_suffix('.json')
            metadata = {}

            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

            result = CubeSelectionResult(
                selected_points=points,
                cube_bounds=metadata.get('cube_bounds'),
                selection_info=metadata
            )

            return result

        except Exception as e:
            print(f"Error loading selection: {e}")
            return None

    def reset_selection(self) -> None:
        """Reset the current selection."""
        self.current_selection = CubeSelectionResult()

    def has_current_selection(self) -> bool:
        """Check if there's a current selection."""
        return self.current_selection.has_selection

    def set_points(self, points: np.ndarray) -> ServiceResult:
        """
        Set new point cloud data for the service.

        Args:
            points: numpy array of shape (N, 3) with x, y, z coordinates

        Returns:
            ServiceResult indicating success/failure
        """
        try:
            validate_point_cloud_data(points)  # This validates but doesn't return
            self.points = points  # Set the validated points
            self.reset_selection()  # Clear any existing selection
            return self.create_success_result(
                data=f"Loaded {len(self.points)} points",
                point_count=len(self.points)
            )
        except Exception as e:
            return self.create_error_result(f"Error setting points: {str(e)}")

    def get_point_count(self) -> int:
        """Get the number of points in the current point cloud."""
        return len(self.points)

    def get_service_info(self) -> Dict[str, Any]:
        """
        Get service information and capabilities.

        Returns:
            Dictionary with service information
        """
        return {
            'service_name': 'CubeSelectionService',
            'description': 'Point cloud cube selection and filtering operations',
            'capabilities': [
                'cube_selection',
                'point_filtering',
                'bounds_calculation',
                'color_generation',
                'selection_statistics',
                'file_operations'
            ],
            'current_state': {
                'point_count': len(self.points),
                'has_selection': self.has_current_selection(),
                'selection_point_count': self.current_selection.point_count if self.has_current_selection() else 0
            },
            'supported_operations': {
                'select_points_in_cube': 'Select points within cube bounds',
                'count_points_in_cube': 'Count points within cube bounds',
                'get_point_colors_for_selection': 'Generate colors for selection visualization',
                'calculate_default_bounds': 'Calculate default cube bounds',
                'validate_cube_bounds': 'Validate cube bounds',
                'get_selection_statistics': 'Get comprehensive selection statistics',
                'save_selection': 'Save current selection to file',
                'load_selection_from_file': 'Load selection from file'
            },
            'configuration': {
                'padding_percentage': self.get_config_value('padding_percentage', 0.1),
                'default_colors': {
                    'selected': [0, 255, 0],  # Green
                    'unselected': [128, 128, 128]  # Gray
                }
            }
        }