#!/usr/bin/env python3
"""
Cube Selection Controller

Controller for coordinating cube selection operations between GUI widgets
and the CubeSelectionService. Handles user interactions and manages state
without containing business logic.

Author: Dimitrije Stojanovic
Date: December 2025
Created: Phase 3 Refactoring - GUI/Business Logic Separation
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from pathlib import Path

from .base_controller import BaseController
from ..services import ServiceResult


class CubeSelectionController(BaseController):
    """
    Controller for cube-based point cloud selection operations.

    Mediates between CubeSelectionWidget and CubeSelectionService:
    - Manages point cloud data loading
    - Coordinates cube selection operations
    - Handles validation and error reporting
    - Manages selection state and statistics
    """

    def __init__(self, point_cloud_data: Optional[np.ndarray] = None,
                 controller_config: Optional[Dict[str, Any]] = None):
        """
        Initialize cube selection controller.

        Args:
            point_cloud_data: Optional initial point cloud data
            controller_config: Controller configuration options
        """
        # Initialize with required services
        super().__init__(
            service_names=self.get_required_services(),
            controller_config=controller_config
        )

        # Controller state
        self._point_cloud_data: Optional[np.ndarray] = None
        self._current_cube_bounds: Optional[Tuple[float, float, float, float, float, float]] = None

        # Initialize with point cloud data if provided
        if point_cloud_data is not None:
            self.set_point_cloud_data(point_cloud_data)

    def get_required_services(self) -> List[str]:
        """Get list of required service names."""
        return ['CubeSelectionService']

    def set_point_cloud_data(self, point_cloud_data: np.ndarray) -> ServiceResult:
        """
        Set point cloud data for cube selection.

        Args:
            point_cloud_data: NumPy array of shape (N, 3) with x, y, z coordinates

        Returns:
            ServiceResult indicating success/failure
        """
        try:
            service = self.get_service('CubeSelectionService')
            if not service:
                return self.create_service_result(
                    success=False,
                    error="CubeSelectionService not available"
                )

            # Set data in service
            result = service.set_points(point_cloud_data)
            if result.success:
                self._point_cloud_data = point_cloud_data
                self.logger.info(f"Point cloud data set: {len(point_cloud_data)} points")

            return result

        except Exception as e:
            return self.handle_service_error(e, "set_point_cloud_data")

    def get_point_cloud_bounds(self) -> Optional[Tuple[float, float, float, float, float, float]]:
        """
        Get bounds of the current point cloud.

        Returns:
            Tuple of (xmin, xmax, ymin, ymax, zmin, zmax) or None if no data
        """
        try:
            service = self.get_service('CubeSelectionService')
            if not service:
                return None

            return service.get_point_cloud_bounds()

        except Exception as e:
            self.logger.error(f"Error getting point cloud bounds: {str(e)}")
            return None

    def calculate_default_cube_bounds(self) -> Optional[Tuple[float, float, float, float, float, float]]:
        """
        Calculate default cube bounds based on point cloud statistics.

        Returns:
            Tuple of (xmin, xmax, ymin, ymax, zmin, zmax) or None if failed
        """
        try:
            service = self.get_service('CubeSelectionService')
            if not service:
                return None

            bounds = service.calculate_default_bounds()
            self._current_cube_bounds = bounds
            return bounds

        except Exception as e:
            self.logger.error(f"Error calculating default bounds: {str(e)}")
            return None

    def validate_cube_bounds(self, cube_bounds: Tuple[float, float, float, float, float, float]) -> Dict[str, Any]:
        """
        Validate cube bounds and return validation result.

        Args:
            cube_bounds: Tuple of (xmin, xmax, ymin, ymax, zmin, zmax)

        Returns:
            Dictionary with validation results
        """
        try:
            service = self.get_service('CubeSelectionService')
            if not service:
                return {'valid': False, 'errors': ['CubeSelectionService not available']}

            return service.validate_cube_bounds(cube_bounds)

        except Exception as e:
            self.logger.error(f"Error validating cube bounds: {str(e)}")
            return {'valid': False, 'errors': [f'Validation error: {str(e)}']}

    def count_points_in_cube(self, cube_bounds: Tuple[float, float, float, float, float, float]) -> Optional[int]:
        """
        Count points within the cube bounds without creating a selection.

        Args:
            cube_bounds: Tuple of (xmin, xmax, ymin, ymax, zmin, zmax)

        Returns:
            Number of points within the cube or None if failed
        """
        try:
            service = self.get_service('CubeSelectionService')
            if not service:
                return None

            return service.count_points_in_cube(cube_bounds)

        except Exception as e:
            self.logger.error(f"Error counting points in cube: {str(e)}")
            return None

    def select_points_in_cube(self, cube_bounds: Tuple[float, float, float, float, float, float]) -> Optional[Any]:
        """
        Select points within the specified cube bounds.

        Args:
            cube_bounds: Tuple of (xmin, xmax, ymin, ymax, zmin, zmax)

        Returns:
            CubeSelectionResult or None if failed
        """
        try:
            service = self.get_service('CubeSelectionService')
            if not service:
                return None

            result = service.select_points_in_cube(cube_bounds)
            if result.has_selection:
                self._current_cube_bounds = cube_bounds
                self.logger.info(f"Selected {result.point_count} points in cube")

            return result

        except Exception as e:
            self.logger.error(f"Error selecting points in cube: {str(e)}")
            return None

    def get_selection_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the current selection.

        Returns:
            Dictionary with selection statistics
        """
        try:
            service = self.get_service('CubeSelectionService')
            if not service:
                return {'error': 'CubeSelectionService not available'}

            return service.get_selection_statistics()

        except Exception as e:
            self.logger.error(f"Error getting selection statistics: {str(e)}")
            return {'error': f'Statistics error: {str(e)}'}

    def save_selection(self, output_path: Path, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Save the current selection to file.

        Args:
            output_path: Path to save the selection
            metadata: Optional metadata to include

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            service = self.get_service('CubeSelectionService')
            if not service:
                return False

            # Add controller metadata
            full_metadata = metadata or {}
            full_metadata.update({
                'controller': self.__class__.__name__,
                'cube_bounds': self._current_cube_bounds,
                'point_cloud_size': len(self._point_cloud_data) if self._point_cloud_data is not None else 0
            })

            success = service.save_selection(output_path, full_metadata)
            if success:
                self.logger.info(f"Selection saved to: {output_path}")

            return success

        except Exception as e:
            self.logger.error(f"Error saving selection: {str(e)}")
            return False

    def get_point_colors_for_selection(self, cube_bounds: Tuple[float, float, float, float, float, float]) -> Optional[np.ndarray]:
        """
        Generate colors for points based on cube selection.

        Args:
            cube_bounds: Tuple of (xmin, xmax, ymin, ymax, zmin, zmax)

        Returns:
            Array of RGB colors for each point or None if failed
        """
        try:
            service = self.get_service('CubeSelectionService')
            if not service:
                return None

            return service.get_point_colors_for_selection(cube_bounds)

        except Exception as e:
            self.logger.error(f"Error getting point colors: {str(e)}")
            return None

    def reset_selection(self) -> None:
        """Reset the current selection."""
        try:
            service = self.get_service('CubeSelectionService')
            if service:
                service.reset_selection()

            self._current_cube_bounds = None
            self.logger.info("Selection reset")

        except Exception as e:
            self.logger.error(f"Error resetting selection: {str(e)}")

    def has_point_cloud_data(self) -> bool:
        """Check if point cloud data is loaded."""
        return self._point_cloud_data is not None

    def has_current_selection(self) -> bool:
        """Check if there's a current selection."""
        try:
            service = self.get_service('CubeSelectionService')
            if not service:
                return False

            return service.has_current_selection()

        except Exception as e:
            self.logger.error(f"Error checking current selection: {str(e)}")
            return False

    def get_point_count(self) -> int:
        """Get the number of points in the current point cloud."""
        try:
            service = self.get_service('CubeSelectionService')
            if not service:
                return 0

            return service.get_point_count()

        except Exception as e:
            self.logger.error(f"Error getting point count: {str(e)}")
            return 0

    def get_current_cube_bounds(self) -> Optional[Tuple[float, float, float, float, float, float]]:
        """Get the current cube bounds."""
        return self._current_cube_bounds

    def get_controller_status(self) -> Dict[str, Any]:
        """
        Get detailed controller status information.

        Returns:
            Dictionary with status information
        """
        base_info = self.get_controller_info()

        status_info = {
            'has_point_cloud_data': self.has_point_cloud_data(),
            'point_count': self.get_point_count(),
            'has_current_selection': self.has_current_selection(),
            'current_cube_bounds': self._current_cube_bounds,
            'point_cloud_bounds': self.get_point_cloud_bounds()
        }

        base_info.update(status_info)
        return base_info

    def load_point_cloud_from_service(self, service_name: str, method_name: str, *args, **kwargs) -> ServiceResult:
        """
        Load point cloud data from another service.

        Args:
            service_name: Name of the service to load from
            method_name: Method name to call on the service
            *args: Arguments for the method
            **kwargs: Keyword arguments for the method

        Returns:
            ServiceResult indicating success/failure
        """
        try:
            # This would typically be used with a file loading service
            # or visualization service that has point cloud data
            result = self.safe_call_service(service_name, method_name, *args, **kwargs)

            if result is not None and hasattr(result, 'shape') and len(result.shape) == 2 and result.shape[1] >= 3:
                # Looks like valid point cloud data
                return self.set_point_cloud_data(result)
            else:
                return self.create_service_result(
                    success=False,
                    error=f"Invalid point cloud data from {service_name}.{method_name}"
                )

        except Exception as e:
            return self.handle_service_error(e, "load_point_cloud_from_service")