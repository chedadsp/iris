#!/usr/bin/env python3
"""
Human Positioning Controller

Controller for coordinating human model positioning operations between GUI widgets
and the HumanPositioningService. Handles user interactions and manages state
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


class HumanPositioningController(BaseController):
    """
    Controller for human model positioning and manipulation operations.

    Mediates between HumanPositioningWidget and HumanPositioningService:
    - Manages point cloud data for positioning context
    - Coordinates human model generation and updates
    - Handles model parameter changes
    - Manages combined model operations
    """

    def __init__(self, point_cloud_data: Optional[np.ndarray] = None,
                 controller_config: Optional[Dict[str, Any]] = None):
        """
        Initialize human positioning controller.

        Args:
            point_cloud_data: Optional initial point cloud data for positioning context
            controller_config: Controller configuration options
        """
        # Initialize with required services
        super().__init__(
            service_names=self.get_required_services(),
            controller_config=controller_config
        )

        # Controller state
        self._point_cloud_data: Optional[np.ndarray] = None
        self._current_model_parameters: Dict[str, Any] = {
            'position': (0.0, 0.0, 0.0),
            'scale': 1.0,
            'rotation': 0.0,
            'model_type': 'seated'
        }

        # Initialize with point cloud data if provided
        if point_cloud_data is not None:
            self.set_point_cloud_data(point_cloud_data)

    def get_required_services(self) -> List[str]:
        """Get list of required service names."""
        return ['HumanPositioningService']

    def set_point_cloud_data(self, point_cloud_data: np.ndarray) -> ServiceResult:
        """
        Set point cloud data for human positioning context.

        Args:
            point_cloud_data: NumPy array of shape (N, 3) with x, y, z coordinates

        Returns:
            ServiceResult indicating success/failure
        """
        try:
            service = self.get_service('HumanPositioningService')
            if not service:
                return self.create_service_result(
                    success=False,
                    error="HumanPositioningService not available"
                )

            # Set data in service
            result = service.set_point_cloud_data(point_cloud_data)
            if result.success:
                self._point_cloud_data = point_cloud_data
                self.logger.info(f"Point cloud data set for positioning: {len(point_cloud_data)} points")

            return result

        except Exception as e:
            return self.handle_service_error(e, "set_point_cloud_data")

    def calculate_optimal_position(self) -> Optional[Tuple[float, float, float]]:
        """
        Calculate optimal position for human model based on point cloud geometry.

        Returns:
            Optimal (x, y, z) position or None if failed
        """
        try:
            service = self.get_service('HumanPositioningService')
            if not service:
                return None

            position = service.calculate_optimal_position()
            if position:
                # Update current parameters
                self._current_model_parameters['position'] = position
                self.logger.info(f"Calculated optimal position: {position}")

            return position

        except Exception as e:
            self.logger.error(f"Error calculating optimal position: {str(e)}")
            return None

    def generate_human_model(self, position: Tuple[float, float, float],
                           scale: float = 1.0, rotation: float = 0.0,
                           model_type: str = 'seated') -> Optional[Any]:
        """
        Generate human model with specified parameters.

        Args:
            position: (x, y, z) position for the human
            scale: Scale factor for the human size
            rotation: Rotation angle in degrees around Z-axis
            model_type: Type of human model ('seated' or 'standing')

        Returns:
            HumanModelResult or None if failed
        """
        try:
            service = self.get_service('HumanPositioningService')
            if not service:
                return None

            result = service.generate_human_model(
                position=position,
                scale=scale,
                rotation=rotation,
                model_type=model_type
            )

            if result and result.has_model:
                # Update current parameters
                self._current_model_parameters.update({
                    'position': position,
                    'scale': scale,
                    'rotation': rotation,
                    'model_type': model_type
                })
                self.logger.info(f"Generated {model_type} human model with {result.point_count} points")

            return result

        except Exception as e:
            self.logger.error(f"Error generating human model: {str(e)}")
            return None

    def update_model_position(self, new_position: Tuple[float, float, float]) -> Optional[Any]:
        """
        Update the position of the current model.

        Args:
            new_position: New (x, y, z) position

        Returns:
            Updated HumanModelResult or None if failed
        """
        try:
            service = self.get_service('HumanPositioningService')
            if not service:
                return None

            result = service.update_model_position(new_position)
            if result and result.has_model:
                self._current_model_parameters['position'] = new_position
                self.logger.info(f"Updated model position to: {new_position}")

            return result

        except Exception as e:
            self.logger.error(f"Error updating model position: {str(e)}")
            return None

    def update_model_scale(self, new_scale: float) -> Optional[Any]:
        """
        Update the scale of the current model.

        Args:
            new_scale: New scale factor

        Returns:
            Updated HumanModelResult or None if failed
        """
        try:
            service = self.get_service('HumanPositioningService')
            if not service:
                return None

            result = service.update_model_scale(new_scale)
            if result and result.has_model:
                self._current_model_parameters['scale'] = new_scale
                self.logger.info(f"Updated model scale to: {new_scale}")

            return result

        except Exception as e:
            self.logger.error(f"Error updating model scale: {str(e)}")
            return None

    def update_model_rotation(self, new_rotation: float) -> Optional[Any]:
        """
        Update the rotation of the current model.

        Args:
            new_rotation: New rotation angle in degrees

        Returns:
            Updated HumanModelResult or None if failed
        """
        try:
            service = self.get_service('HumanPositioningService')
            if not service:
                return None

            result = service.update_model_rotation(new_rotation)
            if result and result.has_model:
                self._current_model_parameters['rotation'] = new_rotation
                self.logger.info(f"Updated model rotation to: {new_rotation}°")

            return result

        except Exception as e:
            self.logger.error(f"Error updating model rotation: {str(e)}")
            return None

    def get_model_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the current model.

        Returns:
            Dictionary with model statistics
        """
        try:
            service = self.get_service('HumanPositioningService')
            if not service:
                return {'error': 'HumanPositioningService not available'}

            stats = service.get_model_statistics()

            # Add controller-level information
            if isinstance(stats, dict) and 'error' not in stats:
                stats['controller_parameters'] = self._current_model_parameters.copy()

            return stats

        except Exception as e:
            self.logger.error(f"Error getting model statistics: {str(e)}")
            return {'error': f'Statistics error: {str(e)}'}

    def save_combined_model(self, output_path: Path, include_metadata: bool = True) -> bool:
        """
        Save the combined human model with point cloud.

        Args:
            output_path: Path to save the combined model
            include_metadata: Whether to include metadata file

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            service = self.get_service('HumanPositioningService')
            if not service:
                return False

            success = service.save_combined_model(output_path, include_metadata)
            if success:
                self.logger.info(f"Combined model saved to: {output_path}")

            return success

        except Exception as e:
            self.logger.error(f"Error saving combined model: {str(e)}")
            return False

    def combine_with_point_cloud(self) -> Optional[np.ndarray]:
        """
        Get the combined human model and point cloud data.

        Returns:
            Combined point cloud array or None if failed
        """
        try:
            service = self.get_service('HumanPositioningService')
            if not service:
                return None

            return service.combine_with_point_cloud()

        except Exception as e:
            self.logger.error(f"Error combining with point cloud: {str(e)}")
            return None

    def get_supported_model_types(self) -> List[str]:
        """
        Get list of supported human model types.

        Returns:
            List of model type names
        """
        try:
            # This could come from service configuration or be hardcoded
            # For now, return the standard types
            return ['seated', 'standing']

        except Exception as e:
            self.logger.error(f"Error getting supported model types: {str(e)}")
            return ['seated', 'standing']  # Fallback

    def reset_model(self) -> None:
        """Reset the current human model."""
        try:
            service = self.get_service('HumanPositioningService')
            if service:
                service.reset_model()

            # Reset controller parameters to defaults
            self._current_model_parameters = {
                'position': (0.0, 0.0, 0.0),
                'scale': 1.0,
                'rotation': 0.0,
                'model_type': 'seated'
            }

            self.logger.info("Human model reset")

        except Exception as e:
            self.logger.error(f"Error resetting model: {str(e)}")

    def has_point_cloud_data(self) -> bool:
        """Check if point cloud data is loaded."""
        return self._point_cloud_data is not None

    def has_current_model(self) -> bool:
        """Check if there's a current human model."""
        try:
            service = self.get_service('HumanPositioningService')
            if not service:
                return False

            return service.has_current_model()

        except Exception as e:
            self.logger.error(f"Error checking current model: {str(e)}")
            return False

    def get_point_cloud_count(self) -> int:
        """Get the number of points in the current point cloud."""
        try:
            service = self.get_service('HumanPositioningService')
            if not service:
                return 0

            return service.get_point_cloud_count()

        except Exception as e:
            self.logger.error(f"Error getting point cloud count: {str(e)}")
            return 0

    def get_current_model_parameters(self) -> Dict[str, Any]:
        """Get the current model parameters."""
        return self._current_model_parameters.copy()

    def set_model_parameters(self, parameters: Dict[str, Any]) -> bool:
        """
        Set model parameters and regenerate model if one exists.

        Args:
            parameters: Dictionary with parameter updates

        Returns:
            True if successful, False otherwise
        """
        try:
            # Update current parameters
            self._current_model_parameters.update(parameters)

            # If we have a current model, regenerate it with new parameters
            if self.has_current_model():
                result = self.generate_human_model(
                    position=self._current_model_parameters['position'],
                    scale=self._current_model_parameters['scale'],
                    rotation=self._current_model_parameters['rotation'],
                    model_type=self._current_model_parameters['model_type']
                )
                return result is not None and result.has_model

            return True

        except Exception as e:
            self.logger.error(f"Error setting model parameters: {str(e)}")
            return False

    def get_controller_status(self) -> Dict[str, Any]:
        """
        Get detailed controller status information.

        Returns:
            Dictionary with status information
        """
        base_info = self.get_controller_info()

        status_info = {
            'has_point_cloud_data': self.has_point_cloud_data(),
            'point_cloud_count': self.get_point_cloud_count(),
            'has_current_model': self.has_current_model(),
            'current_model_parameters': self._current_model_parameters,
            'supported_model_types': self.get_supported_model_types()
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
            # This would typically be used with a cube selection service
            # or file loading service that has point cloud data
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