#!/usr/bin/env python3
"""
Human Positioning Service

Pure business logic for human model generation, positioning, and transformation
without GUI dependencies. This service handles all human model algorithms,
geometric calculations, and point cloud operations.

Author: Dimitrije Stojanovic
Date: December 2025
Extracted: From interactive_human_positioner.py (Phase 3 Refactoring)
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from pathlib import Path
import json

from ..config import HumanModelConfig
from ..error_handling import with_error_handling, validate_point_cloud_data
from .base_service import BaseService, ServiceResult


class HumanModelResult:
    """Data class for human model positioning results."""

    def __init__(self, human_points: Optional[np.ndarray] = None,
                 position: Optional[Tuple[float, float, float]] = None,
                 scale: float = 1.0,
                 rotation: float = 0.0,
                 metadata: Optional[Dict[str, Any]] = None):
        self.human_points = human_points
        self.position = position
        self.scale = scale
        self.rotation = rotation
        self.metadata = metadata or {}

    @property
    def has_model(self) -> bool:
        """Check if there's a valid human model."""
        return self.human_points is not None and len(self.human_points) > 0

    @property
    def point_count(self) -> int:
        """Get number of human model points."""
        return len(self.human_points) if self.human_points is not None else 0


class HumanModelGenerator:
    """Generate realistic human model point clouds."""

    @staticmethod
    def generate_seated_human(center_position: Tuple[float, float, float],
                            scale: float = 1.0,
                            rotation_deg: float = 0.0,
                            num_points: int = 1200,
                            config: Optional[HumanModelConfig] = None) -> np.ndarray:
        """
        Generate a seated human model with realistic proportions.

        Args:
            center_position: (x, y, z) center position for the human
            scale: Scale factor for the human size
            rotation_deg: Rotation angle in degrees around Z-axis
            num_points: Number of points to generate
            config: Optional HumanModelConfig for proportions

        Returns:
            numpy array of human model points
        """
        if config is None:
            config = HumanModelConfig()

        x_center, y_center, z_center = center_position
        proportions = config.PROPORTIONS

        # Extract proportions with scaling
        torso_height = 0.6 * scale
        torso_width = 0.4 * scale
        torso_depth = 0.25 * scale

        head_radius = proportions['head_radii'][0] * scale
        head_height = proportions['head_height_offset'] * scale

        leg_height = 0.4 * scale
        leg_width = 0.15 * scale

        arm_length = proportions['arm_length'] * scale
        arm_width = proportions['upper_arm_radius'] * 2 * scale

        points = []

        # Generate head (sphere at top)
        head_center_z = z_center + torso_height + head_height/2
        for _ in range(int(num_points * 0.1)):  # 10% of points for head
            theta = np.random.uniform(0, 2*np.pi)
            phi = np.random.uniform(0, np.pi)
            r = np.random.uniform(0, head_radius)

            x = x_center + r * np.sin(phi) * np.cos(theta)
            y = y_center + r * np.sin(phi) * np.sin(theta)
            z = head_center_z + r * np.cos(phi)

            points.append([x, y, z])

        # Generate torso (rectangular body)
        for _ in range(int(num_points * 0.4)):  # 40% of points for torso
            x = x_center + np.random.uniform(-torso_width/2, torso_width/2)
            y = y_center + np.random.uniform(-torso_depth/2, torso_depth/2)
            z = z_center + np.random.uniform(0, torso_height)

            points.append([x, y, z])

        # Generate arms (extending from torso)
        for _ in range(int(num_points * 0.2)):  # 20% of points for arms
            # Left arm
            arm_x = x_center + np.random.uniform(-torso_width/2 - arm_length, -torso_width/2)
            arm_y = y_center + np.random.uniform(-arm_width/2, arm_width/2)
            arm_z = z_center + np.random.uniform(torso_height*0.6, torso_height*0.9)
            points.append([arm_x, arm_y, arm_z])

            # Right arm
            arm_x = x_center + np.random.uniform(torso_width/2, torso_width/2 + arm_length)
            points.append([arm_x, arm_y, arm_z])

        # Generate legs (seated position - legs bent)
        for _ in range(int(num_points * 0.3)):  # 30% of points for legs
            # Left leg (bent forward)
            leg_x = x_center + np.random.uniform(-torso_width/4, 0)
            leg_y = y_center + np.random.uniform(torso_depth/2, torso_depth/2 + leg_height)
            leg_z = z_center + np.random.uniform(0, leg_height)
            points.append([leg_x, leg_y, leg_z])

            # Right leg
            leg_x = x_center + np.random.uniform(0, torso_width/4)
            points.append([leg_x, leg_y, leg_z])

        points = np.array(points)

        # Apply rotation around Z-axis if specified
        if rotation_deg != 0:
            points = HumanModelGenerator._rotate_points(points, center_position, rotation_deg)

        # Add noise for realism
        noise_level = config.DEFAULT_NOISE_LEVEL * scale
        noise = np.random.normal(0, noise_level, points.shape)
        points += noise

        return points

    @staticmethod
    def generate_standing_human(center_position: Tuple[float, float, float],
                              scale: float = 1.0,
                              rotation_deg: float = 0.0,
                              num_points: int = 1500,
                              config: Optional[HumanModelConfig] = None) -> np.ndarray:
        """
        Generate a standing human model with realistic proportions.

        Args:
            center_position: (x, y, z) center position for the human
            scale: Scale factor for the human size
            rotation_deg: Rotation angle in degrees around Z-axis
            num_points: Number of points to generate
            config: Optional HumanModelConfig for proportions

        Returns:
            numpy array of human model points
        """
        if config is None:
            config = HumanModelConfig()

        x_center, y_center, z_center = center_position
        proportions = config.PROPORTIONS

        # Standing human proportions
        total_height = 1.75 * scale  # Average human height
        head_height = proportions['head_height_offset'] * scale
        torso_height = 0.65 * scale
        leg_height = 0.9 * scale

        points = []

        # Head
        head_center_z = z_center + total_height - head_height/2
        for _ in range(int(num_points * 0.12)):
            theta = np.random.uniform(0, 2*np.pi)
            phi = np.random.uniform(0, np.pi)
            r = np.random.uniform(0, proportions['head_radii'][0] * scale)

            x = x_center + r * np.sin(phi) * np.cos(theta)
            y = y_center + r * np.sin(phi) * np.sin(theta)
            z = head_center_z + r * np.cos(phi)

            points.append([x, y, z])

        # Torso
        torso_width = proportions['torso_radii'][0] * 2 * scale
        torso_depth = proportions['torso_radii'][2] * 2 * scale

        for _ in range(int(num_points * 0.35)):
            x = x_center + np.random.uniform(-torso_width/2, torso_width/2)
            y = y_center + np.random.uniform(-torso_depth/2, torso_depth/2)
            z = z_center + leg_height + np.random.uniform(0, torso_height)

            points.append([x, y, z])

        # Arms
        arm_length = proportions['arm_length'] * scale
        for _ in range(int(num_points * 0.25)):
            # Left arm
            arm_x = x_center + np.random.uniform(-torso_width/2 - arm_length/2, -torso_width/2)
            arm_y = y_center + np.random.uniform(-torso_depth/4, torso_depth/4)
            arm_z = z_center + leg_height + np.random.uniform(torso_height*0.3, torso_height*0.8)
            points.append([arm_x, arm_y, arm_z])

            # Right arm
            arm_x = x_center + np.random.uniform(torso_width/2, torso_width/2 + arm_length/2)
            points.append([arm_x, arm_y, arm_z])

        # Legs
        leg_width = proportions['thigh_radius'] * 2 * scale
        for _ in range(int(num_points * 0.28)):
            # Left leg
            leg_x = x_center + np.random.uniform(-torso_width/4, -leg_width/2)
            leg_y = y_center + np.random.uniform(-leg_width/2, leg_width/2)
            leg_z = z_center + np.random.uniform(0, leg_height)
            points.append([leg_x, leg_y, leg_z])

            # Right leg
            leg_x = x_center + np.random.uniform(leg_width/2, torso_width/4)
            points.append([leg_x, leg_y, leg_z])

        points = np.array(points)

        # Apply rotation if specified
        if rotation_deg != 0:
            points = HumanModelGenerator._rotate_points(points, center_position, rotation_deg)

        # Add realistic noise
        noise_level = config.DEFAULT_NOISE_LEVEL * scale
        noise = np.random.normal(0, noise_level, points.shape)
        points += noise

        return points

    @staticmethod
    def _rotate_points(points: np.ndarray, center: Tuple[float, float, float],
                      angle_deg: float) -> np.ndarray:
        """
        Rotate points around a center point by angle around Z-axis.

        Args:
            points: Points to rotate
            center: Center of rotation
            angle_deg: Angle in degrees

        Returns:
            Rotated points
        """
        angle_rad = np.radians(angle_deg)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

        # Translate to origin
        translated = points - np.array(center)

        # Rotation matrix for Z-axis
        rotation_matrix = np.array([
            [cos_a, -sin_a, 0],
            [sin_a,  cos_a, 0],
            [0,      0,     1]
        ])

        # Apply rotation
        rotated = np.dot(translated, rotation_matrix.T)

        # Translate back
        return rotated + np.array(center)


class HumanPositioningService(BaseService):
    """
    Service for human model positioning and transformation operations.

    This service handles all business logic for human model generation,
    positioning, and integration with point clouds without any GUI dependencies.
    """

    def __init__(self, point_cloud_data: Optional[np.ndarray] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the human positioning service.

        Args:
            point_cloud_data: Point cloud data to position humans within
            config: Optional configuration dictionary
        """
        super().__init__(config)

        if point_cloud_data is not None:
            validate_point_cloud_data(point_cloud_data)  # This validates but doesn't return
            self.point_cloud_data = point_cloud_data  # Set the validated data
        else:
            self.point_cloud_data = np.array([]).reshape(0, 3)  # Empty point cloud

        self.current_model = HumanModelResult()

    @with_error_handling("calculate_optimal_position")
    def calculate_optimal_position(self) -> Tuple[float, float, float]:
        """
        Calculate optimal position for human model based on point cloud geometry.

        Returns:
            Optimal (x, y, z) position for human model
        """
        if self.point_cloud_data is None or len(self.point_cloud_data) == 0:
            return (0.0, 0.0, 0.0)

        # Calculate point cloud bounds
        bounds = self.get_point_cloud_bounds()

        # Position human at center of point cloud with some adjustments
        x_center = (bounds[0] + bounds[1]) / 2
        y_center = (bounds[2] + bounds[3]) / 2

        # For interior spaces, position slightly above the floor
        z_min = bounds[4]
        floor_height = z_min + 0.1  # 10cm above the lowest point

        return (x_center, y_center, floor_height)

    def get_point_cloud_bounds(self) -> Tuple[float, float, float, float, float, float]:
        """
        Get bounds of the point cloud.

        Returns:
            Tuple of (xmin, xmax, ymin, ymax, zmin, zmax)
        """
        if self.point_cloud_data is None or len(self.point_cloud_data) == 0:
            return (0, 0, 0, 0, 0, 0)

        return (
            float(np.min(self.point_cloud_data[:, 0])), float(np.max(self.point_cloud_data[:, 0])),
            float(np.min(self.point_cloud_data[:, 1])), float(np.max(self.point_cloud_data[:, 1])),
            float(np.min(self.point_cloud_data[:, 2])), float(np.max(self.point_cloud_data[:, 2]))
        )

    @with_error_handling("generate_human_model")
    def generate_human_model(self, position: Tuple[float, float, float],
                           scale: float = 1.0,
                           rotation: float = 0.0,
                           model_type: str = "seated",
                           num_points: int = 1200) -> HumanModelResult:
        """
        Generate a human model at the specified position.

        Args:
            position: (x, y, z) position for the human
            scale: Scale factor for the human size
            rotation: Rotation angle in degrees
            model_type: Type of model ("seated" or "standing")
            num_points: Number of points to generate

        Returns:
            HumanModelResult with generated model
        """
        config = HumanModelConfig()

        if model_type == "seated":
            human_points = HumanModelGenerator.generate_seated_human(
                center_position=position,
                scale=scale,
                rotation_deg=rotation,
                num_points=num_points,
                config=config
            )
        elif model_type == "standing":
            human_points = HumanModelGenerator.generate_standing_human(
                center_position=position,
                scale=scale,
                rotation_deg=rotation,
                num_points=num_points,
                config=config
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        metadata = {
            'model_type': model_type,
            'generation_time': np.datetime64('now').item(),
            'point_count': len(human_points),
            'bounds': self._calculate_model_bounds(human_points)
        }

        result = HumanModelResult(
            human_points=human_points,
            position=position,
            scale=scale,
            rotation=rotation,
            metadata=metadata
        )

        self.current_model = result
        return result

    def update_model_position(self, new_position: Tuple[float, float, float]) -> HumanModelResult:
        """
        Update the position of the current model.

        Args:
            new_position: New (x, y, z) position

        Returns:
            Updated HumanModelResult
        """
        if not self.current_model.has_model:
            return self.generate_human_model(new_position)

        # Calculate translation vector
        old_pos = self.current_model.position
        if old_pos is None:
            old_pos = (0, 0, 0)

        translation = np.array(new_position) - np.array(old_pos)

        # Apply translation to all points
        updated_points = self.current_model.human_points + translation

        # Update the model
        self.current_model.human_points = updated_points
        self.current_model.position = new_position
        self.current_model.metadata['bounds'] = self._calculate_model_bounds(updated_points)

        return self.current_model

    def update_model_scale(self, new_scale: float) -> HumanModelResult:
        """
        Update the scale of the current model.

        Args:
            new_scale: New scale factor

        Returns:
            Updated HumanModelResult
        """
        if not self.current_model.has_model:
            return self.generate_human_model((0, 0, 0), scale=new_scale)

        # Regenerate model with new scale at current position
        return self.generate_human_model(
            position=self.current_model.position,
            scale=new_scale,
            rotation=self.current_model.rotation,
            model_type=self.current_model.metadata.get('model_type', 'seated')
        )

    def update_model_rotation(self, new_rotation: float) -> HumanModelResult:
        """
        Update the rotation of the current model.

        Args:
            new_rotation: New rotation angle in degrees

        Returns:
            Updated HumanModelResult
        """
        if not self.current_model.has_model:
            return self.generate_human_model((0, 0, 0), rotation=new_rotation)

        # Regenerate model with new rotation at current position
        return self.generate_human_model(
            position=self.current_model.position,
            scale=self.current_model.scale,
            rotation=new_rotation,
            model_type=self.current_model.metadata.get('model_type', 'seated')
        )

    def combine_with_point_cloud(self) -> np.ndarray:
        """
        Combine the current human model with the point cloud.

        Returns:
            Combined point cloud array
        """
        if not self.current_model.has_model:
            return self.point_cloud_data

        return np.vstack([self.point_cloud_data, self.current_model.human_points])

    def save_combined_model(self, output_path: Path, include_metadata: bool = True) -> bool:
        """
        Save the combined point cloud with human model.

        Args:
            output_path: Path to save the combined model
            include_metadata: Whether to save metadata as JSON

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            combined_points = self.combine_with_point_cloud()
            np.save(output_path, combined_points)

            if include_metadata and self.current_model.has_model:
                metadata_path = output_path.with_suffix('.json')
                metadata = {
                    'total_points': len(combined_points),
                    'original_points': len(self.point_cloud_data),
                    'human_points': self.current_model.point_count,
                    'human_position': self.current_model.position,
                    'human_scale': self.current_model.scale,
                    'human_rotation': self.current_model.rotation,
                    **self.current_model.metadata
                }

                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)

            return True

        except Exception as e:
            print(f"Error saving combined model: {e}")
            return False

    def _calculate_model_bounds(self, points: np.ndarray) -> Dict[str, List[float]]:
        """Calculate bounds of the human model."""
        if len(points) == 0:
            return {'x': [0, 0], 'y': [0, 0], 'z': [0, 0]}

        return {
            'x': [float(np.min(points[:, 0])), float(np.max(points[:, 0]))],
            'y': [float(np.min(points[:, 1])), float(np.max(points[:, 1]))],
            'z': [float(np.min(points[:, 2])), float(np.max(points[:, 2]))]
        }

    def get_model_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the current human model.

        Returns:
            Dictionary with model statistics
        """
        if not self.current_model.has_model:
            return {'error': 'No current model'}

        points = self.current_model.human_points

        return {
            'point_count': len(points),
            'position': self.current_model.position,
            'scale': self.current_model.scale,
            'rotation': self.current_model.rotation,
            'centroid': [float(np.mean(points[:, i])) for i in range(3)],
            'bounds': self._calculate_model_bounds(points),
            'density': len(points) / self._calculate_model_volume(),
            **self.current_model.metadata
        }

    def _calculate_model_volume(self) -> float:
        """Calculate approximate volume of the human model."""
        if not self.current_model.has_model:
            return 0.0

        bounds = self._calculate_model_bounds(self.current_model.human_points)
        volume = (
            (bounds['x'][1] - bounds['x'][0]) *
            (bounds['y'][1] - bounds['y'][0]) *
            (bounds['z'][1] - bounds['z'][0])
        )

        return max(volume, 0.001)  # Avoid division by zero

    def reset_model(self) -> None:
        """Reset the current human model."""
        self.current_model = HumanModelResult()

    def has_current_model(self) -> bool:
        """Check if there's a current human model."""
        return self.current_model.has_model

    def set_point_cloud_data(self, point_cloud_data: np.ndarray) -> ServiceResult:
        """
        Set new point cloud data for the service.

        Args:
            point_cloud_data: New point cloud data

        Returns:
            ServiceResult indicating success/failure
        """
        try:
            validate_point_cloud_data(point_cloud_data)  # This validates but doesn't return
            self.point_cloud_data = point_cloud_data  # Set the validated data
            return self.create_success_result(
                data=f"Loaded {len(self.point_cloud_data)} points",
                point_count=len(self.point_cloud_data)
            )
        except Exception as e:
            return self.create_error_result(f"Error setting point cloud data: {str(e)}")

    def get_point_cloud_count(self) -> int:
        """Get the number of points in the current point cloud."""
        return len(self.point_cloud_data)

    def get_service_info(self) -> Dict[str, Any]:
        """
        Get service information and capabilities.

        Returns:
            Dictionary with service information
        """
        return {
            'service_name': 'HumanPositioningService',
            'description': 'Human model generation, positioning, and transformation operations',
            'capabilities': [
                'human_model_generation',
                'optimal_positioning',
                'model_transformation',
                'point_cloud_integration',
                'model_statistics',
                'file_operations'
            ],
            'current_state': {
                'point_cloud_count': len(self.point_cloud_data),
                'has_model': self.has_current_model(),
                'model_point_count': self.current_model.point_count if self.has_current_model() else 0,
                'model_position': self.current_model.position if self.has_current_model() else None,
                'model_scale': self.current_model.scale if self.has_current_model() else None,
                'model_rotation': self.current_model.rotation if self.has_current_model() else None
            },
            'supported_operations': {
                'calculate_optimal_position': 'Calculate optimal position for human model',
                'generate_human_model': 'Generate human model with specified parameters',
                'update_model_position': 'Update human model position',
                'update_model_scale': 'Update human model scale',
                'update_model_rotation': 'Update human model rotation',
                'combine_with_point_cloud': 'Combine human model with point cloud',
                'get_model_statistics': 'Get comprehensive model statistics',
                'save_combined_model': 'Save combined model to file'
            },
            'supported_model_types': [
                'seated',
                'standing'
            ],
            'configuration': {
                'default_scale': self.get_config_value('default_scale', 1.0),
                'default_model_type': self.get_config_value('default_model_type', 'seated'),
                'num_points': self.get_config_value('num_points', 1200),
                'position_optimization': self.get_config_value('position_optimization', True)
            }
        }