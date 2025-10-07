#!/usr/bin/env python3
"""
Unified Human Model Generation - Consolidated human model implementations.

Combines and unifies the two HumanModelGenerator classes from:
- interactive_human_positioner.py (simple geometric implementation)
- add_human_model.py (sophisticated ellipsoid/cylinder implementation)

This eliminates code duplication and provides a single, comprehensive human model API.

Author: Dimitrije Stojanovic
Date: September 2025
Refactored: November 2025 (Code Deduplication)
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum


class HumanModelType(Enum):
    """Available human model types."""
    SIMPLE_SEATED = "simple_seated"
    DETAILED_SEATED = "detailed_seated"
    PARAMETRIC_SEATED = "parametric_seated"


@dataclass
class HumanModelConfig:
    """Configuration for human model generation."""
    center_position: Tuple[float, float, float]
    scale: float = 1.0
    rotation_deg: float = 0.0
    num_points: int = 1200
    noise_level: float = 0.02
    model_type: HumanModelType = HumanModelType.SIMPLE_SEATED

    def __post_init__(self):
        """Validate configuration parameters."""
        if len(self.center_position) != 3:
            raise ValueError("center_position must be a 3-element tuple")

        if self.scale <= 0:
            raise ValueError("scale must be positive")

        if self.num_points <= 0:
            raise ValueError("num_points must be positive")

        if self.noise_level < 0:
            raise ValueError("noise_level must be non-negative")


class BaseHumanGenerator(ABC):
    """Abstract base class for human model generators."""

    @abstractmethod
    def generate_model(self, config: HumanModelConfig) -> np.ndarray:
        """
        Generate human model points.

        Args:
            config: Human model configuration

        Returns:
            Array of human model points
        """
        pass

    def add_noise_and_variation(self, points: np.ndarray, noise_level: float) -> np.ndarray:
        """Add realistic noise to make the human model look more natural."""
        if noise_level <= 0:
            return points

        noise = np.random.normal(0, noise_level, points.shape)
        return points + noise

    def apply_rotation(self, points: np.ndarray, center: Tuple[float, float, float],
                      rotation_deg: float) -> np.ndarray:
        """Apply Z-axis rotation to points."""
        if rotation_deg == 0.0:
            return points

        from ..spatial.geometric_transforms import GeometricTransformer
        return GeometricTransformer.rotate_points_z_axis(points, center, rotation_deg)


class SimpleSeatedHumanGenerator(BaseHumanGenerator):
    """
    Simple geometric human model generator.

    Based on the implementation from interactive_human_positioner.py.
    Uses basic geometric shapes with random point distribution.
    """

    def generate_model(self, config: HumanModelConfig) -> np.ndarray:
        """Generate simple seated human model."""
        x_center, y_center, z_center = config.center_position

        # Human body proportions (seated) - scaled by config.scale
        scale = config.scale
        torso_height = 0.6 * scale
        torso_width = 0.4 * scale
        torso_depth = 0.25 * scale

        head_radius = 0.1 * scale
        head_height = 0.15 * scale

        leg_height = 0.4 * scale
        leg_width = 0.15 * scale

        arm_length = 0.5 * scale
        arm_width = 0.08 * scale

        points = []
        num_points = config.num_points

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
            # Upper legs (thighs) - horizontal when seated
            thigh_x = x_center + np.random.uniform(-leg_width, leg_width)
            thigh_y = y_center + np.random.uniform(0, leg_height)
            thigh_z = z_center + np.random.uniform(-0.1*scale, 0.1*scale)
            points.append([thigh_x, thigh_y, thigh_z])

            # Lower legs (shins) - vertical when seated
            shin_x = x_center + np.random.uniform(-leg_width/2, leg_width/2)
            shin_y = y_center + leg_height + np.random.uniform(-0.05*scale, 0.05*scale)
            shin_z = z_center + np.random.uniform(-leg_height, 0)
            points.append([shin_x, shin_y, shin_z])

        # Convert to array and add noise
        points = np.array(points)
        points = self.add_noise_and_variation(points, config.noise_level * scale)

        # Apply rotation if specified
        if config.rotation_deg != 0.0:
            points = self.apply_rotation(points, config.center_position, config.rotation_deg)

        return points


class DetailedSeatedHumanGenerator(BaseHumanGenerator):
    """
    Detailed human model generator using ellipsoids and cylinders.

    Based on the implementation from add_human_model.py.
    Provides more realistic human geometry using mathematical shapes.
    """

    def generate_ellipsoid_points(self, center: Tuple[float, float, float],
                                radii: Tuple[float, float, float],
                                num_points: int, z_range: Optional[Tuple[float, float]] = None) -> np.ndarray:
        """Generate points on an ellipsoid surface."""
        # Generate random angles
        u = np.random.uniform(0, 2 * np.pi, num_points)
        v = np.random.uniform(0, np.pi, num_points)

        # Convert to cartesian coordinates
        x = radii[0] * np.sin(v) * np.cos(u) + center[0]
        y = radii[1] * np.sin(v) * np.sin(u) + center[1]
        z = radii[2] * np.cos(v) + center[2]

        # Filter z range if specified
        if z_range is not None:
            mask = (z >= z_range[0]) & (z <= z_range[1])
            x, y, z = x[mask], y[mask], z[mask]

        return np.column_stack([x, y, z])

    def generate_cylinder_points(self, center: Tuple[float, float, float],
                               radius: float, height: float, num_points: int) -> np.ndarray:
        """Generate points on a cylinder surface."""
        # Generate random angles and heights
        theta = np.random.uniform(0, 2 * np.pi, num_points)
        z = np.random.uniform(center[2] - height/2, center[2] + height/2, num_points)

        # Convert to cartesian coordinates
        x = radius * np.cos(theta) + center[0]
        y = radius * np.sin(theta) + center[1]

        return np.column_stack([x, y, z])

    def generate_model(self, config: HumanModelConfig) -> np.ndarray:
        """Generate detailed seated human model."""
        seat_x, seat_y, seat_z = config.center_position
        scale = config.scale

        # Scale point counts based on density
        base_density = max(10, int(100 * (config.num_points / 1200)))  # Normalize to expected count

        human_parts = []

        # 1. Head (ellipsoid)
        head_center = (seat_x, seat_y, seat_z + 0.65 * scale)
        head_radii = (0.10 * scale, 0.12 * scale, 0.11 * scale)
        head_points = self.generate_ellipsoid_points(head_center, head_radii, base_density)
        human_parts.append(head_points)

        # 2. Neck (small cylinder)
        neck_center = (seat_x, seat_y, seat_z + 0.55 * scale)
        neck_points = self.generate_cylinder_points(
            neck_center, 0.06 * scale, 0.08 * scale, base_density // 4
        )
        human_parts.append(neck_points)

        # 3. Torso (ellipsoid, upper body)
        torso_center = (seat_x, seat_y, seat_z + 0.35 * scale)
        torso_radii = (0.18 * scale, 0.12 * scale, 0.25 * scale)
        torso_points = self.generate_ellipsoid_points(torso_center, torso_radii, base_density * 2)
        human_parts.append(torso_points)

        # 4. Shoulders (extend torso width)
        for side in [-1, 1]:
            shoulder_pos = (seat_x + side * 0.20 * scale, seat_y, seat_z + 0.50 * scale)
            shoulder_points = self.generate_ellipsoid_points(
                shoulder_pos, (0.08 * scale, 0.08 * scale, 0.08 * scale), base_density // 2
            )
            human_parts.append(shoulder_points)

        # 5. Arms (cylinders from shoulders)
        arm_length = 0.30 * scale
        for side in [-1, 1]:  # Left and right
            # Upper arm
            upper_arm_center = (seat_x + side * 0.25 * scale, seat_y, seat_z + 0.35 * scale)
            upper_arm_points = self.generate_cylinder_points(
                upper_arm_center, 0.05 * scale, arm_length, base_density // 2
            )
            human_parts.append(upper_arm_points)

            # Forearm (bent at elbow)
            forearm_center = (seat_x + side * 0.30 * scale, seat_y + 0.15 * scale, seat_z + 0.20 * scale)
            forearm_points = self.generate_cylinder_points(
                forearm_center, 0.04 * scale, 0.25 * scale, base_density // 2
            )
            human_parts.append(forearm_points)

            # Hands
            hand_center = (seat_x + side * 0.35 * scale, seat_y + 0.25 * scale, seat_z + 0.15 * scale)
            hand_points = self.generate_ellipsoid_points(
                hand_center, (0.04 * scale, 0.08 * scale, 0.02 * scale), base_density // 4
            )
            human_parts.append(hand_points)

        # 6. Seated lower torso/hips
        hip_center = (seat_x, seat_y, seat_z)
        hip_radii = (0.15 * scale, 0.20 * scale, 0.10 * scale)
        hip_points = self.generate_ellipsoid_points(hip_center, hip_radii, base_density)
        human_parts.append(hip_points)

        # 7. Thighs (horizontal cylinders for seated position)
        thigh_length = 0.35 * scale
        for side in [-1, 1]:  # Left and right thigh
            thigh_center = (seat_x + side * 0.08 * scale, seat_y + 0.15 * scale, seat_z - 0.05 * scale)
            # Generate points along the thigh
            theta = np.random.uniform(0, 2 * np.pi, base_density)
            r = np.random.uniform(0, 0.08 * scale, base_density)  # Thigh radius
            y_offset = np.random.uniform(0, thigh_length, base_density)

            x = r * np.cos(theta) + thigh_center[0]
            y = y_offset + thigh_center[1]
            z = r * np.sin(theta) + thigh_center[2]

            thigh_points = np.column_stack([x, y, z])
            human_parts.append(thigh_points)

        # 8. Knees
        for side in [-1, 1]:
            knee_center = (seat_x + side * 0.08 * scale, seat_y + 0.45 * scale, seat_z - 0.05 * scale)
            knee_points = self.generate_ellipsoid_points(
                knee_center, (0.06 * scale, 0.06 * scale, 0.06 * scale), base_density // 3
            )
            human_parts.append(knee_points)

        # 9. Lower legs (vertical cylinders)
        lower_leg_height = 0.30 * scale
        for side in [-1, 1]:
            lower_leg_center = (seat_x + side * 0.08 * scale, seat_y + 0.45 * scale, seat_z - 0.20 * scale)
            lower_leg_points = self.generate_cylinder_points(
                lower_leg_center, 0.05 * scale, lower_leg_height, base_density // 2
            )
            human_parts.append(lower_leg_points)

        # 10. Feet
        for side in [-1, 1]:
            foot_center = (seat_x + side * 0.08 * scale, seat_y + 0.55 * scale, seat_z - 0.35 * scale)
            foot_points = self.generate_ellipsoid_points(
                foot_center, (0.05 * scale, 0.12 * scale, 0.04 * scale), base_density // 3
            )
            human_parts.append(foot_points)

        # Combine all points
        all_human_points = np.vstack(human_parts)

        # Add noise for realism
        all_human_points = self.add_noise_and_variation(all_human_points, config.noise_level * scale)

        # Apply rotation if specified
        if config.rotation_deg != 0.0:
            all_human_points = self.apply_rotation(all_human_points, config.center_position, config.rotation_deg)

        return all_human_points


class ParametricSeatedHumanGenerator(BaseHumanGenerator):
    """
    Parametric human model generator with configurable proportions.

    Provides maximum flexibility for different human body types and sizes.
    """

    def __init__(self, body_proportions: Optional[Dict[str, float]] = None):
        """
        Initialize with custom body proportions.

        Args:
            body_proportions: Dictionary of body proportion ratios
        """
        self.proportions = body_proportions or self._default_proportions()

    def _default_proportions(self) -> Dict[str, float]:
        """Get default human body proportions for seated position."""
        return {
            'torso_height_ratio': 0.6,
            'torso_width_ratio': 0.4,
            'torso_depth_ratio': 0.25,
            'head_radius_ratio': 0.1,
            'head_height_ratio': 0.15,
            'leg_height_ratio': 0.4,
            'leg_width_ratio': 0.15,
            'arm_length_ratio': 0.5,
            'arm_width_ratio': 0.08,
            'head_points_ratio': 0.1,
            'torso_points_ratio': 0.4,
            'arm_points_ratio': 0.2,
            'leg_points_ratio': 0.3
        }

    def generate_model(self, config: HumanModelConfig) -> np.ndarray:
        """Generate parametric seated human model."""
        # Use the simple generator as base but with configurable proportions
        generator = SimpleSeatedHumanGenerator()

        # Create a modified config with parametric adjustments
        # For now, delegate to simple generator - can be expanded for full parametric control
        return generator.generate_model(config)


class HumanModelFactory:
    """Factory for creating human model generators."""

    _generators = {
        HumanModelType.SIMPLE_SEATED: SimpleSeatedHumanGenerator,
        HumanModelType.DETAILED_SEATED: DetailedSeatedHumanGenerator,
        HumanModelType.PARAMETRIC_SEATED: ParametricSeatedHumanGenerator,
    }

    @classmethod
    def create_generator(cls, model_type: HumanModelType, **kwargs) -> BaseHumanGenerator:
        """
        Create a human model generator.

        Args:
            model_type: Type of human model to create
            **kwargs: Additional arguments for specific generator types

        Returns:
            Human model generator instance
        """
        if model_type not in cls._generators:
            raise ValueError(f"Unknown model type: {model_type}")

        generator_class = cls._generators[model_type]
        return generator_class(**kwargs)

    @classmethod
    def create_human_model(cls, config: HumanModelConfig) -> np.ndarray:
        """
        Convenience method to create a human model directly.

        Args:
            config: Human model configuration

        Returns:
            Generated human model points
        """
        generator = cls.create_generator(config.model_type)
        return generator.generate_model(config)

    @classmethod
    def available_models(cls) -> list:
        """Get list of available human model types."""
        return list(cls._generators.keys())


class HumanModelPositioner:
    """Helper for positioning human models within point cloud bounds."""

    @staticmethod
    def calculate_optimal_seat_position(point_cloud_bounds: Dict[str, float]) -> Tuple[float, float, float]:
        """
        Calculate optimal seat position for human model.

        Args:
            point_cloud_bounds: Dict with x_min, x_max, y_min, y_max, z_min, z_max

        Returns:
            Optimal seat position (x, y, z)
        """
        # Calculate dimensions
        x_range = point_cloud_bounds['x_max'] - point_cloud_bounds['x_min']
        y_range = point_cloud_bounds['y_max'] - point_cloud_bounds['y_min']

        # Position human in the center-front of cockpit (driver's seat)
        seat_x = point_cloud_bounds['x_min'] + x_range * 0.3  # 30% from front
        seat_y = point_cloud_bounds['y_min'] + y_range * 0.4  # 40% from left (driver side)
        seat_z = point_cloud_bounds['z_max'] - 0.1  # Slightly below roof

        return (seat_x, seat_y, seat_z)

    @staticmethod
    def scale_for_cockpit(point_cloud_bounds: Dict[str, float]) -> float:
        """
        Calculate appropriate scale for human model based on cockpit size.

        Args:
            point_cloud_bounds: Dict with point cloud bounds

        Returns:
            Recommended scale factor
        """
        z_range = point_cloud_bounds['z_max'] - point_cloud_bounds['z_min']

        # Typical seated human height is about 1.3m, scale based on available height
        typical_seated_height = 1.3
        available_height = z_range * 0.8  # Use 80% of available height

        return max(0.5, min(2.0, available_height / typical_seated_height))


# Convenience functions for backward compatibility
def generate_seated_human(center_position: Tuple[float, float, float],
                         scale: float = 1.0, rotation_deg: float = 0.0,
                         num_points: int = 1200,
                         model_type: HumanModelType = HumanModelType.SIMPLE_SEATED) -> np.ndarray:
    """
    Convenience function to generate a seated human model.

    Args:
        center_position: Center position (x, y, z)
        scale: Scale factor
        rotation_deg: Rotation in degrees
        num_points: Number of points to generate
        model_type: Type of human model

    Returns:
        Generated human model points
    """
    config = HumanModelConfig(
        center_position=center_position,
        scale=scale,
        rotation_deg=rotation_deg,
        num_points=num_points,
        model_type=model_type
    )

    return HumanModelFactory.create_human_model(config)