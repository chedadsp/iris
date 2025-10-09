#!/usr/bin/env python3
"""
Cockpit Refiner Processor

Refines interior detection to specifically identify cockpit areas
based on vehicle geometry and dashboard detection.

Author: Dimitrije Stojanovic
Date: September 2025
"""

import numpy as np

from ..interfaces.processors import AnalysisStep
from ..models.point_cloud_data import AnalysisResults
from ..config import AnalysisConfig


class CockpitRefiner(AnalysisStep):
    """Refines interior points to focus on cockpit areas."""

    def __init__(self, config: AnalysisConfig):
        self.config = config

    @property
    def name(self) -> str:
        return "Cockpit Refinement"

    def validate_input(self, results: AnalysisResults) -> bool:
        """Validate that we have vehicle and interior points."""
        return (results.has_vehicle_detection and
                results.has_interior_detection and
                len(results.interior_points) > 0)

    def analyze(self, results: AnalysisResults, **kwargs) -> AnalysisResults:
        """
        Refine interior points to specifically identify cockpit areas.

        Args:
            results: Current analysis results
            **kwargs: Additional parameters

        Returns:
            Updated analysis results with refined cockpit points
        """
        if not self.validate_input(results):
            print("No interior points available for cockpit refinement")
            return results

        print("Refining cockpit detection...")

        vehicle_points = results.vehicle_points
        interior_points = results.interior_points

        # Look for typical cockpit characteristics
        z_min = vehicle_points[:, 2].min()

        # Detect dashboard area
        dashboard_candidates = self._detect_dashboard_area(vehicle_points, z_min)

        if len(dashboard_candidates) == 0:
            print("No dashboard area detected - keeping original interior points")
            return results

        print(f"Found {len(dashboard_candidates):,} dashboard candidate points")

        # Determine vehicle orientation
        vehicle_orientation = self._analyze_vehicle_orientation(vehicle_points, dashboard_candidates)

        # Refine interior points based on cockpit geometry
        refined_interior = self._refine_cockpit_geometry(
            interior_points, vehicle_points, vehicle_orientation, z_min
        )

        if len(refined_interior) > 0:
            results.interior_points = refined_interior
            print(f"Refined to {len(refined_interior):,} cockpit-specific points")
        else:
            print("No points passed cockpit refinement, keeping original interior points")

        return results

    def _detect_dashboard_area(self, vehicle_points: np.ndarray, z_min: float) -> np.ndarray:
        """Detect dashboard area based on height and density."""
        dashboard_height_min = z_min + self.config.DASHBOARD_HEIGHT_MIN
        dashboard_height_max = z_min + self.config.DASHBOARD_HEIGHT_MAX

        dashboard_mask = ((vehicle_points[:, 2] >= dashboard_height_min) &
                         (vehicle_points[:, 2] <= dashboard_height_max))

        return vehicle_points[dashboard_mask]

    def _analyze_vehicle_orientation(self, vehicle_points: np.ndarray,
                                   dashboard_points: np.ndarray) -> dict:
        """Analyze vehicle orientation based on dashboard distribution."""
        vehicle_xy = vehicle_points[:, :2]
        x_range = vehicle_xy[:, 0].max() - vehicle_xy[:, 0].min()
        y_range = vehicle_xy[:, 1].max() - vehicle_xy[:, 1].min()

        # Determine primary axis (longer dimension)
        if x_range > y_range:
            primary_axis = 0
            secondary_axis = 1
        else:
            primary_axis = 1
            secondary_axis = 0

        # Divide vehicle into front and back halves
        axis_min = vehicle_xy[:, primary_axis].min()
        axis_max = vehicle_xy[:, primary_axis].max()
        axis_mid = (axis_min + axis_max) / 2

        # Count dashboard points in each half
        dashboard_xy = dashboard_points[:, :2]
        front_half_mask = dashboard_xy[:, primary_axis] >= axis_mid
        back_half_mask = dashboard_xy[:, primary_axis] < axis_mid

        front_count = np.sum(front_half_mask)
        back_count = np.sum(back_half_mask)

        # The half with more dashboard points is likely the front
        front_is_positive = front_count > back_count

        orientation = {
            'primary_axis': primary_axis,
            'secondary_axis': secondary_axis,
            'axis_min': axis_min,
            'axis_max': axis_max,
            'axis_mid': axis_mid,
            'front_is_positive': front_is_positive,
            'x_range': x_range,
            'y_range': y_range
        }

        print(f"Vehicle orientation analysis:")
        print(f"  Primary axis: {['X', 'Y'][primary_axis]} (range: {axis_max - axis_min:.2f}m)")
        print(f"  Secondary axis: {['X', 'Y'][secondary_axis]} (range: {y_range if primary_axis == 0 else x_range:.2f}m)")
        print(f"  Front direction: {'Positive' if front_is_positive else 'Negative'} {['X', 'Y'][primary_axis]}")

        return orientation

    def _refine_cockpit_geometry(self, interior_points: np.ndarray,
                               vehicle_points: np.ndarray,
                               orientation: dict, z_min: float) -> np.ndarray:
        """Refine interior points based on cockpit geometry."""
        if len(interior_points) == 0:
            return interior_points

        interior_xy = interior_points[:, :2]
        vehicle_xy = vehicle_points[:, :2]

        # Extract orientation parameters
        primary_axis = orientation['primary_axis']
        secondary_axis = orientation['secondary_axis']
        axis_min = orientation['axis_min']
        axis_max = orientation['axis_max']
        front_is_positive = orientation['front_is_positive']

        # Keep points in the front 2/3 of the vehicle (where driver/passenger sit)
        if front_is_positive:
            cockpit_boundary = axis_min + (axis_max - axis_min) * self.config.COCKPIT_FRONT_RATIO
            cockpit_mask = interior_xy[:, primary_axis] >= cockpit_boundary
        else:
            cockpit_boundary = axis_max - (axis_max - axis_min) * self.config.COCKPIT_FRONT_RATIO
            cockpit_mask = interior_xy[:, primary_axis] <= cockpit_boundary

        # Keep points not too far to the sides (center area where seats are)
        secondary_center = vehicle_xy[:, secondary_axis].mean()
        secondary_range = vehicle_xy[:, secondary_axis].max() - vehicle_xy[:, secondary_axis].min()
        side_tolerance = secondary_range * 0.35  # Within 35% of center width

        center_mask = np.abs(interior_xy[:, secondary_axis] - secondary_center) <= side_tolerance

        # Keep points at appropriate height for seated passengers
        seat_mask = ((interior_points[:, 2] >= z_min + 0.6) &
                    (interior_points[:, 2] <= z_min + 1.6))

        # Combine all cockpit constraints
        final_cockpit_mask = cockpit_mask & center_mask & seat_mask

        return interior_points[final_cockpit_mask]