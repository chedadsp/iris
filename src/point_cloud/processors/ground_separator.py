#!/usr/bin/env python3
"""
Ground Separation Processor

Separates ground points from non-ground points using RANSAC plane fitting
with fallback to simple height-based thresholding.

Author: Dimitrije Stojanovic
Date: September 2025
"""

import numpy as np
from typing import Tuple, Optional

from ..interfaces.processors import AnalysisStep
from ..models.point_cloud_data import AnalysisResults
from ..config import AnalysisConfig
from ..error_handling import PointCloudError


class GroundSeparator(AnalysisStep):
    """Separates ground points from non-ground points."""

    def __init__(self, config: AnalysisConfig):
        self.config = config

    @property
    def name(self) -> str:
        return "Ground Separation"

    def validate_input(self, results: AnalysisResults) -> bool:
        """Validate that we have point data to process."""
        return (results.raw_data is not None and
                results.raw_data.points is not None and
                len(results.raw_data.points) > 0)

    def analyze(self, results: AnalysisResults, **kwargs) -> AnalysisResults:
        """
        Separate ground points from non-ground points.

        Args:
            results: Current analysis results
            **kwargs: Additional parameters

        Returns:
            Updated analysis results with ground separation
        """
        if not self.validate_input(results):
            raise PointCloudError("Invalid input: no point data available for ground separation")

        print("Separating ground from non-ground points...")

        points = results.raw_data.points

        try:
            # Try RANSAC-based separation first
            ground_points, non_ground_points = self._ransac_separation(points)
            method_used = "RANSAC"
        except Exception as e:
            print(f"RANSAC failed: {e}, using simple height threshold")
            ground_points, non_ground_points = self._simple_separation(points)
            method_used = "Simple Height Threshold"

        # Update results
        results.ground_points = ground_points
        results.non_ground_points = non_ground_points

        print(f"{method_used} ground separation: {len(ground_points):,} ground points, "
              f"{len(non_ground_points):,} non-ground points")

        return results

    def _ransac_separation(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Separate ground using RANSAC plane fitting.

        Args:
            points: Input point cloud (Nx3)

        Returns:
            Tuple of (ground_points, non_ground_points)
        """
        try:
            from sklearn.linear_model import RANSACRegressor
        except ImportError:
            raise PointCloudError("scikit-learn not available for RANSAC ground separation")

        # Get candidate ground points (lowest points)
        z_min = points[:, 2].min()
        z_threshold = z_min + self.config.GROUND_HEIGHT_THRESHOLD

        ground_candidates_mask = points[:, 2] <= z_threshold
        ground_candidates = points[ground_candidates_mask]

        if len(ground_candidates) < self.config.RANSAC_MIN_SAMPLES:
            raise PointCloudError(
                f"Not enough ground candidates ({len(ground_candidates)}) "
                f"for RANSAC (minimum {self.config.RANSAC_MIN_SAMPLES})"
            )

        # Fit plane using RANSAC (Z = aX + bY + c)
        X = ground_candidates[:, :2]  # X, Y coordinates
        y = ground_candidates[:, 2]   # Z coordinates

        ransac = RANSACRegressor(
            random_state=42,
            residual_threshold=self.config.GROUND_PLANE_TOLERANCE,
            max_trials=self.config.RANSAC_MAX_TRIALS
        )
        ransac.fit(X, y)

        # Predict ground height for all points
        predicted_ground_z = ransac.predict(points[:, :2])

        # Points within threshold of predicted ground are ground points
        height_above_ground = points[:, 2] - predicted_ground_z
        ground_mask = height_above_ground <= self.config.GROUND_FINAL_THRESHOLD

        ground_points = points[ground_mask]
        non_ground_points = points[~ground_mask]

        return ground_points, non_ground_points

    def _simple_separation(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simple ground separation using height threshold.

        Args:
            points: Input point cloud (Nx3)

        Returns:
            Tuple of (ground_points, non_ground_points)
        """
        z_min = points[:, 2].min()
        ground_threshold = z_min + self.config.SIMPLE_GROUND_THRESHOLD

        ground_mask = points[:, 2] <= ground_threshold

        ground_points = points[ground_mask]
        non_ground_points = points[~ground_mask]

        return ground_points, non_ground_points