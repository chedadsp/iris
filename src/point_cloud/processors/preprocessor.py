#!/usr/bin/env python3
"""
Point Cloud Preprocessor

Handles initial preprocessing of point cloud data including outlier removal
and noise filtering.

Author: Dimitrije Stojanovic
Date: September 2025
"""

import numpy as np
from typing import Optional

from ..interfaces.processors import AnalysisStep
from ..models.point_cloud_data import PointCloudData, AnalysisResults
from ..config import AnalysisConfig


class PointCloudPreprocessor(AnalysisStep):
    """Preprocessor for point cloud data to remove outliers and noise."""

    def __init__(self, config: AnalysisConfig):
        self.config = config

    @property
    def name(self) -> str:
        return "Point Cloud Preprocessing"

    def analyze(self, results: AnalysisResults, **kwargs) -> AnalysisResults:
        """
        Preprocess point cloud data by removing outliers and noise.

        Args:
            results: Current analysis results
            **kwargs: Additional parameters

        Returns:
            Updated analysis results with preprocessed data
        """
        print("Preprocessing points...")

        data = results.raw_data

        # Remove points that are too far from the origin (likely noise)
        distances = np.linalg.norm(data.points, axis=1)
        distance_threshold = np.percentile(
            distances,
            getattr(self.config, 'DISTANCE_PERCENTILE_THRESHOLD', 95)
        )
        valid_indices = distances < distance_threshold

        # Apply filtering to all data arrays
        filtered_points = data.points[valid_indices]
        filtered_colors = data.colors[valid_indices] if data.colors is not None else None
        filtered_intensity = data.intensity[valid_indices] if data.intensity is not None else None

        # Update metadata
        updated_metadata = data.metadata.copy()
        updated_metadata.update({
            'preprocessing_applied': True,
            'original_point_count': data.size,
            'distance_threshold': distance_threshold,
            'outliers_removed': data.size - len(filtered_points)
        })

        print(f"After preprocessing: {len(filtered_points):,} points "
              f"(removed {data.size - len(filtered_points):,} outliers)")

        # Update the raw data with preprocessed points
        results.raw_data = PointCloudData(
            points=filtered_points,
            colors=filtered_colors,
            intensity=filtered_intensity,
            metadata=updated_metadata
        )

        return results