#!/usr/bin/env python3
"""
Vehicle Identification Processor

Identifies points that belong to the vehicle using DBSCAN clustering
and geometric analysis.

Author: Dimitrije Stojanovic
Date: September 2025
"""

import numpy as np
from typing import Optional

from ..interfaces.processors import AnalysisStep
from ..models.point_cloud_data import AnalysisResults
from ..config import AnalysisConfig
from ..error_handling import PointCloudError


class VehicleIdentifier(AnalysisStep):
    """Identifies vehicle points using clustering and geometric analysis."""

    def __init__(self, config: AnalysisConfig):
        self.config = config

    @property
    def name(self) -> str:
        return "Vehicle Identification"

    def validate_input(self, results: AnalysisResults) -> bool:
        """Validate that we have point data to process."""
        return (results.raw_data is not None and
                results.raw_data.points is not None and
                len(results.raw_data.points) > 0)

    def analyze(self, results: AnalysisResults, **kwargs) -> AnalysisResults:
        """
        Identify points that belong to the vehicle.

        Args:
            results: Current analysis results
            **kwargs: Additional parameters

        Returns:
            Updated analysis results with vehicle points identified
        """
        if not self.validate_input(results):
            raise PointCloudError("Invalid input: no point data available for vehicle identification")

        print("Identifying vehicle points...")

        # Work with non-ground points to avoid ground clutter
        if results.has_ground_separation:
            points_to_analyze = results.non_ground_points
            print("Using non-ground points for vehicle identification")
        else:
            points_to_analyze = results.raw_data.points
            print("No ground separation performed, using all points")

        # Perform DBSCAN clustering
        cluster_labels = self._dbscan_clustering(points_to_analyze)

        # Find the largest cluster (likely the vehicle)
        largest_cluster_points = self._find_largest_cluster(points_to_analyze, cluster_labels)

        # Apply height filtering
        vehicle_points = self._apply_height_filter(largest_cluster_points, results)

        # Update results
        results.vehicle_points = vehicle_points

        print(f"Identified {len(vehicle_points):,} vehicle points")

        return results

    def _dbscan_clustering(self, points: np.ndarray) -> np.ndarray:
        """
        Perform DBSCAN clustering to identify dense regions.

        Args:
            points: Points to cluster

        Returns:
            Cluster labels for each point
        """
        try:
            from sklearn.cluster import DBSCAN
        except ImportError:
            raise PointCloudError("scikit-learn not available for DBSCAN clustering")

        clustering = DBSCAN(
            eps=self.config.DBSCAN_EPS,
            min_samples=self.config.DBSCAN_MIN_SAMPLES
        )

        cluster_labels = clustering.fit_predict(points)

        # Count clusters
        unique_labels = np.unique(cluster_labels[cluster_labels != -1])
        noise_points = np.sum(cluster_labels == -1)

        print(f"DBSCAN clustering: {len(unique_labels)} clusters found, "
              f"{noise_points:,} noise points")

        return cluster_labels

    def _find_largest_cluster(self, points: np.ndarray, cluster_labels: np.ndarray) -> np.ndarray:
        """
        Find and return points from the largest cluster.

        Args:
            points: Original points
            cluster_labels: Cluster labels from DBSCAN

        Returns:
            Points from the largest cluster
        """
        # Find the largest cluster (excluding noise points with label -1)
        unique_labels, counts = np.unique(cluster_labels[cluster_labels != -1], return_counts=True)

        if len(unique_labels) == 0:
            print("No clusters found, using all input points")
            return points.copy()

        # Get the largest cluster
        largest_cluster_label = unique_labels[np.argmax(counts)]
        largest_cluster_size = np.max(counts)

        print(f"Largest cluster: label {largest_cluster_label} with {largest_cluster_size:,} points")

        vehicle_mask = cluster_labels == largest_cluster_label
        return points[vehicle_mask]

    def _apply_height_filter(self, cluster_points: np.ndarray, results: AnalysisResults) -> np.ndarray:
        """
        Apply height filtering to remove points outside typical vehicle height range.

        Args:
            cluster_points: Points from the largest cluster
            results: Analysis results to get reference heights

        Returns:
            Height-filtered vehicle points
        """
        if len(cluster_points) == 0:
            return cluster_points

        # Determine reference height
        if results.has_ground_separation:
            # Use non-ground points as reference
            z_min = results.non_ground_points[:, 2].min()
        else:
            # Use cluster points as reference
            z_min = cluster_points[:, 2].min()

        z_max = z_min + self.config.VEHICLE_MAX_HEIGHT

        # Apply height filter
        height_mask = ((cluster_points[:, 2] >= z_min) &
                      (cluster_points[:, 2] <= z_max))

        filtered_points = cluster_points[height_mask]

        print(f"Height filtering: {len(filtered_points):,} points within "
              f"vehicle height range ({z_min:.2f}m to {z_max:.2f}m)")

        return filtered_points