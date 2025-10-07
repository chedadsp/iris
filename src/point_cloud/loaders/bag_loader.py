#!/usr/bin/env python3
"""
ROS Bag File Loader

Loader for ROS bag files containing point cloud data.
This is experimental functionality that may fail on uncommon PointCloud2 formats.

Author: Dimitrije Stojanovic
Date: September 2025
"""

import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any

from .base_loader import BaseFileLoader
from ..models.point_cloud_data import PointCloudData
from ..error_handling import PointCloudError


class BagLoader(BaseFileLoader):
    """Loader for ROS bag files."""

    def __init__(self):
        super().__init__(['.bag'])

    def _load_file(self, file_path: Path) -> PointCloudData:
        """Load point cloud data from ROS bag file using visualise module."""
        try:
            # Import the read_points function from visualise module
            from ..visualise_raw_lidar_files import read_points

            # Use the read_points function
            points, intensity, extra_data = read_points(str(file_path))

            # Handle extra data (could be ring data or other sensor information)
            colors = None
            ring_data = None

            if extra_data is not None:
                # Check if extra_data looks like RGB (3 columns)
                if hasattr(extra_data, 'shape') and len(extra_data.shape) == 2:
                    if extra_data.shape[1] == 3:
                        colors = extra_data
                    elif extra_data.shape[1] == 1:
                        ring_data = extra_data

            # Create metadata
            metadata = {
                'file_format': 'ROS_BAG',
                'has_intensity': intensity is not None,
                'has_colors': colors is not None,
                'has_ring_data': ring_data is not None,
                'experimental': True,  # Mark as experimental
            }

            # Store ring data in metadata if available
            if ring_data is not None:
                metadata['ring_data'] = ring_data

            # Create point cloud data
            data = PointCloudData(
                points=points,
                colors=colors,
                intensity=intensity,
                metadata=metadata
            )

            # Print loading information
            self._print_loading_info(file_path, data.size)
            self._print_bounds_info(data)
            print("Note: ROS bag support is experimental and may fail on uncommon PointCloud2 formats")

            return data

        except ImportError as e:
            raise PointCloudError(f"visualise_raw_lidar_files module or ROS bag dependencies not available: {e}") from e
        except Exception as e:
            raise PointCloudError(f"Error loading ROS bag file {file_path}: {e}") from e