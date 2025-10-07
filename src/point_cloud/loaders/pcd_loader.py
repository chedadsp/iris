#!/usr/bin/env python3
"""
PCD File Loader

Loader for Point Cloud Data (PCD) files, supporting ASCII format.

Author: Dimitrije Stojanovic
Date: September 2025
"""

import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any

from .base_loader import BaseFileLoader
from ..models.point_cloud_data import PointCloudData
from ..error_handling import PointCloudError


class PCDLoader(BaseFileLoader):
    """Loader for PCD (Point Cloud Data) files."""

    def __init__(self):
        super().__init__(['.pcd'])

    def _load_file(self, file_path: Path) -> PointCloudData:
        """Load point cloud data from PCD file using visualise module."""
        try:
            # Import the read_points function from visualise module
            from ..visualise_raw_lidar_files import read_points

            # Use the read_points function
            points, intensity, extra_data = read_points(str(file_path))

            # Handle extra data (could be RGB or ring data)
            colors = None
            if extra_data is not None:
                # Check if extra_data looks like RGB (3 columns)
                if hasattr(extra_data, 'shape') and len(extra_data.shape) == 2 and extra_data.shape[1] == 3:
                    colors = extra_data

            # Create metadata
            metadata = {
                'file_format': 'PCD',
                'has_intensity': intensity is not None,
                'has_colors': colors is not None,
                'has_extra_data': extra_data is not None,
            }

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

            return data

        except ImportError as e:
            raise PointCloudError(f"visualise_raw_lidar_files module not available: {e}") from e
        except Exception as e:
            raise PointCloudError(f"Error loading PCD file {file_path}: {e}") from e