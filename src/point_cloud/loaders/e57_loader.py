#!/usr/bin/env python3
"""
E57 File Loader

Specialized loader for E57 LIDAR files with support for multiple scans,
color data, and intensity information.

Author: Dimitrije Stojanovic
Date: September 2025
"""

import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
from pye57 import E57

from .base_loader import BaseFileLoader
from ..models.point_cloud_data import PointCloudData
from ..error_handling import PointCloudError


class E57Loader(BaseFileLoader):
    """Loader for E57 LIDAR files."""

    def __init__(self):
        super().__init__(['.e57'])

    def _load_file(self, file_path: Path) -> PointCloudData:
        """Load point cloud data from E57 file."""
        try:
            # Open E57 file
            e57 = E57(str(file_path))

            # Get scan data (default to first scan)
            scan_data = e57.read_scan(0, ignore_missing_fields=True)

            # Extract coordinates
            points = self._extract_coordinates(scan_data)

            # Extract optional data
            colors = self._extract_colors(scan_data)
            intensity = self._extract_intensity(scan_data)

            # Create metadata
            metadata = self._create_metadata(e57, scan_data)

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

        except Exception as e:
            raise PointCloudError(f"Error loading E57 file {file_path}: {e}") from e

    def _extract_coordinates(self, scan_data: Dict[str, np.ndarray]) -> np.ndarray:
        """Extract XYZ coordinates from scan data."""
        required_fields = ['cartesianX', 'cartesianY', 'cartesianZ']

        if not all(field in scan_data for field in required_fields):
            missing = [field for field in required_fields if field not in scan_data]
            raise ValueError(f"Required coordinate fields missing: {missing}")

        return np.column_stack([
            scan_data['cartesianX'],
            scan_data['cartesianY'],
            scan_data['cartesianZ']
        ])

    def _extract_colors(self, scan_data: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        """Extract RGB color data if available."""
        color_fields = ['colorRed', 'colorGreen', 'colorBlue']

        if all(field in scan_data for field in color_fields):
            return np.column_stack([
                scan_data['colorRed'],
                scan_data['colorGreen'],
                scan_data['colorBlue']
            ])

        return None

    def _extract_intensity(self, scan_data: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        """Extract intensity data if available."""
        if 'intensity' in scan_data:
            return scan_data['intensity']
        return None

    def _create_metadata(self, e57: E57, scan_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Create metadata dictionary with file information."""
        metadata = {
            'file_format': 'E57',
            'scan_index': 0,  # Currently using first scan only
            'available_fields': list(scan_data.keys()),
            'has_colors': 'colorRed' in scan_data,
            'has_intensity': 'intensity' in scan_data,
        }

        # Add scan count if available
        try:
            metadata['total_scans'] = e57.scan_count
        except:
            metadata['total_scans'] = 1

        return metadata

    def load_specific_scan(self, file_path: Path, scan_index: int = 0) -> PointCloudData:
        """Load a specific scan from E57 file."""
        try:
            validated_path = self.load(file_path)  # This will validate but we need custom logic

            e57 = E57(str(file_path))

            if scan_index >= e57.scan_count:
                raise ValueError(f"Scan index {scan_index} out of range. File has {e57.scan_count} scans.")

            scan_data = e57.read_scan(scan_index, ignore_missing_fields=True)

            # Extract data using same methods
            points = self._extract_coordinates(scan_data)
            colors = self._extract_colors(scan_data)
            intensity = self._extract_intensity(scan_data)

            metadata = self._create_metadata(e57, scan_data)
            metadata['scan_index'] = scan_index

            data = PointCloudData(
                points=points,
                colors=colors,
                intensity=intensity,
                metadata=metadata
            )

            print(f"Loaded scan {scan_index} with {data.size:,} points from {file_path.name}")
            self._print_bounds_info(data)

            return data

        except Exception as e:
            raise PointCloudError(f"Error loading E57 scan {scan_index} from {file_path}: {e}") from e