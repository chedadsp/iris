#!/usr/bin/env python3
"""
Base File Loader

Abstract base class for all file loaders with common functionality.

Author: Dimitrije Stojanovic
Date: September 2025
"""

from pathlib import Path
from typing import List

from ..interfaces.processors import FileLoader
from ..models.point_cloud_data import PointCloudData
from ..error_handling import validate_file_path


class BaseFileLoader(FileLoader):
    """Base class for file loaders with common validation logic."""

    def __init__(self, supported_extensions: List[str]):
        """Initialize with list of supported file extensions."""
        self.supported_extensions = [ext.lower() for ext in supported_extensions]

    def can_load(self, file_path: Path) -> bool:
        """Check if this loader can handle the given file type."""
        return file_path.suffix.lower() in self.supported_extensions

    def load(self, file_path: Path) -> PointCloudData:
        """Load point cloud data from file with validation."""
        # Validate file path
        validated_path = validate_file_path(
            str(file_path),
            must_exist=True,
            allowed_extensions=self.supported_extensions
        )

        # Delegate to specific loader implementation
        return self._load_file(validated_path)

    def _load_file(self, file_path: Path) -> PointCloudData:
        """Implementation-specific loading logic."""
        raise NotImplementedError("Subclasses must implement _load_file method")

    def _print_loading_info(self, file_path: Path, points_count: int) -> None:
        """Print standardized loading information."""
        print(f"Loaded {points_count:,} points from {file_path.suffix.upper()} file: {file_path.name}")

    def _print_bounds_info(self, data: PointCloudData) -> None:
        """Print point cloud bounds information."""
        (x_min, x_max), (y_min, y_max), (z_min, z_max) = data.bounds
        print(f"Point cloud bounds: "
              f"X({x_min:.2f}, {x_max:.2f}), "
              f"Y({y_min:.2f}, {y_max:.2f}), "
              f"Z({z_min:.2f}, {z_max:.2f})")