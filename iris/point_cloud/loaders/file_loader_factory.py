#!/usr/bin/env python3
"""
File Loader Factory

Factory pattern implementation for creating appropriate file loaders
based on file extension.

Author: Dimitrije Stojanovic
Date: September 2025
"""

from pathlib import Path
from typing import List, Optional

from ..interfaces.processors import FileLoader
from ..models.point_cloud_data import PointCloudData
from ..error_handling import PointCloudError

from .e57_loader import E57Loader
from .pcd_loader import PCDLoader
from .bag_loader import BagLoader


class FileLoaderFactory:
    """Factory for creating appropriate file loaders."""

    def __init__(self):
        # Register available loaders
        self._loaders: List[FileLoader] = [
            E57Loader(),
            PCDLoader(),
            BagLoader(),
        ]

    def create_loader(self, file_path: Path) -> FileLoader:
        """Create appropriate loader for the given file type."""
        for loader in self._loaders:
            if loader.can_load(file_path):
                return loader

        # No suitable loader found
        supported_extensions = []
        for loader in self._loaders:
            if hasattr(loader, 'supported_extensions'):
                supported_extensions.extend(loader.supported_extensions)

        raise PointCloudError(
            f"Unsupported file format: {file_path.suffix}. "
            f"Supported formats are: {', '.join(supported_extensions)}"
        )

    def load_file(self, file_path: Path) -> PointCloudData:
        """Load file using appropriate loader."""
        loader = self.create_loader(file_path)
        return loader.load(file_path)

    def get_supported_extensions(self) -> List[str]:
        """Get list of all supported file extensions."""
        extensions = []
        for loader in self._loaders:
            if hasattr(loader, 'supported_extensions'):
                extensions.extend(loader.supported_extensions)
        return list(set(extensions))  # Remove duplicates

    def register_loader(self, loader: FileLoader) -> None:
        """Register a new file loader."""
        if loader not in self._loaders:
            self._loaders.append(loader)

    def unregister_loader(self, loader_type: type) -> None:
        """Unregister a loader by type."""
        self._loaders = [loader for loader in self._loaders if not isinstance(loader, loader_type)]


# Global factory instance
_global_factory = FileLoaderFactory()


def load_point_cloud_file(file_path: str) -> PointCloudData:
    """Convenience function to load point cloud file using global factory."""
    return _global_factory.load_file(Path(file_path))


def get_supported_formats() -> List[str]:
    """Get list of supported file formats."""
    return _global_factory.get_supported_extensions()