#!/usr/bin/env python3
"""
Abstract Interfaces for Point Cloud Processing Components

Defines the contracts that all processing components must implement.
These interfaces enable dependency injection and make testing easier.

Author: Dimitrije Stojanovic
Date: September 2025
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

from ..models.point_cloud_data import PointCloudData, AnalysisResults


class FileLoader(ABC):
    """Abstract base class for file loaders."""

    @abstractmethod
    def can_load(self, file_path: Path) -> bool:
        """Check if this loader can handle the given file type."""
        pass

    @abstractmethod
    def load(self, file_path: Path) -> PointCloudData:
        """Load point cloud data from file."""
        pass


class PointCloudProcessor(ABC):
    """Abstract base class for point cloud preprocessing operations."""

    @abstractmethod
    def process(self, data: PointCloudData, **kwargs) -> PointCloudData:
        """Process point cloud data and return modified data."""
        pass


class AnalysisStep(ABC):
    """Abstract base class for analysis pipeline steps."""

    @abstractmethod
    def analyze(self, results: AnalysisResults, **kwargs) -> AnalysisResults:
        """Perform analysis step and update results."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this analysis step for logging."""
        pass

    def validate_input(self, results: AnalysisResults) -> bool:
        """Validate that input results are suitable for this step."""
        return results.raw_data is not None

    def on_step_start(self, results: AnalysisResults) -> None:
        """Called before analysis step starts."""
        pass

    def on_step_complete(self, results: AnalysisResults) -> None:
        """Called after analysis step completes."""
        pass


class Visualizer(ABC):
    """Abstract base class for visualization components."""

    @abstractmethod
    def visualize(self, results: AnalysisResults, mode: str = "default", **kwargs) -> None:
        """Create visualization from analysis results."""
        pass

    @abstractmethod
    def supported_modes(self) -> list:
        """List of supported visualization modes."""
        pass


class ResultExporter(ABC):
    """Abstract base class for result exporters."""

    @abstractmethod
    def export(self, results: AnalysisResults, output_dir: Path, **kwargs) -> Dict[str, Path]:
        """Export analysis results to files."""
        pass

    @abstractmethod
    def supported_formats(self) -> list:
        """List of supported export formats."""
        pass