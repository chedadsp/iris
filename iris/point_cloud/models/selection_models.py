#!/usr/bin/env python3
"""
Selection Models - Data models for point cloud selections and interactive operations.

Contains data structures and models for managing selections, transformations,
and interactive operations without GUI dependencies.

Author: Dimitrije Stojanovic
Date: September 2025
Refactored: November 2025 (GUI/Business Logic Separation)
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class SelectionState(Enum):
    """State of an interactive selection operation."""
    INACTIVE = "inactive"
    ACTIVE = "active"
    UPDATING = "updating"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class SelectionInfo:
    """Information about a point cloud selection."""
    total_points: int
    selected_points: int
    selection_ratio: float
    bounds: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def selection_percentage(self) -> float:
        """Get selection ratio as percentage."""
        return self.selection_ratio * 100.0


@dataclass
class InteractiveState:
    """State management for interactive operations."""
    selection_state: SelectionState = SelectionState.INACTIVE
    is_updating: bool = False
    last_update_count: Optional[int] = None
    error_message: Optional[str] = None

    def set_error(self, message: str):
        """Set error state with message."""
        self.selection_state = SelectionState.ERROR
        self.error_message = message

    def clear_error(self):
        """Clear error state."""
        if self.selection_state == SelectionState.ERROR:
            self.selection_state = SelectionState.INACTIVE
        self.error_message = None

    def start_update(self):
        """Mark update as starting."""
        self.is_updating = True
        self.selection_state = SelectionState.UPDATING

    def end_update(self, point_count: int):
        """Mark update as complete."""
        self.is_updating = False
        self.last_update_count = point_count
        self.selection_state = SelectionState.ACTIVE


@dataclass
class HumanPositionState:
    """State for human model positioning operations."""
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    scale: float = 1.0
    rotation_deg: float = 0.0
    model_type: str = "simple_seated"
    interactive_state: InteractiveState = field(default_factory=InteractiveState)

    def update_transform(self, center: Optional[Tuple[float, float, float]] = None,
                        scale: Optional[float] = None,
                        rotation_deg: Optional[float] = None):
        """Update transformation parameters."""
        if center is not None:
            self.center = center
        if scale is not None:
            self.scale = max(0.1, min(3.0, scale))  # Clamp scale
        if rotation_deg is not None:
            self.rotation_deg = rotation_deg % 360  # Normalize rotation

    @property
    def transform_dict(self) -> Dict[str, Any]:
        """Get transformation as dictionary."""
        return {
            'center': self.center,
            'scale': self.scale,
            'rotation_deg': self.rotation_deg
        }


@dataclass
class CubeSelectionState:
    """State for cube selection operations."""
    cube_bounds: Optional[List[float]] = None
    selected_count: int = 0
    total_points: int = 0
    interactive_state: InteractiveState = field(default_factory=InteractiveState)

    @property
    def selection_info(self) -> SelectionInfo:
        """Get selection information."""
        ratio = self.selected_count / self.total_points if self.total_points > 0 else 0.0

        bounds_dict = None
        if self.cube_bounds and len(self.cube_bounds) == 6:
            x_min, x_max, y_min, y_max, z_min, z_max = self.cube_bounds
            bounds_dict = {
                'x_min': x_min, 'x_max': x_max,
                'y_min': y_min, 'y_max': y_max,
                'z_min': z_min, 'z_max': z_max
            }

        return SelectionInfo(
            total_points=self.total_points,
            selected_points=self.selected_count,
            selection_ratio=ratio,
            bounds=bounds_dict
        )

    def update_selection(self, cube_bounds: List[float], selected_count: int):
        """Update selection parameters."""
        self.cube_bounds = cube_bounds.copy() if cube_bounds else None
        self.selected_count = selected_count

    def should_update_visualization(self, threshold_ratio: float = 0.001) -> bool:
        """Determine if visualization should be updated based on change magnitude."""
        if self.interactive_state.last_update_count is None:
            return True

        if self.total_points == 0:
            return False

        change_threshold = self.total_points * threshold_ratio
        change = abs(self.selected_count - self.interactive_state.last_update_count)
        return change > change_threshold


@dataclass
class VisualizationState:
    """State for visualization components."""
    window_size: Tuple[int, int] = (1600, 1200)
    background_color: str = "black"
    point_size: int = 3
    opacity: float = 0.8
    show_axes: bool = True
    show_grid: bool = False
    camera_position: str = "iso"

    def update_settings(self, **kwargs):
        """Update visualization settings."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


@dataclass
class FileOperationResult:
    """Result of file operations."""
    success: bool
    file_path: Optional[Path] = None
    message: str = ""
    points_saved: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success_result(cls, file_path: Path, points_saved: int, message: str = "") -> 'FileOperationResult':
        """Create a successful result."""
        return cls(
            success=True,
            file_path=file_path,
            message=message,
            points_saved=points_saved
        )

    @classmethod
    def error_result(cls, message: str) -> 'FileOperationResult':
        """Create an error result."""
        return cls(success=False, message=message)


class SelectionManager:
    """Manager for selection operations and state."""

    def __init__(self, output_dir: str = "output"):
        """Initialize selection manager."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def save_selection(self, points: np.ndarray, filename: str,
                      metadata: Optional[Dict[str, Any]] = None) -> FileOperationResult:
        """
        Save selection to file.

        Args:
            points: Points to save
            filename: Base filename (without extension)
            metadata: Optional metadata to save alongside

        Returns:
            FileOperationResult with operation status
        """
        try:
            # Save points
            points_file = self.output_dir / f"{filename}.npy"
            np.save(points_file, points)

            # Save metadata if provided
            if metadata:
                import json
                metadata_file = self.output_dir / f"{filename}_metadata.json"
                with open(metadata_file, 'w') as f:
                    # Convert numpy types to native Python types for JSON serialization
                    json_metadata = self._convert_for_json(metadata)
                    json.dump(json_metadata, f, indent=2)

            return FileOperationResult.success_result(
                points_file, len(points),
                f"Saved {len(points):,} points to {points_file}"
            )

        except Exception as e:
            return FileOperationResult.error_result(f"Failed to save selection: {e}")

    def load_selection(self, filename: str) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
        """
        Load selection from file.

        Args:
            filename: Base filename (without extension)

        Returns:
            Tuple of (points array, metadata dict) or (None, None) if not found
        """
        try:
            # Load points
            points_file = self.output_dir / f"{filename}.npy"
            if not points_file.exists():
                return None, None

            points = np.load(points_file)

            # Load metadata if available
            metadata = None
            metadata_file = self.output_dir / f"{filename}_metadata.json"
            if metadata_file.exists():
                import json
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)

            return points, metadata

        except Exception as e:
            print(f"Error loading selection {filename}: {e}")
            return None, None

    def list_saved_selections(self) -> List[str]:
        """
        List all saved selection files.

        Returns:
            List of base filenames (without .npy extension)
        """
        if not self.output_dir.exists():
            return []

        npy_files = self.output_dir.glob("*.npy")
        return [f.stem for f in npy_files if not f.stem.endswith('_metadata')]

    def _convert_for_json(self, obj: Any) -> Any:
        """Convert numpy types to JSON-serializable types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_for_json(item) for item in obj]
        else:
            return obj


class StateValidator:
    """Validator for state objects."""

    @staticmethod
    def validate_cube_bounds(bounds: List[float]) -> bool:
        """Validate cube bounds format and values."""
        if not bounds or len(bounds) != 6:
            return False

        if any(np.isnan(bounds)) or any(np.isinf(bounds)):
            return False

        x_min, x_max, y_min, y_max, z_min, z_max = bounds
        return x_min < x_max and y_min < y_max and z_min < z_max

    @staticmethod
    def validate_human_transform(center: Tuple[float, float, float],
                                scale: float, rotation_deg: float) -> bool:
        """Validate human transformation parameters."""
        # Validate center
        if len(center) != 3 or any(np.isnan(center)) or any(np.isinf(center)):
            return False

        # Validate scale
        if np.isnan(scale) or np.isinf(scale) or scale <= 0:
            return False

        # Validate rotation
        if np.isnan(rotation_deg) or np.isinf(rotation_deg):
            return False

        return True

    @staticmethod
    def validate_window_size(window_size: Tuple[int, int]) -> bool:
        """Validate window size parameters."""
        if len(window_size) != 2:
            return False

        width, height = window_size
        return isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0