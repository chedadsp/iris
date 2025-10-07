#!/usr/bin/env python3
"""
Base Launcher Interface

Provides the abstract base class for all launcher modes, defining the
common interface and shared functionality.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import sys
import gc


class BaseLauncher(ABC):
    """Abstract base class for all launcher modes."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base launcher.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.mode = self.__class__.__name__.replace('Launcher', '').lower()

    @abstractmethod
    def launch(self, **kwargs) -> int:
        """
        Launch the application in the specific mode.

        Args:
            **kwargs: Mode-specific parameters

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        pass

    @abstractmethod
    def validate_requirements(self) -> Tuple[bool, str]:
        """
        Validate that all requirements for this mode are met.

        Returns:
            Tuple of (success: bool, error_message: str)
        """
        pass

    def get_mode_name(self) -> str:
        """Get the name of this launcher mode."""
        return self.mode

    def cleanup(self) -> None:
        """
        Clean up resources after launch completes.
        This method should be called regardless of success/failure.
        """
        # Force garbage collection to help with VTK cleanup
        gc.collect()
        gc.collect()

    def print_mode_info(self) -> None:
        """Print information about this launcher mode."""
        print(f"🚀 Starting {self.get_mode_name()} mode...")

    def handle_error(self, error: Exception, context: str = "") -> int:
        """
        Handle errors in a consistent way across launcher modes.

        Args:
            error: The exception that occurred
            context: Additional context about where the error occurred

        Returns:
            Exit code for the error
        """
        error_msg = f"❌ {self.get_mode_name().title()} mode failed"
        if context:
            error_msg += f" during {context}"
        error_msg += f": {error}"

        print(error_msg)
        return 1


class LauncherError(Exception):
    """Base exception for launcher-related errors."""
    pass


class LauncherValidationError(LauncherError):
    """Exception raised when launcher validation fails."""
    pass


class LauncherExecutionError(LauncherError):
    """Exception raised when launcher execution fails."""
    pass