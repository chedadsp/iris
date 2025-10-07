#!/usr/bin/env python3
"""
Base Service Class

Provides common interface and functionality for all business logic services.
This establishes the service layer pattern for separating business logic from GUI.

Author: Dimitrije Stojanovic
Date: December 2025
Created: Phase 3 Refactoring - Service Layer Architecture
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import logging
from pathlib import Path

from ..error_handling import with_error_handling


class ServiceResult:
    """
    Base class for service operation results.

    Provides consistent structure for all service return values.
    """

    def __init__(self, success: bool = True, data: Any = None,
                 error: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize service result.

        Args:
            success: Whether the operation was successful
            data: The result data (if successful)
            error: Error message (if unsuccessful)
            metadata: Additional information about the operation
        """
        self.success = success
        self.data = data
        self.error = error
        self.metadata = metadata or {}

    @property
    def is_success(self) -> bool:
        """Check if operation was successful."""
        return self.success and self.error is None

    @property
    def is_error(self) -> bool:
        """Check if operation resulted in error."""
        return not self.success or self.error is not None

    def __bool__(self) -> bool:
        """Allow boolean evaluation of result."""
        return self.is_success


class BaseService(ABC):
    """
    Abstract base class for all business logic services.

    Provides common functionality and establishes service layer patterns:
    - Error handling and logging
    - Configuration management
    - Result formatting
    - Resource management
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize base service.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Setup service-specific logging."""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value with fallback to default.

        Args:
            key: Configuration key (supports dot notation for nested keys)
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        if '.' in key:
            keys = key.split('.')
            value = self.config
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            return value
        else:
            return self.config.get(key, default)

    def create_result(self, success: bool = True, data: Any = None,
                     error: Optional[str] = None, **metadata) -> ServiceResult:
        """
        Create standardized service result.

        Args:
            success: Whether operation was successful
            data: Result data
            error: Error message if any
            **metadata: Additional metadata

        Returns:
            ServiceResult instance
        """
        return ServiceResult(
            success=success,
            data=data,
            error=error,
            metadata=metadata
        )

    def create_success_result(self, data: Any = None, **metadata) -> ServiceResult:
        """Create successful result."""
        return self.create_result(success=True, data=data, **metadata)

    def create_error_result(self, error: str, **metadata) -> ServiceResult:
        """Create error result."""
        return self.create_result(success=False, error=error, **metadata)

    @with_error_handling("service_operation")
    def safe_execute(self, operation_name: str, operation_func, *args, **kwargs) -> ServiceResult:
        """
        Safely execute an operation with error handling.

        Args:
            operation_name: Name of the operation for logging
            operation_func: Function to execute
            *args: Arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            ServiceResult with operation outcome
        """
        try:
            self.logger.info(f"Starting {operation_name}")
            result = operation_func(*args, **kwargs)
            self.logger.info(f"Completed {operation_name}")

            if isinstance(result, ServiceResult):
                return result
            else:
                return self.create_success_result(data=result, operation=operation_name)

        except Exception as e:
            error_msg = f"Error in {operation_name}: {str(e)}"
            self.logger.error(error_msg)
            return self.create_error_result(error_msg, operation=operation_name)

    def validate_required_config(self, required_keys: list) -> ServiceResult:
        """
        Validate that required configuration keys are present.

        Args:
            required_keys: List of required configuration keys

        Returns:
            ServiceResult indicating validation success/failure
        """
        missing_keys = []

        for key in required_keys:
            if self.get_config_value(key) is None:
                missing_keys.append(key)

        if missing_keys:
            error_msg = f"Missing required configuration keys: {', '.join(missing_keys)}"
            return self.create_error_result(error_msg)

        return self.create_success_result()

    def save_result_to_file(self, result: Any, output_path: Path,
                           metadata: Optional[Dict[str, Any]] = None) -> ServiceResult:
        """
        Save service result to file (to be implemented by subclasses).

        Args:
            result: Result data to save
            output_path: Path to save the result
            metadata: Optional metadata to include

        Returns:
            ServiceResult indicating save success/failure
        """
        # Base implementation - subclasses should override for specific formats
        try:
            # Basic validation
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Log the save operation
            self.logger.info(f"Saving result to {output_path}")

            return self.create_success_result(
                data=str(output_path),
                saved_path=str(output_path),
                metadata_included=metadata is not None
            )

        except Exception as e:
            error_msg = f"Error saving result: {str(e)}"
            self.logger.error(error_msg)
            return self.create_error_result(error_msg)

    @abstractmethod
    def get_service_info(self) -> Dict[str, Any]:
        """
        Get service information and capabilities.

        Returns:
            Dictionary with service information
        """
        pass

    def __str__(self) -> str:
        """String representation of the service."""
        return f"{self.__class__.__name__}(config_keys={list(self.config.keys())})"

    def __repr__(self) -> str:
        """Detailed representation of the service."""
        return f"{self.__class__.__name__}(config={self.config})"