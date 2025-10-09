#!/usr/bin/env python3
"""
Standardized Error Handling Module

Provides consistent error handling patterns, custom exceptions, and logging
utilities for the LIDAR point cloud processing application.

Author: Dimitrije Stojanovic
Date: September 2025
Created: Code Review Fixes
"""

import logging
import functools
import traceback
from typing import Any, Callable, Optional, Type, Union
from pathlib import Path
import sys


class LidarProcessingError(Exception):
    """Base exception for LIDAR processing errors."""
    pass


class FileFormatError(LidarProcessingError):
    """Raised when file format is invalid or unsupported."""
    pass


class PointCloudError(LidarProcessingError):
    """Raised for point cloud processing errors."""
    pass


class VTKOperationError(LidarProcessingError):
    """Raised for VTK visualization errors."""
    pass


class DependencyError(LidarProcessingError):
    """Raised when required dependencies are missing."""
    pass


class ValidationError(LidarProcessingError):
    """Raised for data validation errors."""
    pass


class ErrorHandler:
    """Centralized error handling and logging."""
    
    def __init__(self, logger_name: str = "lidar_processing"):
        """
        Initialize error handler.
        
        Args:
            logger_name: Name for the logger instance
        """
        self.logger = logging.getLogger(logger_name)
        self._setup_logging()
    
    def _setup_logging(self):
        """Set up logging configuration if not already configured."""
        if not self.logger.handlers:
            # Create console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(formatter)
            
            # Add handler to logger
            self.logger.addHandler(console_handler)
            self.logger.setLevel(logging.INFO)
    
    def log_error(self, error: Exception, context: str = "", 
                  include_traceback: bool = False) -> None:
        """
        Log an error with context information.
        
        Args:
            error: Exception that occurred
            context: Context where the error occurred
            include_traceback: Whether to include full traceback
        """
        error_type = type(error).__name__
        error_msg = str(error)
        
        if context:
            message = f"Error in {context}: {error_type}: {error_msg}"
        else:
            message = f"{error_type}: {error_msg}"
        
        self.logger.error(message)
        
        if include_traceback:
            self.logger.error("Traceback:", exc_info=True)
    
    def log_warning(self, message: str, context: str = "") -> None:
        """Log a warning message."""
        if context:
            full_message = f"Warning in {context}: {message}"
        else:
            full_message = f"Warning: {message}"
        
        self.logger.warning(full_message)
    
    def log_info(self, message: str, context: str = "") -> None:
        """Log an info message."""
        if context:
            full_message = f"{context}: {message}"
        else:
            full_message = message
        
        self.logger.info(full_message)
    
    def handle_exception(self, error: Exception, context: str = "",
                        reraise: bool = True, fallback_result: Any = None) -> Any:
        """
        Handle an exception with consistent logging and optional re-raising.
        
        Args:
            error: Exception that occurred
            context: Context where the error occurred
            reraise: Whether to re-raise the exception
            fallback_result: Value to return if not re-raising
            
        Returns:
            fallback_result if not re-raising, otherwise raises the exception
        """
        self.log_error(error, context, include_traceback=not reraise)
        
        if reraise:
            raise error
        else:
            return fallback_result


def with_error_handling(
    context: str = "",
    reraise: bool = True,
    fallback_result: Any = None,
    expected_exceptions: tuple = (Exception,),
    logger_name: str = "lidar_processing"
):
    """
    Decorator for consistent error handling across functions.
    
    Args:
        context: Context description for error messages
        reraise: Whether to re-raise caught exceptions
        fallback_result: Value to return if not re-raising
        expected_exceptions: Tuple of exception types to catch
        logger_name: Name for the error handler logger
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            error_handler = ErrorHandler(logger_name)
            
            try:
                return func(*args, **kwargs)
            except expected_exceptions as e:
                func_context = context or f"{func.__module__}.{func.__name__}"
                return error_handler.handle_exception(
                    e, func_context, reraise, fallback_result
                )
        
        return wrapper
    return decorator


def validate_file_path(file_path: Union[str, Path], 
                      must_exist: bool = True,
                      allowed_extensions: Optional[list] = None) -> Path:
    """
    Validate file path with security checks.
    
    Args:
        file_path: Path to validate
        must_exist: Whether the file must exist
        allowed_extensions: List of allowed file extensions
        
    Returns:
        Validated Path object
        
    Raises:
        ValidationError: If validation fails
    """
    if not file_path:
        raise ValidationError("File path cannot be empty")
    
    # Convert to Path object
    path = Path(file_path)
    
    # Security check: prevent path traversal
    try:
        # Resolve the path and check if it's within allowed bounds
        resolved_path = path.resolve()
        
        # Check for path traversal attempts
        if '..' in str(path) and not path.is_absolute():
            raise ValidationError(f"Path traversal detected in: {file_path}")
            
    except (OSError, RuntimeError) as e:
        raise ValidationError(f"Invalid file path: {file_path} - {e}")
    
    # Check if file exists when required
    if must_exist and not resolved_path.exists():
        raise ValidationError(f"File does not exist: {resolved_path}")
    
    # Check file extension
    if allowed_extensions:
        if resolved_path.suffix.lower() not in [ext.lower() for ext in allowed_extensions]:
            raise ValidationError(
                f"File extension {resolved_path.suffix} not allowed. "
                f"Allowed extensions: {allowed_extensions}"
            )
    
    return resolved_path


def validate_point_cloud_data(points: Any, min_points: int = 1) -> None:
    """
    Validate point cloud data.
    
    Args:
        points: Point cloud data to validate
        min_points: Minimum number of points required
        
    Raises:
        ValidationError: If validation fails
    """
    import numpy as np
    
    if points is None:
        raise ValidationError("Point cloud data cannot be None")
    
    if not isinstance(points, np.ndarray):
        raise ValidationError("Point cloud data must be numpy array")
    
    if len(points.shape) != 2:
        raise ValidationError(f"Point cloud must be 2D array, got shape {points.shape}")
    
    if points.shape[1] != 3:
        raise ValidationError(f"Point cloud must have 3 columns (x,y,z), got {points.shape[1]}")
    
    if len(points) < min_points:
        raise ValidationError(f"Point cloud must have at least {min_points} points, got {len(points)}")
    
    if not np.isfinite(points).all():
        raise ValidationError("Point cloud contains non-finite values (NaN or inf)")


class OperationCallback:
    """Callback handler for operations with error handling."""
    
    def __init__(self, callback: Optional[Callable] = None, 
                 error_callback: Optional[Callable] = None):
        """
        Initialize callback handler.
        
        Args:
            callback: Success callback function
            error_callback: Error callback function
        """
        self.callback = callback
        self.error_callback = error_callback
        self.error_handler = ErrorHandler()
    
    def on_success(self, result: Any) -> None:
        """Handle successful operation."""
        try:
            if self.callback:
                self.callback(result)
        except Exception as e:
            self.error_handler.log_error(e, "success callback")
    
    def on_error(self, error: Exception, context: str = "") -> None:
        """Handle error in operation."""
        self.error_handler.log_error(error, context)
        
        try:
            if self.error_callback:
                self.error_callback(error)
        except Exception as callback_error:
            self.error_handler.log_error(callback_error, "error callback")


def safe_operation(operation: Callable, operation_name: str = "operation",
                  max_retries: int = 1, **kwargs) -> Any:
    """
    Execute operation with error handling and optional retries.
    
    Args:
        operation: Function to execute
        operation_name: Name for logging
        max_retries: Maximum number of retry attempts
        **kwargs: Arguments to pass to operation
        
    Returns:
        Result of the operation or None if failed
    """
    error_handler = ErrorHandler()
    last_error = None
    
    for attempt in range(max_retries):
        try:
            result = operation(**kwargs)
            if attempt > 0:
                error_handler.log_info(f"{operation_name} succeeded on retry {attempt + 1}")
            return result
            
        except Exception as e:
            last_error = e
            error_handler.log_error(e, f"{operation_name} attempt {attempt + 1}")
            
            if attempt < max_retries - 1:
                error_handler.log_info(f"Retrying {operation_name}...")
    
    error_handler.log_error(last_error, f"{operation_name} failed after {max_retries} attempts", 
                          include_traceback=True)
    return None


# Global error handler instance
global_error_handler = ErrorHandler("lidar_global")


if __name__ == "__main__":
    # Test the error handling system
    print("Error Handling Module Test")
    
    # Test validation
    try:
        validate_file_path("nonexistent.txt")
    except ValidationError as e:
        print(f"Validation test passed: {e}")
    
    # Test error handling decorator
    @with_error_handling("test_function", reraise=False, fallback_result="fallback")
    def test_function():
        raise ValueError("Test error")
    
    result = test_function()
    print(f"Decorator test result: {result}")
    
    print("Error handling module tests completed.")