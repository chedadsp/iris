#!/usr/bin/env python3
"""
Base Controller Class

Abstract base class for all controllers providing common functionality
for mediating between GUI widgets and business logic services.

Author: Dimitrije Stojanovic
Date: December 2025
Created: Phase 3 Refactoring - GUI/Business Logic Separation
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
import logging

from ..services import BaseService, ServiceResult, get_service


class BaseController(ABC):
    """
    Abstract base class for all controllers.

    Controllers mediate between GUI widgets and business logic services:
    - Handle user interactions from widgets
    - Coordinate service calls and data flow
    - Manage application state
    - Provide error handling and validation
    - Maintain separation between UI and business logic
    """

    def __init__(self, service_names: Optional[List[str]] = None,
                 controller_config: Optional[Dict[str, Any]] = None):
        """
        Initialize base controller.

        Args:
            service_names: List of service names this controller depends on
            controller_config: Optional controller configuration
        """
        self.service_names = service_names or []
        self.config = controller_config or {}
        self.services: Dict[str, BaseService] = {}
        self.logger = logging.getLogger(f"{self.__class__.__name__}")

        # Controller state
        self._is_initialized = False

        # Setup logging and initialize services
        self._setup_logging()
        self._initialize_services()

        self._is_initialized = True

    def _setup_logging(self) -> None:
        """Setup controller-specific logging."""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def _initialize_services(self) -> None:
        """Initialize required services."""
        for service_name in self.service_names:
            try:
                service = get_service(service_name)
                if service:
                    self.services[service_name] = service
                    self.logger.info(f"Initialized service: {service_name}")
                else:
                    self.logger.error(f"Failed to get service: {service_name}")
            except Exception as e:
                self.logger.error(f"Error initializing service {service_name}: {str(e)}")

    def get_service(self, service_name: str) -> Optional[BaseService]:
        """
        Get a service instance.

        Args:
            service_name: Name of the service

        Returns:
            Service instance or None if not found
        """
        return self.services.get(service_name)

    def safe_call_service(self, service_name: str, method_name: str, *args, **kwargs) -> Any:
        """
        Safely call a service method with error handling.

        Args:
            service_name: Name of the service
            method_name: Name of the service method
            *args: Arguments to pass to the method
            **kwargs: Keyword arguments to pass to the method

        Returns:
            Method result or None if failed
        """
        service = self.get_service(service_name)
        if not service:
            self.logger.error(f"Service not available: {service_name}")
            return None

        if not hasattr(service, method_name):
            self.logger.error(f"Service method not found: {service_name}.{method_name}")
            return None

        try:
            method = getattr(service, method_name)
            result = method(*args, **kwargs)
            self.logger.debug(f"Service method {service_name}.{method_name} called successfully")
            return result
        except Exception as e:
            self.logger.error(f"Error calling {service_name}.{method_name}: {str(e)}")
            return None

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """
        Get controller configuration value.

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

    def set_config_value(self, key: str, value: Any) -> None:
        """
        Set controller configuration value.

        Args:
            key: Configuration key
            value: Configuration value
        """
        self.config[key] = value
        self.logger.debug(f"Config updated: {key} = {value}")

    def validate_services(self, required_services: Optional[List[str]] = None) -> bool:
        """
        Validate that required services are available.

        Args:
            required_services: List of required service names (defaults to all configured services)

        Returns:
            True if all required services are available, False otherwise
        """
        services_to_check = required_services or self.service_names

        for service_name in services_to_check:
            if service_name not in self.services:
                self.logger.error(f"Required service not available: {service_name}")
                return False

        return True

    def create_service_result(self, success: bool = True, data: Any = None,
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

    def handle_service_error(self, error: Exception, operation: str) -> ServiceResult:
        """
        Handle service errors consistently.

        Args:
            error: Exception that occurred
            operation: Name of the operation that failed

        Returns:
            ServiceResult with error information
        """
        error_msg = f"Error in {operation}: {str(error)}"
        self.logger.error(error_msg)
        return self.create_service_result(
            success=False,
            error=error_msg,
            operation=operation,
            exception_type=type(error).__name__
        )

    def is_initialized(self) -> bool:
        """Check if controller is initialized."""
        return self._is_initialized

    def get_controller_info(self) -> Dict[str, Any]:
        """
        Get information about the controller.

        Returns:
            Dictionary with controller information
        """
        return {
            'controller_name': self.__class__.__name__,
            'is_initialized': self._is_initialized,
            'configured_services': self.service_names,
            'available_services': list(self.services.keys()),
            'config_keys': list(self.config.keys()),
            'services_status': {
                name: service is not None
                for name, service in self.services.items()
            }
        }

    def cleanup(self) -> None:
        """Clean up controller resources."""
        try:
            self.logger.info(f"Cleaning up controller: {self.__class__.__name__}")

            # Clear service references
            self.services.clear()

            # Clear configuration
            self.config.clear()

        except Exception as e:
            self.logger.error(f"Error during controller cleanup: {str(e)}")

    @abstractmethod
    def get_required_services(self) -> List[str]:
        """
        Get list of required service names.

        Returns:
            List of service names this controller requires
        """
        pass

    def __str__(self) -> str:
        """String representation of the controller."""
        return f"{self.__class__.__name__}(services={len(self.services)}, initialized={self._is_initialized})"

    def __repr__(self) -> str:
        """Detailed representation of the controller."""
        return f"{self.__class__.__name__}(services={list(self.services.keys())}, config={self.config})"