#!/usr/bin/env python3
"""
Service Registry

Central registry for managing and accessing business logic services.
Provides service discovery, dependency injection, and lifecycle management.

Author: Dimitrije Stojanovic
Date: December 2025
Created: Phase 3 Refactoring - Service Layer Architecture
"""

from typing import Dict, Type, Any, Optional, List, Callable
import logging
from threading import Lock

from .base_service import BaseService, ServiceResult


class ServiceRegistry:
    """
    Central registry for managing business logic services.

    Implements service locator and dependency injection patterns:
    - Service registration and discovery
    - Singleton instance management
    - Dependency resolution
    - Configuration propagation
    """

    _instance: Optional['ServiceRegistry'] = None
    _lock = Lock()

    def __new__(cls) -> 'ServiceRegistry':
        """Ensure singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the service registry."""
        if not hasattr(self, '_initialized'):
            self._services: Dict[str, Type[BaseService]] = {}
            self._instances: Dict[str, BaseService] = {}
            self._configurations: Dict[str, Dict[str, Any]] = {}
            self._dependencies: Dict[str, List[str]] = {}
            self._factories: Dict[str, Callable[..., BaseService]] = {}
            self.logger = logging.getLogger(self.__class__.__name__)
            self._initialized = True

    def register_service(self, service_class: Type[BaseService], name: Optional[str] = None,
                        config: Optional[Dict[str, Any]] = None,
                        dependencies: Optional[List[str]] = None,
                        factory: Optional[Callable[..., BaseService]] = None) -> None:
        """
        Register a service class with the registry.

        Args:
            service_class: The service class to register
            name: Optional service name (defaults to class name)
            config: Optional configuration for the service
            dependencies: Optional list of dependency service names
            factory: Optional factory function for creating service instances
        """
        service_name = name or service_class.__name__

        self._services[service_name] = service_class
        if config:
            self._configurations[service_name] = config
        if dependencies:
            self._dependencies[service_name] = dependencies
        if factory:
            self._factories[service_name] = factory

        self.logger.info(f"Registered service: {service_name}")

    def get_service(self, service_name: str, **kwargs) -> Optional[BaseService]:
        """
        Get a service instance by name.

        Args:
            service_name: Name of the service
            **kwargs: Additional arguments for service creation

        Returns:
            Service instance or None if not found
        """
        # Return existing instance if available (singleton pattern)
        if service_name in self._instances:
            return self._instances[service_name]

        # Check if service is registered
        if service_name not in self._services:
            self.logger.error(f"Service not registered: {service_name}")
            return None

        try:
            # Resolve dependencies first
            dependencies = self._resolve_dependencies(service_name)
            if dependencies is None:
                return None

            # Prepare configuration
            config = self._configurations.get(service_name, {})
            config.update(kwargs)
            config['dependencies'] = dependencies

            # Create service instance
            if service_name in self._factories:
                instance = self._factories[service_name](config=config)
            else:
                service_class = self._services[service_name]
                instance = service_class(config=config)

            # Store instance for singleton access
            self._instances[service_name] = instance

            self.logger.info(f"Created service instance: {service_name}")
            return instance

        except Exception as e:
            self.logger.error(f"Error creating service {service_name}: {str(e)}")
            return None

    def _resolve_dependencies(self, service_name: str) -> Optional[Dict[str, BaseService]]:
        """
        Resolve dependencies for a service.

        Args:
            service_name: Name of the service

        Returns:
            Dictionary of dependency services or None if resolution failed
        """
        dependencies = {}
        required_deps = self._dependencies.get(service_name, [])

        for dep_name in required_deps:
            dep_service = self.get_service(dep_name)
            if dep_service is None:
                self.logger.error(f"Failed to resolve dependency {dep_name} for {service_name}")
                return None
            dependencies[dep_name] = dep_service

        return dependencies

    def list_services(self) -> List[str]:
        """Get list of registered service names."""
        return list(self._services.keys())

    def list_instances(self) -> List[str]:
        """Get list of active service instance names."""
        return list(self._instances.keys())

    def get_service_info(self, service_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a registered service.

        Args:
            service_name: Name of the service

        Returns:
            Service information dictionary or None if not found
        """
        if service_name not in self._services:
            return None

        info = {
            'name': service_name,
            'class': self._services[service_name].__name__,
            'module': self._services[service_name].__module__,
            'has_config': service_name in self._configurations,
            'has_dependencies': service_name in self._dependencies,
            'has_factory': service_name in self._factories,
            'has_instance': service_name in self._instances,
            'dependencies': self._dependencies.get(service_name, [])
        }

        # Get service-specific info if instance exists
        if service_name in self._instances:
            try:
                service_info = self._instances[service_name].get_service_info()
                info['service_info'] = service_info
            except Exception as e:
                info['service_info_error'] = str(e)

        return info

    def clear_instance(self, service_name: str) -> bool:
        """
        Clear a service instance (for testing or reinitialization).

        Args:
            service_name: Name of the service

        Returns:
            True if instance was cleared, False if not found
        """
        if service_name in self._instances:
            del self._instances[service_name]
            self.logger.info(f"Cleared service instance: {service_name}")
            return True
        return False

    def clear_all_instances(self) -> None:
        """Clear all service instances."""
        self._instances.clear()
        self.logger.info("Cleared all service instances")

    def configure_service(self, service_name: str, config: Dict[str, Any]) -> None:
        """
        Configure a service (before instantiation).

        Args:
            service_name: Name of the service
            config: Configuration dictionary
        """
        self._configurations[service_name] = config
        self.logger.info(f"Configured service: {service_name}")

    def validate_services(self) -> ServiceResult:
        """
        Validate all registered services.

        Returns:
            ServiceResult indicating validation outcome
        """
        validation_results = []

        for service_name in self._services:
            try:
                # Try to get service info (validates basic registration)
                info = self.get_service_info(service_name)
                if info is None:
                    validation_results.append(f"Service {service_name}: Registration invalid")
                    continue

                # Check dependencies exist
                deps = self._dependencies.get(service_name, [])
                for dep in deps:
                    if dep not in self._services:
                        validation_results.append(f"Service {service_name}: Missing dependency {dep}")

                # Try instantiating if no dependencies (to validate constructor)
                if not deps:
                    try:
                        test_instance = self.get_service(service_name)
                        if test_instance is None:
                            validation_results.append(f"Service {service_name}: Failed to instantiate")
                        else:
                            # Clear test instance
                            self.clear_instance(service_name)
                    except Exception as e:
                        validation_results.append(f"Service {service_name}: Instantiation error - {str(e)}")

            except Exception as e:
                validation_results.append(f"Service {service_name}: Validation error - {str(e)}")

        if validation_results:
            return ServiceResult(
                success=False,
                error="Service validation failed",
                metadata={'validation_errors': validation_results}
            )
        else:
            return ServiceResult(
                success=True,
                data="All services valid",
                metadata={'validated_services': list(self._services.keys())}
            )

    def __str__(self) -> str:
        """String representation of the registry."""
        return f"ServiceRegistry(services={len(self._services)}, instances={len(self._instances)})"


# Global service registry instance
service_registry = ServiceRegistry()


def register_service(service_class: Type[BaseService], name: Optional[str] = None,
                    config: Optional[Dict[str, Any]] = None,
                    dependencies: Optional[List[str]] = None) -> None:
    """Convenience function to register a service with the global registry."""
    service_registry.register_service(service_class, name, config, dependencies)


def get_service(service_name: str, **kwargs) -> Optional[BaseService]:
    """Convenience function to get a service from the global registry."""
    return service_registry.get_service(service_name, **kwargs)