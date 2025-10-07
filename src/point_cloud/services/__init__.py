#!/usr/bin/env python3
"""
Service Layer Module

This module contains the business logic services extracted from GUI components.
Services handle data processing, algorithms, and business rules without any
GUI dependencies.

Author: Dimitrije Stojanovic
Date: December 2025
Created: Phase 3 Refactoring
"""

from .base_service import BaseService, ServiceResult
from .service_registry import ServiceRegistry, service_registry, register_service, get_service
from .cube_selection_service import CubeSelectionService, CubeSelectionResult
from .human_positioning_service import HumanPositioningService, HumanModelResult
from .configuration_service import ConfigurationService

# Register services with the global registry
register_service(CubeSelectionService, name='CubeSelectionService')
register_service(HumanPositioningService, name='HumanPositioningService')
register_service(ConfigurationService, name='ConfigurationService')

__all__ = [
    # Base classes
    'BaseService',
    'ServiceResult',

    # Service registry
    'ServiceRegistry',
    'service_registry',
    'register_service',
    'get_service',

    # Services
    'CubeSelectionService',
    'CubeSelectionResult',
    'HumanPositioningService',
    'HumanModelResult',
    'ConfigurationService',
]