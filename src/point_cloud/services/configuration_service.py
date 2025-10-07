#!/usr/bin/env python3
"""
Configuration Service

Centralized configuration management for all point cloud services.
Provides configuration loading, validation, and service-specific configuration.

Author: Dimitrije Stojanovic
Date: December 2025
Created: Phase 3 Refactoring - Service Layer Architecture
"""

from typing import Dict, Any, Optional
from pathlib import Path
import json

from .base_service import BaseService, ServiceResult
from ..config import GUIConfig, HumanModelConfig


class ConfigurationService(BaseService):
    """
    Centralized configuration management service.

    Handles loading, validation, and distribution of configuration
    to other services in the system.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the configuration service.

        Args:
            config: Optional base configuration dictionary
        """
        super().__init__(config)
        self._service_configs: Dict[str, Dict[str, Any]] = {}
        self._default_configs = self._load_default_configurations()

    def _load_default_configurations(self) -> Dict[str, Dict[str, Any]]:
        """Load default configurations from existing config classes."""
        defaults = {}

        try:
            # Load GUI configuration
            gui_config = GUIConfig()
            defaults['gui'] = {
                'visualization': {
                    'default_window_size': gui_config.DEFAULT_WINDOW_SIZE,
                    'default_point_size': gui_config.DEFAULT_POINT_SIZE,
                    'background_color': gui_config.BACKGROUND_COLOR,
                    'selection_color': gui_config.SELECTION_COLOR
                },
                'interaction': {
                    'double_click_delay': gui_config.DOUBLE_CLICK_DELAY,
                    'cube_opacity': gui_config.CUBE_OPACITY,
                    'default_cube_size': gui_config.DEFAULT_CUBE_SIZE
                }
            }
        except Exception as e:
            self.logger.warning(f"Could not load GUI config: {e}")
            defaults['gui'] = {}

        try:
            # Load human model configuration
            human_config = HumanModelConfig()
            defaults['human_model'] = {
                'proportions': human_config.PROPORTIONS,
                'default_scale': 1.0,
                'default_model_type': 'seated',
                'default_num_points': 1200,
                'seated_height': 0.6,
                'standing_height': 1.7
            }
        except Exception as e:
            self.logger.warning(f"Could not load human model config: {e}")
            defaults['human_model'] = {}

        # Service-specific defaults
        defaults['cube_selection'] = {
            'padding_percentage': 0.1,
            'optimization_enabled': True,
            'default_colors': {
                'selected': [0, 255, 0],
                'unselected': [128, 128, 128]
            }
        }

        defaults['human_positioning'] = {
            'position_optimization': True,
            'auto_scale': True,
            'collision_detection': False,
            'model_types': ['seated', 'standing']
        }

        return defaults

    def get_service_configuration(self, service_name: str) -> ServiceResult:
        """
        Get configuration for a specific service.

        Args:
            service_name: Name of the service

        Returns:
            ServiceResult with configuration data
        """
        # Check if we have service-specific configuration
        if service_name in self._service_configs:
            config = self._service_configs[service_name]
        else:
            # Fall back to defaults
            config = self._default_configs.get(service_name, {})

        return self.create_success_result(
            data=config,
            service_name=service_name,
            config_source='custom' if service_name in self._service_configs else 'default'
        )

    def set_service_configuration(self, service_name: str, configuration: Dict[str, Any]) -> ServiceResult:
        """
        Set configuration for a specific service.

        Args:
            service_name: Name of the service
            configuration: Configuration dictionary

        Returns:
            ServiceResult indicating success/failure
        """
        try:
            # Validate configuration structure (basic validation)
            if not isinstance(configuration, dict):
                return self.create_error_result("Configuration must be a dictionary")

            # Store configuration
            self._service_configs[service_name] = configuration.copy()

            self.logger.info(f"Updated configuration for service: {service_name}")
            return self.create_success_result(
                data=f"Configuration updated for {service_name}",
                service_name=service_name,
                config_keys=list(configuration.keys())
            )

        except Exception as e:
            return self.create_error_result(f"Error setting configuration: {str(e)}")

    def load_configuration_from_file(self, file_path: Path, service_name: Optional[str] = None) -> ServiceResult:
        """
        Load configuration from JSON file.

        Args:
            file_path: Path to JSON configuration file
            service_name: Optional service name (if None, loads all configurations)

        Returns:
            ServiceResult indicating success/failure
        """
        try:
            if not file_path.exists():
                return self.create_error_result(f"Configuration file not found: {file_path}")

            with open(file_path, 'r') as f:
                config_data = json.load(f)

            if service_name:
                # Load configuration for specific service
                if service_name in config_data:
                    self._service_configs[service_name] = config_data[service_name]
                    return self.create_success_result(
                        data=f"Loaded configuration for {service_name}",
                        service_name=service_name,
                        file_path=str(file_path)
                    )
                else:
                    return self.create_error_result(f"Service {service_name} not found in configuration file")
            else:
                # Load all configurations
                loaded_services = []
                for svc_name, svc_config in config_data.items():
                    self._service_configs[svc_name] = svc_config
                    loaded_services.append(svc_name)

                return self.create_success_result(
                    data=f"Loaded configurations for {len(loaded_services)} services",
                    loaded_services=loaded_services,
                    file_path=str(file_path)
                )

        except json.JSONDecodeError as e:
            return self.create_error_result(f"Invalid JSON in configuration file: {str(e)}")
        except Exception as e:
            return self.create_error_result(f"Error loading configuration file: {str(e)}")

    def save_configuration_to_file(self, file_path: Path, service_name: Optional[str] = None) -> ServiceResult:
        """
        Save configuration to JSON file.

        Args:
            file_path: Path to save the configuration file
            service_name: Optional service name (if None, saves all configurations)

        Returns:
            ServiceResult indicating success/failure
        """
        try:
            # Prepare data to save
            if service_name:
                if service_name not in self._service_configs:
                    return self.create_error_result(f"No configuration found for service: {service_name}")
                save_data = {service_name: self._service_configs[service_name]}
            else:
                save_data = self._service_configs.copy()

            # Ensure parent directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Save to file
            with open(file_path, 'w') as f:
                json.dump(save_data, f, indent=2)

            return self.create_success_result(
                data=f"Configuration saved to {file_path}",
                file_path=str(file_path),
                services_saved=list(save_data.keys())
            )

        except Exception as e:
            return self.create_error_result(f"Error saving configuration file: {str(e)}")

    def get_all_configurations(self) -> Dict[str, Dict[str, Any]]:
        """Get all service configurations (both custom and defaults)."""
        all_configs = {}

        # Include defaults
        for service_name, default_config in self._default_configs.items():
            all_configs[service_name] = default_config.copy()

        # Override with custom configurations
        for service_name, custom_config in self._service_configs.items():
            if service_name in all_configs:
                all_configs[service_name].update(custom_config)
            else:
                all_configs[service_name] = custom_config.copy()

        return all_configs

    def validate_service_configuration(self, service_name: str, configuration: Dict[str, Any]) -> ServiceResult:
        """
        Validate configuration for a specific service.

        Args:
            service_name: Name of the service
            configuration: Configuration to validate

        Returns:
            ServiceResult with validation results
        """
        validation_errors = []

        try:
            # Basic validation rules
            if not isinstance(configuration, dict):
                validation_errors.append("Configuration must be a dictionary")

            # Service-specific validation
            if service_name == 'cube_selection':
                if 'padding_percentage' in configuration:
                    if not isinstance(configuration['padding_percentage'], (int, float)) or \
                       configuration['padding_percentage'] < 0 or configuration['padding_percentage'] > 1:
                        validation_errors.append("padding_percentage must be a number between 0 and 1")

            elif service_name == 'human_model':
                if 'default_scale' in configuration:
                    if not isinstance(configuration['default_scale'], (int, float)) or \
                       configuration['default_scale'] <= 0:
                        validation_errors.append("default_scale must be a positive number")

                if 'default_num_points' in configuration:
                    if not isinstance(configuration['default_num_points'], int) or \
                       configuration['default_num_points'] < 100:
                        validation_errors.append("default_num_points must be an integer >= 100")

            if validation_errors:
                return self.create_error_result(
                    error="Configuration validation failed",
                    validation_errors=validation_errors
                )
            else:
                return self.create_success_result(
                    data="Configuration is valid",
                    service_name=service_name
                )

        except Exception as e:
            return self.create_error_result(f"Error during validation: {str(e)}")

    def get_service_info(self) -> Dict[str, Any]:
        """Get service information and capabilities."""
        return {
            'service_name': 'ConfigurationService',
            'description': 'Centralized configuration management for all point cloud services',
            'capabilities': [
                'service_configuration_management',
                'configuration_file_operations',
                'configuration_validation',
                'default_configuration_loading'
            ],
            'current_state': {
                'custom_configurations': len(self._service_configs),
                'default_configurations': len(self._default_configs),
                'total_configurations': len(self.get_all_configurations())
            },
            'supported_operations': {
                'get_service_configuration': 'Get configuration for a specific service',
                'set_service_configuration': 'Set configuration for a specific service',
                'load_configuration_from_file': 'Load configuration from JSON file',
                'save_configuration_to_file': 'Save configuration to JSON file',
                'validate_service_configuration': 'Validate service configuration',
                'get_all_configurations': 'Get all service configurations'
            },
            'managed_services': list(self._default_configs.keys()),
            'custom_configured_services': list(self._service_configs.keys())
        }