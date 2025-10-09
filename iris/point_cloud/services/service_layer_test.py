#!/usr/bin/env python3
"""
Service Layer Architecture Test

Simple test to validate that the service layer architecture is working correctly.
This serves as both a test and a demonstration of the service layer capabilities.

Author: Dimitrije Stojanovic
Date: December 2025
Created: Phase 3 Refactoring - Service Layer Architecture
"""

import numpy as np
from typing import Dict, Any


def test_service_registry():
    """Test service registry functionality."""
    print("=== Testing Service Registry ===")

    # Import the service components
    from . import service_registry, get_service

    # List registered services
    services = service_registry.list_services()
    print(f"Registered services: {services}")

    # Validate services
    validation_result = service_registry.validate_services()
    print(f"Service validation: {'✅ PASSED' if validation_result.success else '❌ FAILED'}")
    if not validation_result.success:
        print(f"Validation errors: {validation_result.metadata.get('validation_errors', [])}")

    return validation_result.success


def test_configuration_service():
    """Test configuration service functionality."""
    print("\n=== Testing Configuration Service ===")

    try:
        from . import get_service

        # Get configuration service
        config_service = get_service('ConfigurationService')
        if not config_service:
            print("❌ FAILED: Could not create ConfigurationService")
            return False

        print("✅ ConfigurationService created successfully")

        # Test getting service info
        info = config_service.get_service_info()
        print(f"Service capabilities: {info['capabilities']}")

        # Test getting configuration for a service
        cube_config_result = config_service.get_service_configuration('cube_selection')
        if cube_config_result.success:
            print(f"✅ Got cube_selection config: {list(cube_config_result.data.keys())}")
        else:
            print(f"❌ Failed to get cube_selection config: {cube_config_result.error}")
            return False

        # Test setting configuration
        test_config = {'test_setting': 'test_value', 'numeric_setting': 42}
        set_result = config_service.set_service_configuration('test_service', test_config)
        if set_result.success:
            print("✅ Successfully set test configuration")
        else:
            print(f"❌ Failed to set configuration: {set_result.error}")
            return False

        return True

    except Exception as e:
        print(f"❌ FAILED: Configuration service test error: {str(e)}")
        return False


def test_cube_selection_service():
    """Test cube selection service functionality."""
    print("\n=== Testing Cube Selection Service ===")

    try:
        from . import get_service

        # Create test point cloud data
        test_points = np.random.rand(1000, 3) * 10  # Random points in 10x10x10 cube

        # Get cube selection service
        cube_service = get_service('CubeSelectionService')
        if not cube_service:
            print("❌ FAILED: Could not create CubeSelectionService")
            return False

        print("✅ CubeSelectionService created successfully")

        # Set point cloud data
        set_result = cube_service.set_points(test_points)
        if not set_result.success:
            print(f"❌ Failed to set points: {set_result.error}")
            return False

        print(f"✅ Loaded {cube_service.get_point_count()} points")

        # Test service info
        info = cube_service.get_service_info()
        print(f"Service capabilities: {len(info['capabilities'])} capabilities")

        # Test cube selection
        cube_bounds = (2.0, 8.0, 2.0, 8.0, 2.0, 8.0)  # Select middle portion
        selection_result = cube_service.select_points_in_cube(cube_bounds)

        if selection_result.has_selection:
            print(f"✅ Selected {selection_result.point_count} points from cube")
        else:
            print("❌ Cube selection failed")
            return False

        # Test statistics
        stats = cube_service.get_selection_statistics()
        if 'error' not in stats:
            print(f"✅ Selection statistics: {stats['selection_percentage']:.1f}% of points selected")
        else:
            print(f"❌ Failed to get statistics: {stats['error']}")
            return False

        return True

    except Exception as e:
        print(f"❌ FAILED: Cube selection service test error: {str(e)}")
        return False


def test_human_positioning_service():
    """Test human positioning service functionality."""
    print("\n=== Testing Human Positioning Service ===")

    try:
        from . import get_service

        # Create test point cloud data (vehicle interior simulation)
        test_points = np.random.rand(500, 3) * 5  # Random points in 5x5x5 space

        # Get human positioning service
        human_service = get_service('HumanPositioningService')
        if not human_service:
            print("❌ FAILED: Could not create HumanPositioningService")
            return False

        print("✅ HumanPositioningService created successfully")

        # Set point cloud data
        set_result = human_service.set_point_cloud_data(test_points)
        if not set_result.success:
            print(f"❌ Failed to set point cloud data: {set_result.error}")
            return False

        print(f"✅ Loaded {human_service.get_point_cloud_count()} points")

        # Test service info
        info = human_service.get_service_info()
        print(f"Service capabilities: {len(info['capabilities'])} capabilities")
        print(f"Supported model types: {info['supported_model_types']}")

        # Test optimal position calculation
        optimal_pos = human_service.calculate_optimal_position()
        print(f"✅ Calculated optimal position: ({optimal_pos[0]:.2f}, {optimal_pos[1]:.2f}, {optimal_pos[2]:.2f})")

        # Test human model generation
        model_result = human_service.generate_human_model(
            position=optimal_pos,
            scale=1.0,
            model_type='seated'
        )

        if model_result.has_model:
            print(f"✅ Generated human model with {model_result.point_count} points")
        else:
            print("❌ Human model generation failed")
            return False

        # Test model statistics
        stats = human_service.get_model_statistics()
        if 'error' not in stats:
            print(f"✅ Model statistics: scale={stats['scale']}, type={stats.get('model_type', 'N/A')}")
        else:
            print(f"❌ Failed to get model statistics: {stats['error']}")
            return False

        return True

    except Exception as e:
        print(f"❌ FAILED: Human positioning service test error: {str(e)}")
        return False


def run_service_layer_tests() -> bool:
    """Run all service layer tests."""
    print("🧪 Running Service Layer Architecture Tests")
    print("=" * 50)

    tests = [
        ("Service Registry", test_service_registry),
        ("Configuration Service", test_configuration_service),
        ("Cube Selection Service", test_cube_selection_service),
        ("Human Positioning Service", test_human_positioning_service),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"\n✅ {test_name} PASSED")
            else:
                print(f"\n❌ {test_name} FAILED")
        except Exception as e:
            print(f"\n❌ {test_name} FAILED with exception: {str(e)}")

    print("\n" + "=" * 50)
    print(f"🧪 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All service layer tests PASSED! Service architecture is working correctly.")
        return True
    else:
        print(f"⚠️  {total - passed} tests failed. Service layer needs attention.")
        return False


if __name__ == "__main__":
    success = run_service_layer_tests()
    exit(0 if success else 1)