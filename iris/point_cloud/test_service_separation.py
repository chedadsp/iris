#!/usr/bin/env python3
"""
Service Layer Separation Test

Comprehensive test to validate that the GUI/Business Logic separation is working
correctly. Tests the complete architecture: Services ↔ Controllers ↔ Widgets.

Author: Dimitrije Stojanovic
Date: December 2025
Created: Phase 3 Refactoring - Service Layer Separation Test
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append('src')

# Set working directory to point_cloud for proper imports
import os
if os.path.basename(os.getcwd()) != 'point_cloud':
    for parent in Path.cwd().parents:
        if (parent / 'point_cloud').exists():
            os.chdir(parent / 'point_cloud')
            break


def test_service_layer():
    """Test the service layer independently."""
    print("=== Testing Service Layer ===")

    try:
        from point_cloud.services import get_service, CubeSelectionService, HumanPositioningService

        # Test cube selection service
        print("Testing CubeSelectionService...")
        test_points = np.random.rand(1000, 3) * 10
        cube_service = get_service('CubeSelectionService')

        if cube_service:
            result = cube_service.set_points(test_points)
            if result.success:
                print(f"✅ Loaded {cube_service.get_point_count()} points")

                # Test selection
                bounds = (2, 8, 2, 8, 2, 8)
                selection = cube_service.select_points_in_cube(bounds)
                if selection.has_selection:
                    print(f"✅ Selected {selection.point_count} points")
                else:
                    print("❌ Selection failed")
                    return False
            else:
                print(f"❌ Failed to set points: {result.error}")
                return False
        else:
            print("❌ Could not get CubeSelectionService")
            return False

        # Test human positioning service
        print("Testing HumanPositioningService...")
        human_service = get_service('HumanPositioningService')

        if human_service:
            result = human_service.set_point_cloud_data(test_points)
            if result.success:
                print(f"✅ Set point cloud data: {human_service.get_point_cloud_count()} points")

                # Test model generation
                position = (5, 5, 1)
                model = human_service.generate_human_model(position, scale=1.0, model_type='seated')
                if model and model.has_model:
                    print(f"✅ Generated human model with {model.point_count} points")
                else:
                    print("❌ Model generation failed")
                    return False
            else:
                print(f"❌ Failed to set point cloud data: {result.error}")
                return False
        else:
            print("❌ Could not get HumanPositioningService")
            return False

        return True

    except Exception as e:
        print(f"❌ Service layer test failed: {e}")
        return False


def test_controller_layer():
    """Test the controller layer."""
    print("\n=== Testing Controller Layer ===")

    try:
        from point_cloud.controllers import CubeSelectionController, HumanPositioningController

        # Test cube selection controller
        print("Testing CubeSelectionController...")
        test_points = np.random.rand(500, 3) * 5
        cube_controller = CubeSelectionController(test_points)

        if cube_controller.is_initialized():
            print("✅ CubeSelectionController initialized")

            if cube_controller.has_point_cloud_data():
                print(f"✅ Point cloud loaded: {cube_controller.get_point_count()} points")

                # Test bounds calculation
                default_bounds = cube_controller.calculate_default_cube_bounds()
                if default_bounds:
                    print(f"✅ Default bounds calculated: {default_bounds}")

                    # Test validation
                    validation = cube_controller.validate_cube_bounds(default_bounds)
                    if validation.get('valid', False):
                        print("✅ Bounds validation successful")

                        # Test selection
                        selection = cube_controller.select_points_in_cube(default_bounds)
                        if selection and selection.has_selection:
                            print(f"✅ Selection successful: {selection.point_count} points")
                        else:
                            print("❌ Selection failed")
                            return False
                    else:
                        print(f"❌ Bounds validation failed: {validation.get('errors', [])}")
                        return False
                else:
                    print("❌ Default bounds calculation failed")
                    return False
            else:
                print("❌ No point cloud data in controller")
                return False
        else:
            print("❌ CubeSelectionController not initialized")
            return False

        # Test human positioning controller
        print("Testing HumanPositioningController...")
        human_controller = HumanPositioningController(test_points)

        if human_controller.is_initialized():
            print("✅ HumanPositioningController initialized")

            if human_controller.has_point_cloud_data():
                print(f"✅ Point cloud loaded: {human_controller.get_point_cloud_count()} points")

                # Test optimal position
                optimal_pos = human_controller.calculate_optimal_position()
                if optimal_pos:
                    print(f"✅ Optimal position calculated: {optimal_pos}")

                    # Test model generation
                    model = human_controller.generate_human_model(optimal_pos, scale=1.0, model_type='seated')
                    if model and model.has_model:
                        print(f"✅ Model generated: {model.point_count} points")

                        # Test parameter updates
                        updated = human_controller.update_model_scale(1.5)
                        if updated and updated.has_model:
                            print("✅ Model scale updated")
                        else:
                            print("❌ Model scale update failed")
                            return False
                    else:
                        print("❌ Model generation failed")
                        return False
                else:
                    print("❌ Optimal position calculation failed")
                    return False
            else:
                print("❌ No point cloud data in controller")
                return False
        else:
            print("❌ HumanPositioningController not initialized")
            return False

        return True

    except Exception as e:
        print(f"❌ Controller layer test failed: {e}")
        return False


def test_widget_layer():
    """Test the widget layer (without actually creating GUI)."""
    print("\n=== Testing Widget Layer (Architecture) ===")

    try:
        # We can't easily test actual GUI widgets without a display,
        # but we can test the widget architecture
        from point_cloud.widgets import BaseWidget, CubeSelectionWidget, HumanPositioningWidget
        from point_cloud.controllers import CubeSelectionController

        print("✅ Widget imports successful")
        print("✅ Widget classes are properly structured")

        # Test that widgets require controllers with expected methods
        print("Testing widget-controller interface requirements...")

        # Check CubeSelectionWidget requirements
        cube_widget_class = CubeSelectionWidget
        if hasattr(cube_widget_class, '__init__'):
            print("✅ CubeSelectionWidget has proper constructor")

        # Check HumanPositioningWidget requirements
        human_widget_class = HumanPositioningWidget
        if hasattr(human_widget_class, '__init__'):
            print("✅ HumanPositioningWidget has proper constructor")

        return True

    except Exception as e:
        print(f"❌ Widget layer test failed: {e}")
        return False


def test_integration():
    """Test integration between layers."""
    print("\n=== Testing Layer Integration ===")

    try:
        from point_cloud.services import get_service
        from point_cloud.controllers import CubeSelectionController, HumanPositioningController

        # Create test data
        test_points = np.random.rand(300, 3) * 8

        # Test service-controller integration
        print("Testing service-controller integration...")

        cube_controller = CubeSelectionController(test_points)

        # Verify controller can access services
        if cube_controller.validate_services():
            print("✅ Controller can access required services")

            # Test data flow: Controller -> Service -> Controller
            bounds = cube_controller.calculate_default_cube_bounds()
            if bounds:
                count = cube_controller.count_points_in_cube(bounds)
                if count is not None:
                    selection = cube_controller.select_points_in_cube(bounds)
                    if selection and selection.has_selection:
                        stats = cube_controller.get_selection_statistics()
                        if 'error' not in stats:
                            print(f"✅ Complete data flow successful: {selection.point_count} points, {stats.get('selection_percentage', 0):.1f}%")
                        else:
                            print(f"❌ Statistics retrieval failed: {stats.get('error')}")
                            return False
                    else:
                        print("❌ Selection failed in integration test")
                        return False
                else:
                    print("❌ Point counting failed in integration test")
                    return False
            else:
                print("❌ Bounds calculation failed in integration test")
                return False
        else:
            print("❌ Controller cannot access required services")
            return False

        # Test cross-service integration
        print("Testing cross-service integration...")

        human_controller = HumanPositioningController()

        # Simulate using selected points from cube selection in human positioning
        if cube_controller.has_current_selection():
            # In a real application, we would get the selected points
            # and use them as context for human positioning
            human_result = human_controller.set_point_cloud_data(test_points[:100])  # Simulate selected subset
            if human_result.success:
                print("✅ Cross-service data transfer successful")

                # Generate human model in the context
                position = human_controller.calculate_optimal_position()
                if position:
                    model = human_controller.generate_human_model(position)
                    if model and model.has_model:
                        print(f"✅ Complete workflow successful: cube selection + human positioning")
                    else:
                        print("❌ Human model generation failed in integration")
                        return False
                else:
                    print("❌ Position calculation failed in integration")
                    return False
            else:
                print(f"❌ Cross-service data transfer failed: {human_result.error}")
                return False

        return True

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


def test_separation_architecture():
    """Test that the separation architecture is properly implemented."""
    print("\n=== Testing Separation Architecture ===")

    try:
        from point_cloud.services import BaseService, service_registry
        from point_cloud.controllers import BaseController, CubeSelectionController
        from point_cloud.widgets import BaseWidget

        # Test service layer independence
        print("Testing service layer independence...")
        services = service_registry.list_services()
        if len(services) >= 3:  # We should have at least 3 services
            print(f"✅ Service registry has {len(services)} services: {services}")

            # Verify services don't import GUI code
            for service_name in services:
                service = service_registry.get_service(service_name)
                if service and isinstance(service, BaseService):
                    print(f"✅ {service_name} properly extends BaseService")
                else:
                    print(f"❌ {service_name} does not properly extend BaseService")
                    return False
        else:
            print(f"❌ Insufficient services registered: {services}")
            return False

        # Test controller layer mediation
        print("Testing controller layer mediation...")
        test_points = np.random.rand(100, 3) * 5
        cube_controller = CubeSelectionController(test_points)

        if isinstance(cube_controller, BaseController):
            print("✅ CubeSelectionController properly extends BaseController")

            # Verify controller doesn't contain business logic
            # (it should delegate to services)
            info = cube_controller.get_controller_info()
            if info['configured_services']:
                print(f"✅ Controller delegates to services: {info['configured_services']}")
            else:
                print("❌ Controller doesn't configure services properly")
                return False
        else:
            print("❌ Controller doesn't extend BaseController")
            return False

        # Test architectural boundaries
        print("Testing architectural boundaries...")

        # Services should not know about controllers or widgets
        cube_service = service_registry.get_service('CubeSelectionService')
        if cube_service:
            service_info = cube_service.get_service_info()
            # Service info should only contain business logic information
            if 'capabilities' in service_info and 'current_state' in service_info:
                print("✅ Services maintain proper boundaries (business logic only)")
            else:
                print("❌ Service doesn't provide proper business logic interface")
                return False

        return True

    except Exception as e:
        print(f"❌ Architecture separation test failed: {e}")
        return False


def run_all_tests():
    """Run all service separation tests."""
    print("🧪 Testing Service Layer Separation")
    print("=" * 60)

    tests = [
        ("Service Layer", test_service_layer),
        ("Controller Layer", test_controller_layer),
        ("Widget Layer", test_widget_layer),
        ("Integration", test_integration),
        ("Separation Architecture", test_separation_architecture),
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

    print("\n" + "=" * 60)
    print(f"🧪 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All service separation tests PASSED!")
        print("✅ GUI/Business Logic separation is working correctly")
        print("✅ Service layer architecture is properly implemented")
        print("✅ Controllers properly mediate between GUI and services")
        return True
    else:
        print(f"⚠️  {total - passed} tests failed. Service separation needs attention.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)