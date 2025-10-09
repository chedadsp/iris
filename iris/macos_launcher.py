#!/usr/bin/env python3
"""
macOS Launcher for LIDAR GUI Application

This launcher sets up the proper VTK environment for macOS to prevent segmentation faults
and provides a stable GUI experience.

Author: Dimitrije Stojanovic  
Date: September 2025
"""

import os
import sys
import gc
import atexit
from pathlib import Path


def setup_macos_environment():
    """Set up macOS-specific environment for VTK stability."""
    # VTK environment variables for macOS stability
    vtk_env_vars = {
        'VTK_RENDER_WINDOW_MAIN_THREAD': '1',
        'VTK_USE_COCOA': '1', 
        'VTK_SILENCE_GET_VOID_POINTER_WARNINGS': '1',
        'VTK_DEBUG_LEAKS': '0',
        'VTK_USE_OFFSCREEN': '0',
        # Additional macOS-specific settings
        'PYVISTA_OFF_SCREEN': '0',
        'PYVISTA_USE_PANEL': '0'
    }
    
    for key, value in vtk_env_vars.items():
        os.environ[key] = value
        print(f"Set {key}={value}")


def setup_python_path():
    """Set up Python path to include point cloud modules."""
    current_dir = Path(__file__).parent
    point_cloud_dir = current_dir / "point_cloud"
    
    if point_cloud_dir.exists():
        sys.path.insert(0, str(current_dir))
        print(f"Added to Python path: {current_dir}")


def cleanup_on_exit():
    """Clean up resources on exit."""
    print("Cleaning up VTK resources...")
    gc.collect()


def check_dependencies():
    """Check if required dependencies are available."""
    missing_deps = []
    
    try:
        import tkinter
        print("✓ tkinter available")
    except ImportError:
        missing_deps.append("tkinter")
    
    try:
        import pyvista
        print("✓ PyVista available")
    except ImportError:
        missing_deps.append("pyvista")
    
    try:
        import vtk
        print("✓ VTK available")
    except ImportError:
        missing_deps.append("vtk")
    
    try:
        import numpy
        print("✓ NumPy available")
    except ImportError:
        missing_deps.append("numpy")
    
    try:
        import pye57
        print("✓ pye57 available")
    except ImportError:
        missing_deps.append("pye57")
    
    try:
        import sklearn
        print("✓ scikit-learn available")
    except ImportError:
        missing_deps.append("scikit-learn")
    
    if missing_deps:
        print(f"\n❌ Missing dependencies: {', '.join(missing_deps)}")
        print("Install with: poetry install")
        return False
    
    print("\n✅ All dependencies available")
    return True


def launch_gui():
    """Launch the LIDAR GUI application."""
    try:
        # Import the GUI application
        from lidar_gui_app import LidarGUIApp
        
        print("🚀 Starting LIDAR GUI Application...")
        
        # Create and run the application
        app = LidarGUIApp()
        app.run()
        
    except ImportError as e:
        print(f"❌ Failed to import GUI application: {e}")
        print("Make sure lidar_gui_app.py is in the same directory.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Failed to start GUI application: {e}")
        sys.exit(1)


def main():
    """Main entry point for macOS launcher."""
    print("=" * 60)
    print("LIDAR Point Cloud Analysis Suite - macOS Launcher")
    print("=" * 60)
    
    # Check if running on macOS
    if sys.platform != "darwin":
        print("⚠️  This launcher is optimized for macOS.")
        print("You can run the GUI directly with: python lidar_gui_app.py")
    else:
        print("✅ Running on macOS - applying VTK optimizations")
    
    # Set up environment
    print("\n📋 Setting up environment...")
    setup_macos_environment()
    setup_python_path()
    
    # Register cleanup
    atexit.register(cleanup_on_exit)
    
    # Check dependencies
    print("\n🔍 Checking dependencies...")
    if not check_dependencies():
        print("\n💡 Run 'poetry install' to install missing dependencies.")
        sys.exit(1)
    
    # Launch GUI
    print("\n" + "=" * 60)
    launch_gui()


if __name__ == "__main__":
    main()