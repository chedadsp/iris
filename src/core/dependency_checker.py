#!/usr/bin/env python3
"""
Dependency Checker

Consolidated dependency checking logic extracted from the original
launcher implementations. Provides consistent dependency validation
across all launcher modes.
"""

from typing import Dict, List, Tuple, Optional
import importlib.util
import sys


class DependencyInfo:
    """Information about a dependency."""

    def __init__(self, name: str, required: bool = True, description: str = ""):
        self.name = name
        self.required = required
        self.description = description
        self.available = False
        self.error_message = ""


class DependencyChecker:
    """Centralized dependency checking and reporting."""

    # Core dependencies required by all modes
    CORE_DEPENDENCIES = [
        DependencyInfo('numpy', required=True, description='Numerical computing'),
        DependencyInfo('pathlib', required=True, description='Path handling'),
    ]

    # GUI-specific dependencies
    GUI_DEPENDENCIES = [
        DependencyInfo('tkinter', required=True, description='GUI framework'),
        DependencyInfo('pyvista', required=False, description='3D visualization'),
        DependencyInfo('vtk', required=False, description='VTK rendering backend'),
    ]

    # Analysis-specific dependencies
    ANALYSIS_DEPENDENCIES = [
        DependencyInfo('pye57', required=False, description='E57 file format support'),
        DependencyInfo('sklearn', required=False, description='Machine learning algorithms'),
        DependencyInfo('scipy', required=False, description='Scientific computing'),
        DependencyInfo('matplotlib', required=False, description='Plotting and visualization'),
    ]

    # Optional ROS support
    ROS_DEPENDENCIES = [
        DependencyInfo('bagpy', required=False, description='ROS bag file support'),
    ]

    def __init__(self):
        """Initialize the dependency checker."""
        self.dependencies: Dict[str, DependencyInfo] = {}
        self._register_default_dependencies()

    def _register_default_dependencies(self):
        """Register all default dependencies."""
        for deps in [self.CORE_DEPENDENCIES, self.GUI_DEPENDENCIES,
                    self.ANALYSIS_DEPENDENCIES, self.ROS_DEPENDENCIES]:
            for dep in deps:
                self.dependencies[dep.name] = dep

    def check_dependency(self, dep_name: str) -> bool:
        """
        Check if a specific dependency is available.

        Args:
            dep_name: Name of the dependency to check

        Returns:
            True if available, False otherwise
        """
        if dep_name not in self.dependencies:
            return False

        dep = self.dependencies[dep_name]

        try:
            # Special handling for tkinter (built-in module)
            if dep_name == 'tkinter':
                import tkinter
                dep.available = True
            # Special handling for pathlib (built-in since Python 3.4)
            elif dep_name == 'pathlib':
                from pathlib import Path
                dep.available = True
            else:
                # Try to import the module
                spec = importlib.util.find_spec(dep_name)
                if spec is not None:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    dep.available = True
                else:
                    dep.available = False
                    dep.error_message = f"Module '{dep_name}' not found"

        except ImportError as e:
            dep.available = False
            dep.error_message = str(e)
        except Exception as e:
            dep.available = False
            dep.error_message = f"Unexpected error: {e}"

        return dep.available

    def check_all_dependencies(self) -> None:
        """Check all registered dependencies."""
        for dep_name in self.dependencies:
            self.check_dependency(dep_name)

    def check_mode_dependencies(self, mode: str) -> Tuple[bool, List[str], List[str]]:
        """
        Check dependencies required for a specific mode.

        Args:
            mode: Launcher mode ('gui', 'cli', 'headless')

        Returns:
            Tuple of (all_required_available, missing_required, missing_optional)
        """
        # Define which dependencies are needed for each mode
        mode_requirements = {
            'gui': self.CORE_DEPENDENCIES + self.GUI_DEPENDENCIES + self.ANALYSIS_DEPENDENCIES,
            'cli': self.CORE_DEPENDENCIES + self.ANALYSIS_DEPENDENCIES,
            'headless': self.CORE_DEPENDENCIES + self.ANALYSIS_DEPENDENCIES,
        }

        if mode not in mode_requirements:
            raise ValueError(f"Unknown mode: {mode}")

        required_deps = mode_requirements[mode]
        missing_required = []
        missing_optional = []

        for dep_info in required_deps:
            is_available = self.check_dependency(dep_info.name)

            if not is_available:
                if dep_info.required:
                    missing_required.append(dep_info.name)
                else:
                    missing_optional.append(dep_info.name)

        all_required_available = len(missing_required) == 0
        return all_required_available, missing_required, missing_optional

    def get_dependency_status(self) -> Dict[str, bool]:
        """
        Get the status of all dependencies.

        Returns:
            Dictionary mapping dependency names to availability status
        """
        return {name: dep.available for name, dep in self.dependencies.items()}

    def print_dependency_report(self, mode: Optional[str] = None) -> None:
        """
        Print a comprehensive dependency report.

        Args:
            mode: Optional mode to focus the report on
        """
        print("🔍 Checking dependencies...")

        if mode:
            # Mode-specific report
            success, missing_required, missing_optional = self.check_mode_dependencies(mode)
            print(f"\nDependency check for {mode} mode:")

            if success:
                print(f"✅ All required dependencies available for {mode} mode")
            else:
                print(f"❌ Missing required dependencies for {mode} mode:")
                for dep in missing_required:
                    print(f"  - {dep}: {self.dependencies[dep].description}")

            if missing_optional:
                print(f"⚠️  Missing optional dependencies:")
                for dep in missing_optional:
                    print(f"  - {dep}: {self.dependencies[dep].description}")
                print("Some features may be disabled.")
        else:
            # Full report
            self.check_all_dependencies()

            for dep_name, dep_info in self.dependencies.items():
                status = "✅" if dep_info.available else "❌"
                req_label = "required" if dep_info.required else "optional"
                print(f"{status} {dep_name} ({req_label}): {dep_info.description}")

                if not dep_info.available and dep_info.error_message:
                    print(f"    Error: {dep_info.error_message}")

    def get_installation_suggestions(self, missing_deps: List[str]) -> List[str]:
        """
        Get installation suggestions for missing dependencies.

        Args:
            missing_deps: List of missing dependency names

        Returns:
            List of installation commands
        """
        suggestions = []

        if missing_deps:
            suggestions.append("💡 Install missing dependencies with:")
            suggestions.append("poetry install")

            # Special suggestions for optional dependencies
            if 'bagpy' in missing_deps:
                suggestions.append("poetry install -E ros  # For ROS bag support")

            if 'torch' in missing_deps or 'torchvision' in missing_deps:
                suggestions.append("poetry install -E ptv3  # For AI enhancement")

        return suggestions