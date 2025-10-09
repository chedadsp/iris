#!/usr/bin/env python3
"""
GUI Widgets Module

Pure GUI widgets separated from business logic. These widgets handle only
user interface concerns and delegate all business logic to services via controllers.

Author: Dimitrije Stojanovic
Date: December 2025
Created: Phase 3 Refactoring - GUI/Business Logic Separation
"""

from .base_widget import BaseWidget, WidgetError
from .cube_selection_widget import CubeSelectionWidget
from .human_positioning_widget import HumanPositioningWidget

__all__ = [
    # Base classes
    'BaseWidget',
    'WidgetError',

    # Widgets
    'CubeSelectionWidget',
    'HumanPositioningWidget',
]