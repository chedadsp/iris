#!/usr/bin/env python3
"""
Base Widget Class

Abstract base class for all GUI widgets providing common functionality
and establishing widget patterns for pure UI components.

Author: Dimitrije Stojanovic
Date: December 2025
Created: Phase 3 Refactoring - GUI/Business Logic Separation
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Callable, List, Tuple
import tkinter as tk
from tkinter import ttk
import logging


class WidgetError(Exception):
    """Exception raised by widget operations."""
    pass


class BaseWidget(ABC):
    """
    Abstract base class for all GUI widgets.

    Provides common functionality for pure UI components:
    - Widget lifecycle management
    - Event handling and callbacks
    - Configuration management
    - Error handling
    - Logging
    """

    def __init__(self, parent: tk.Widget, controller: Optional[Any] = None,
                 widget_config: Optional[Dict[str, Any]] = None):
        """
        Initialize base widget.

        Args:
            parent: Parent tkinter widget
            controller: Optional controller for handling business logic
            widget_config: Optional widget configuration
        """
        self.parent = parent
        self.controller = controller
        self.config = widget_config or {}
        self.logger = logging.getLogger(f"{self.__class__.__name__}")

        # Widget state
        self._is_initialized = False
        self._is_enabled = True
        self._callbacks: Dict[str, List[Callable]] = {}

        # Main widget container
        self.frame: Optional[tk.Frame] = None

        # Initialize widget
        self._setup_logging()
        self._create_widget()
        self._setup_bindings()

        self._is_initialized = True

    def _setup_logging(self) -> None:
        """Setup widget-specific logging."""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    @abstractmethod
    def _create_widget(self) -> None:
        """
        Create the widget's UI components.

        Must be implemented by subclasses to define the actual UI structure.
        """
        pass

    def _setup_bindings(self) -> None:
        """
        Setup widget event bindings.

        Can be overridden by subclasses for custom event handling.
        """
        pass

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """
        Get widget configuration value.

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        return self.config.get(key, default)

    def set_config_value(self, key: str, value: Any) -> None:
        """
        Set widget configuration value.

        Args:
            key: Configuration key
            value: Configuration value
        """
        self.config[key] = value
        self.logger.debug(f"Config updated: {key} = {value}")

    def add_callback(self, event_type: str, callback: Callable) -> None:
        """
        Add callback for widget events.

        Args:
            event_type: Type of event (e.g., 'selection_changed', 'value_updated')
            callback: Callback function to execute
        """
        if event_type not in self._callbacks:
            self._callbacks[event_type] = []
        self._callbacks[event_type].append(callback)
        self.logger.debug(f"Added callback for event: {event_type}")

    def remove_callback(self, event_type: str, callback: Callable) -> bool:
        """
        Remove callback for widget events.

        Args:
            event_type: Type of event
            callback: Callback function to remove

        Returns:
            True if callback was removed, False if not found
        """
        if event_type in self._callbacks and callback in self._callbacks[event_type]:
            self._callbacks[event_type].remove(callback)
            self.logger.debug(f"Removed callback for event: {event_type}")
            return True
        return False

    def trigger_callback(self, event_type: str, *args, **kwargs) -> None:
        """
        Trigger callbacks for a specific event type.

        Args:
            event_type: Type of event
            *args: Arguments to pass to callbacks
            **kwargs: Keyword arguments to pass to callbacks
        """
        if event_type in self._callbacks:
            for callback in self._callbacks[event_type]:
                try:
                    callback(*args, **kwargs)
                except Exception as e:
                    self.logger.error(f"Error in callback for {event_type}: {str(e)}")

    def set_enabled(self, enabled: bool) -> None:
        """
        Enable or disable the widget.

        Args:
            enabled: Whether the widget should be enabled
        """
        self._is_enabled = enabled

        if self.frame:
            state = 'normal' if enabled else 'disabled'
            self._set_children_state(self.frame, state)

    def _set_children_state(self, widget: tk.Widget, state: str) -> None:
        """
        Recursively set state of all child widgets.

        Args:
            widget: Parent widget
            state: State to set ('normal' or 'disabled')
        """
        try:
            widget.configure(state=state)
        except tk.TclError:
            # Some widgets don't support state configuration
            pass

        for child in widget.winfo_children():
            self._set_children_state(child, state)

    def is_enabled(self) -> bool:
        """Check if widget is enabled."""
        return self._is_enabled

    def is_initialized(self) -> bool:
        """Check if widget is initialized."""
        return self._is_initialized

    def show(self) -> None:
        """Show the widget."""
        if self.frame:
            self.frame.pack(fill=tk.BOTH, expand=True)

    def hide(self) -> None:
        """Hide the widget."""
        if self.frame:
            self.frame.pack_forget()

    def destroy(self) -> None:
        """Destroy the widget and clean up resources."""
        try:
            self.logger.info(f"Destroying widget: {self.__class__.__name__}")

            # Clear callbacks
            self._callbacks.clear()

            # Destroy UI components
            if self.frame:
                self.frame.destroy()

            # Clear references
            self.controller = None
            self.parent = None

        except Exception as e:
            self.logger.error(f"Error destroying widget: {str(e)}")

    def validate_controller(self, required_methods: List[str]) -> bool:
        """
        Validate that the controller has required methods.

        Args:
            required_methods: List of required method names

        Returns:
            True if controller is valid, False otherwise
        """
        if not self.controller:
            self.logger.error("No controller set")
            return False

        for method_name in required_methods:
            if not hasattr(self.controller, method_name):
                self.logger.error(f"Controller missing required method: {method_name}")
                return False
            if not callable(getattr(self.controller, method_name)):
                self.logger.error(f"Controller method not callable: {method_name}")
                return False

        return True

    def safe_call_controller(self, method_name: str, *args, **kwargs) -> Any:
        """
        Safely call a controller method with error handling.

        Args:
            method_name: Name of the controller method
            *args: Arguments to pass to the method
            **kwargs: Keyword arguments to pass to the method

        Returns:
            Method result or None if failed
        """
        if not self.controller:
            self.logger.error("No controller available")
            return None

        if not hasattr(self.controller, method_name):
            self.logger.error(f"Controller method not found: {method_name}")
            return None

        try:
            method = getattr(self.controller, method_name)
            result = method(*args, **kwargs)
            self.logger.debug(f"Controller method {method_name} called successfully")
            return result
        except Exception as e:
            self.logger.error(f"Error calling controller method {method_name}: {str(e)}")
            return None

    def create_labeled_entry(self, parent: tk.Widget, label_text: str,
                           initial_value: str = "", **entry_kwargs) -> Tuple[tk.Label, tk.Entry]:
        """
        Utility method to create a labeled entry widget.

        Args:
            parent: Parent widget
            label_text: Label text
            initial_value: Initial entry value
            **entry_kwargs: Additional Entry widget arguments

        Returns:
            Tuple of (Label, Entry) widgets
        """
        label = tk.Label(parent, text=label_text)
        entry = tk.Entry(parent, **entry_kwargs)
        if initial_value:
            entry.insert(0, initial_value)
        return label, entry

    def create_labeled_scale(self, parent: tk.Widget, label_text: str,
                           from_: float, to: float, initial_value: float = None,
                           **scale_kwargs) -> Tuple[tk.Label, tk.Scale]:
        """
        Utility method to create a labeled scale widget.

        Args:
            parent: Parent widget
            label_text: Label text
            from_: Minimum value
            to: Maximum value
            initial_value: Initial scale value
            **scale_kwargs: Additional Scale widget arguments

        Returns:
            Tuple of (Label, Scale) widgets
        """
        label = tk.Label(parent, text=label_text)
        scale = tk.Scale(parent, from_=from_, to=to, orient=tk.HORIZONTAL, **scale_kwargs)
        if initial_value is not None:
            scale.set(initial_value)
        return label, scale

    def get_widget_info(self) -> Dict[str, Any]:
        """
        Get information about the widget.

        Returns:
            Dictionary with widget information
        """
        return {
            'widget_name': self.__class__.__name__,
            'is_initialized': self._is_initialized,
            'is_enabled': self._is_enabled,
            'has_controller': self.controller is not None,
            'config_keys': list(self.config.keys()),
            'callback_events': list(self._callbacks.keys()),
            'has_frame': self.frame is not None
        }

    def __str__(self) -> str:
        """String representation of the widget."""
        return f"{self.__class__.__name__}(enabled={self._is_enabled}, initialized={self._is_initialized})"

    def __repr__(self) -> str:
        """Detailed representation of the widget."""
        return f"{self.__class__.__name__}(controller={self.controller}, config={self.config})"