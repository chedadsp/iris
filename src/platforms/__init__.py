#!/usr/bin/env python3
"""
Platform-specific optimizations module.

This module contains platform-specific code that was previously
scattered across multiple launcher implementations.

Note: Renamed from 'platform' to 'platforms' to avoid naming
conflict with Python's built-in platform module.
"""

from .platform_detector import PlatformDetector
from .macos_optimizer import MacOSOptimizer

__all__ = [
    'PlatformDetector',
    'MacOSOptimizer',
]