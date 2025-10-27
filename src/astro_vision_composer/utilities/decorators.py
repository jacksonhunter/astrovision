"""Decorators for marking experimental and deprecated functionality.

This module provides decorators to mark code quality and status, ensuring
users are aware of experimental or low-quality components.
"""

import warnings
import functools
from typing import Callable, Any


def experimental(quality: str = "EXPERIMENTAL", warning: str = None):
    """Mark a function or method as experimental.

    Args:
        quality: Quality level (e.g., "LOW", "EXPERIMENTAL", "BETA")
        warning: Custom warning message to display

    Returns:
        Decorated function that issues warnings when called
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Construct warning message
            msg = f"[{quality}] {func.__name__} is experimental and may produce unreliable results."
            if warning:
                msg += f" {warning}"

            # Issue runtime warning
            warnings.warn(msg, category=UserWarning, stacklevel=2)

            # Call original function
            return func(*args, **kwargs)

        # Update docstring
        if wrapper.__doc__:
            warning_text = f"\n\n    .. warning::\n        **{quality} QUALITY**: This function is experimental.\n"
            if warning:
                warning_text += f"        {warning}\n"
            wrapper.__doc__ = warning_text + wrapper.__doc__

        return wrapper
    return decorator


def deprecated(reason: str = None, version: str = None, alternative: str = None):
    """Mark a function or method as deprecated.

    Args:
        reason: Reason for deprecation
        version: Version when deprecated
        alternative: Suggested alternative function/method

    Returns:
        Decorated function that issues deprecation warnings
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Construct deprecation message
            msg = f"{func.__name__} is deprecated"

            if version:
                msg += f" as of version {version}"

            if reason:
                msg += f". Reason: {reason}"

            if alternative:
                msg += f". Use {alternative} instead"

            # Issue deprecation warning
            warnings.warn(msg, category=DeprecationWarning, stacklevel=2)

            # Call original function
            return func(*args, **kwargs)

        # Update docstring
        if wrapper.__doc__:
            deprecation_text = f"\n\n    .. deprecated:: {version or 'current'}\n"
            if reason:
                deprecation_text += f"        {reason}\n"
            if alternative:
                deprecation_text += f"        Use :func:`{alternative}` instead.\n"
            wrapper.__doc__ = deprecation_text + wrapper.__doc__

        return wrapper
    return decorator


def requires_validation(message: str = "This function requires validation before production use"):
    """Mark a function as requiring validation.

    Args:
        message: Custom validation requirement message

    Returns:
        Decorated function that logs validation requirements
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            import logging
            logger = logging.getLogger(func.__module__)
            logger.warning(f"{func.__name__}: {message}")
            return func(*args, **kwargs)

        # Update docstring
        if wrapper.__doc__:
            validation_text = f"\n\n    .. note::\n        **VALIDATION REQUIRED**: {message}\n"
            wrapper.__doc__ = validation_text + wrapper.__doc__

        return wrapper
    return decorator