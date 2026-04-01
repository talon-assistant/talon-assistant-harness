"""core/config.py — Configuration utilities for Talon Assistant."""


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning a new dict.
    Keys present in base but missing from override keep their base value."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result
