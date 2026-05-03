"""
Safety evaluation datasets for OmniGuard
"""

from .base import SafetyDataset, SafetyEvaluator

# Lazy imports to avoid heavy dependencies
def __getattr__(name):
    if name == "BeaverTails":
        from .beavertails import BeaverTails
        return BeaverTails
    elif name == "VLGuard":
        from .vlguard import VLGuard
        return VLGuard
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = ["SafetyDataset", "SafetyEvaluator", "BeaverTails", "VLGuard"]
