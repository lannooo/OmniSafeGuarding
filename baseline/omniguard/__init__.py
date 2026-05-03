"""
OmniGuard: Multimodal Safety Evaluation Framework
"""

__version__ = "1.0.0"

# Lazy imports to avoid circular dependencies and heavy imports
def __getattr__(name):
    if name == "OmniGuardEvaluator":
        from .omniguard_evaluator import OmniGuardEvaluator
        return OmniGuardEvaluator
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = ["OmniGuardEvaluator"]
