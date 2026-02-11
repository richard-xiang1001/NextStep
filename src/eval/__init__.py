"""Evaluation package exports."""

from typing import Any, Dict

__all__ = ["run_baseline_experiment", "run_baseline_methods"]


def run_baseline_experiment(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Lazy export to avoid importing heavy dependencies on package import."""
    from .baselines import run_baseline_experiment as _run_baseline_experiment

    return _run_baseline_experiment(*args, **kwargs)


def run_baseline_methods(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Backward-compatible lazy export."""
    from .baselines import run_baseline_methods as _run_baseline_methods

    return _run_baseline_methods(*args, **kwargs)
