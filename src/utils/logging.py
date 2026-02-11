"""Logging utilities for experiments."""

import json
import logging
from pathlib import Path
from typing import Any, Dict
from datetime import datetime


def setup_logging(
    log_file: str,
    level: int = logging.INFO,
    also_console: bool = True,
) -> logging.Logger:
    """
    Set up logging to file and optionally console.

    Args:
        log_file: Path to log file
        level: Logging level
        also_console: Whether to also log to console

    Returns:
        Configured logger
    """
    logger = logging.getLogger("nextstep")
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers = []

    # File handler
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler
    if also_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_formatter = logging.Formatter(
            "%(levelname)s - %(message)s"
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    return logger


def create_run_manifest(
    output_dir: str,
    experiment_id: str,
    config: Dict[str, Any],
    **kwargs
) -> Dict[str, Any]:
    """
    Create a run.json manifest file with experiment metadata.

    Args:
        output_dir: Output directory path
        experiment_id: Unique experiment identifier
        config: Configuration used for the experiment
        **kwargs: Additional metadata fields

    Returns:
        Run manifest dictionary
    """
    import subprocess

    manifest = {
        "experiment_id": experiment_id,
        "timestamp": datetime.now().isoformat(),
        "config": config,
        **kwargs
    }

    # Add git info if available
    try:
        manifest["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()
        manifest["git_branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()
    except Exception:
        manifest["git_commit"] = "unknown"
        manifest["git_branch"] = "unknown"

    # Save to file
    output_path = Path(output_dir) / "run.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest
