"""Configuration loading utilities."""

import json
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load a JSON configuration file.

    Args:
        config_path: Path to config file (relative to project root or absolute)

    Returns:
        Configuration dictionary
    """
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / config_path

    with open(path, "r") as f:
        config = json.load(f)

    return config


def save_config(config: Dict[str, Any], output_path: str):
    """
    Save configuration to JSON file.

    Args:
        config: Configuration dictionary
        output_path: Output file path
    """
    path = Path(output_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / output_path

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(config, f, indent=2)


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple config dictionaries (later configs override earlier ones).

    Args:
        *configs: Variable number of config dicts

    Returns:
        Merged config dictionary
    """
    result = {}
    for config in configs:
        result.update(config)
    return result
