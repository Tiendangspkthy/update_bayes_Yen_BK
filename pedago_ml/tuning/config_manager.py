import os
import tempfile
from typing import Any, Dict

import yaml

from pedago_ml.utils.logger import logger


def set_nested(config_dict: Dict[str, Any], key_path: str, value: Any) -> None:
    """Set a nested value in a dict using dot-separated keys."""
    keys = key_path.split(".")
    d = config_dict
    for key in keys[:-1]:
        if key not in d or d[key] is None:
            d[key] = {}
        d = d[key]
    d[keys[-1]] = value


class ConfigManager:
    """Writes a YAML copy with updates applied via dotted keys."""

    def create_updated_config(self, original_path: str, updates: Dict[str, Any]) -> str:
        """Write a temporary YAML config with updates and return its path."""
        with open(original_path, "r", encoding="utf-8") as f:
            config_dict: Dict[str, Any] = yaml.safe_load(f)

        for key_path, value in updates.items():
            set_nested(config_dict, key_path, value)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as temp_file:
            yaml.dump(config_dict, temp_file, default_flow_style=False)
            temp_path = temp_file.name

        logger.debug(f"Created temporary config: {temp_path}")
        return temp_path

    def cleanup_config(self, config_path: str) -> None:
        """Delete a temporary YAML config if it exists."""
        if config_path and os.path.exists(config_path):
            try:
                os.unlink(config_path)
                logger.debug(f"Cleaned up temporary config: {config_path}")
            except OSError as e:
                logger.warning(f"Failed to cleanup config {config_path}: {e}")
