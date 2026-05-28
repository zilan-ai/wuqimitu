import json
import os
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "adb_path": "adb",
    "serial": None,
    "adb_address": None,
    "resource_dir": "resource",
    "pipeline_dir": "tasks",
    "log_level": "INFO",
    "log_to_file": False,
    "log_dir": "logs",
    "tasks": {
        "daily_reward": True,
        "mail": True,
        "dispatch": True,
        "visit_friend": True,
        "shop_buy": False,
        "stamina_farm": False,
        "stamina_stage": "1-7",
        "stamina_times": 5,
        "nightmare": False,
        "abyss": False,
    },
    "delays": {
        "between_tasks": 2.0,
        "after_tap": 0.5,
        "after_swipe": 0.8,
        "loading_wait": 5.0,
    },
}


class Config:
    def __init__(self, config_path: Optional[str] = None):
        self._data = dict(DEFAULT_CONFIG)
        if config_path and os.path.exists(config_path):
            self.load(config_path)

    def load(self, path: str):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._deep_merge(self._data, data)
            logger.info("Config loaded: %s", path)
        except Exception as e:
            logger.error("Failed to load config %s: %s", path, e)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split(".")
        data = self._data
        for k in keys:
            if isinstance(data, dict) and k in data:
                data = data[k]
            else:
                return default
        return data

    def set(self, key: str, value: Any):
        keys = key.split(".")
        data = self._data
        for k in keys[:-1]:
            if k not in data or not isinstance(data[k], dict):
                data[k] = {}
            data = data[k]
        data[keys[-1]] = value

    @property
    def data(self) -> Dict[str, Any]:
        return self._data

    @staticmethod
    def _deep_merge(base: dict, override: dict):
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                Config._deep_merge(base[key], value)
            else:
                base[key] = value
