"""Configuration loading: settings.yaml + secrets from .env / environment + credentials.yaml.

Secrets (Alpaca keys, SMTP password, webhook URL) come only from the
environment, optionally populated from a `.env` file at the project root.
`.env` is git-ignored; never put keys in settings.yaml or credentials.yaml.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

CONFIG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CONFIG_DIR.parent
ENV_FILE = PROJECT_ROOT / ".env"

_ENV_OVERRIDES = {
    "ALPACA_API_KEY": ("alpaca", "api_key"),
    "ALPACA_SECRET_KEY": ("alpaca", "secret_key"),
    "ALERT_WEBHOOK_URL": ("webhook", "url"),
    "SMTP_PASSWORD": ("email", "password"),
}


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as fh:
        return yaml.safe_load(fh) or {}


def load_dotenv(path: Path = ENV_FILE) -> Dict[str, str]:
    """Load KEY=VALUE lines into os.environ without overriding variables already set."""
    if not path.exists():
        return {}
    loaded = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        if key and key not in os.environ:
            os.environ[key] = value
            loaded[key] = value
    return loaded


def load_settings(path: Optional[str] = None) -> Dict[str, Any]:
    return _read_yaml(Path(path) if path else CONFIG_DIR / "settings.yaml")


def load_credentials(path: Optional[str] = None, env_file: Optional[Path] = None) -> Dict[str, Any]:
    """Non-secret settings from credentials.yaml (e.g. SMTP host), secrets from .env / environment."""
    load_dotenv(env_file or ENV_FILE)
    creds = _read_yaml(Path(path) if path else CONFIG_DIR / "credentials.yaml")
    if creds.pop("alpaca", None):
        logging.getLogger(__name__).warning(
            "Ignoring Alpaca keys in credentials.yaml: set ALPACA_API_KEY / ALPACA_SECRET_KEY in .env instead")
    for env_var, (section, key) in _ENV_OVERRIDES.items():
        value = os.environ.get(env_var)
        if value:
            creds.setdefault(section, {})[key] = value
    return creds
