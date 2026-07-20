import os
import yaml
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

def _load_yaml(name: str) -> dict:
    p = _ROOT / "config" / name
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_config() -> dict:
    return _load_yaml("settings.yaml")

def env(key: str, default: str = "") -> str:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")
    return os.environ.get(key, default)
