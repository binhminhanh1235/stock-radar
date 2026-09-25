"""Nạp cấu hình YAML tập trung. Mọi ngưỡng tham số nằm trong config/config.yaml."""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

_lock = threading.Lock()
_cache: dict[str, dict] = {}


def load_config(path: str | Path | None = None) -> dict:
    """Nạp config.yaml (có cache theo đường dẫn tuyệt đối)."""
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    p = p.resolve()
    with _lock:
        if p not in _cache:
            if not p.exists():
                raise FileNotFoundError(f"Không tìm thấy file config: {p}")
            with open(p, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            if not isinstance(cfg, dict):
                raise ValueError(f"Config không hợp lệ: {p}")
            _cache[p] = cfg
        return _cache[p]


def cfg_get(cfg: dict, dotted: str, default: Any = None) -> Any:
    """Truy cập khoá lồng nhau dạng 'signals.min_early_score'."""
    node: Any = cfg
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def parameters_version(cfg: dict) -> str:
    """Phiên bản tham số, ghi vào bảng predictions để đối chiếu sau này."""
    return str(cfg.get("version", "unversioned"))
