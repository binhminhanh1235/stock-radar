"""Logging chung: ra console + file logs/app.log."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from app.config import PROJECT_ROOT, cfg_get, load_config

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_configured = False


def setup_logging(level: str | None = None, logfile: str | None = None) -> None:
    global _configured
    root = logging.getLogger()
    if _configured:
        return
    cfg = load_config()
    level = level or str(cfg_get(cfg, "logging.level", "INFO"))
    logfile = logfile or str(cfg_get(cfg, "logging.file", "logs/app.log"))
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(logging.Formatter(_FORMAT))
    root.addHandler(sh)

    try:
        lp = Path(logfile)
        if not lp.is_absolute():
            lp = PROJECT_ROOT / lp
        lp.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(lp, encoding="utf-8")
        fh.setFormatter(logging.Formatter(_FORMAT))
        root.addHandler(fh)
    except OSError:  # pragma: no cover - quyền thư mục
        root.warning("Không mở được file log %s, chỉ log ra console", logfile)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
