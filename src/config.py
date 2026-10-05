"""
Configuration loading utilities.

Every other module in `src/` receives its parameters through a `Config`
object produced here rather than reading `config.yaml` directly or
hard-coding values. This keeps the project's single source of truth in one
place and makes every function's dependencies explicit and testable (pass a
different `Config` in a unit test instead of monkeypatching constants).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Config:
    """Thin, read-only wrapper around the parsed config.yaml dictionary.

    Paths are resolved to absolute paths at construction time so that code
    can be invoked from any working directory (a notebook, a CLI script, the
    Streamlit app) without path-resolution bugs.
    """

    raw: dict[str, Any]
    root: Path

    # -- convenience dict-style access -------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, *keys: str, default: Any = None) -> Any:
        """Safe nested lookup, e.g. cfg.get('target', 'proxy', 'quantile_threshold')."""
        node: Any = self.raw
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node

    # -- resolved paths -----------------------------------------------------------------
    def path(self, key: str) -> Path:
        """Resolve a `paths.<key>` entry from config.yaml to an absolute Path."""
        rel = self.raw["paths"][key]
        return self.root / rel

    def ensure_output_dirs(self) -> None:
        for key in ("model_dir", "reports_dir", "figures_dir"):
            self.path(key).mkdir(parents=True, exist_ok=True)
        self.path("processed_data").parent.mkdir(parents=True, exist_ok=True)


def load_config(config_path: str | os.PathLike | None = None) -> Config:
    """Load `config.yaml` (or an explicit path) into a `Config` object.

    Parameters
    ----------
    config_path:
        Optional explicit path to a YAML config file. Defaults to
        `<project_root>/config.yaml`.
    """
    root = PROJECT_ROOT
    path = Path(config_path) if config_path is not None else root / "config.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found at '{path}'. Run scripts from the project "
            f"root, or pass an explicit --config path."
        )
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return Config(raw=raw, root=root)
