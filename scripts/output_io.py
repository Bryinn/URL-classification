"""Shared serialization for generated pipeline artifacts."""

from __future__ import annotations

import json
from pathlib import Path


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")
