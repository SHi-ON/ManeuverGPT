"""Centralized version metadata for the package."""

from __future__ import annotations

import re
from pathlib import Path


def _load_version() -> str:
    """Load the package version from ``pyproject.toml``.

    The repository keeps a single source of truth in ``pyproject.toml``.
    This module exposes the value at runtime to avoid drift.
    """

    pyproject_path = Path(__file__).resolve().parents[2] / 'pyproject.toml'
    text = pyproject_path.read_text(encoding='utf-8')
    match = re.search(r'^\s*version\s*=\s*"([^"]+)"$', text, re.MULTILINE)
    if not match:
        raise RuntimeError('Unable to load version from pyproject.toml')
    return match.group(1)


__version__ = _load_version()
