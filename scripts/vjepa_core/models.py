#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compat layer para que los scripts en `scripts/vjepa_core/` no dependan de
symlinks externos (BADAS-OPEN/...).

Re-exporta todo desde `src/thesis/models/models.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.exists():
    sys.path.insert(0, str(SRC_ROOT))

from thesis.models.models import *  # noqa: F401,F403

