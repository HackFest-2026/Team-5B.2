"""Import the read-only customer_agent_sim package from the repository root."""

from __future__ import annotations

import sys

from app.config import REPO_DIR

if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import customer_agent_sim as cas  # noqa: E402

MODULE_VERSION = "customer_agent_sim (local copy, shadow mode)"

__all__ = ["MODULE_VERSION", "cas"]
