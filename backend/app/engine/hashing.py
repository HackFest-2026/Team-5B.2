"""Canonical JSON and content hashes, so logically equal inputs always hash identically."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel


def _plain(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", by_alias=True, exclude_none=True)
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=_plain)


def sha256_hex(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def content_hash(value: Any) -> str:
    """Short, stable hash of any contract (e.g. a strategy draft)."""
    return sha256_hex(value)[:16]
