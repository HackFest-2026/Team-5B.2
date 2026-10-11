"""Development-grade role boundary. NOT production security.

Callers send ``Authorization: Bearer dev:<role>:<user>``. Roles:
- analyst: business user; edits, validates and simulates strategies
- manager / admin: also approve, publish and roll back (maker-checker applies)
- customer: may only read the catalog and request quotes (customer-safe view)
A real deployment would replace this with an identity provider; the route-level role checks stay.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from fastapi import Header

from app.errors import ApiError

Role = Literal["analyst", "manager", "admin", "customer"]
BUSINESS: tuple[Role, ...] = ("analyst", "manager", "admin")
APPROVERS: tuple[Role, ...] = ("manager", "admin")
ANY: tuple[Role, ...] = ("analyst", "manager", "admin", "customer")

_TOKEN = re.compile(r"^Bearer dev:(analyst|manager|admin|customer):([A-Za-z0-9_.\-@]{1,64})$")


@dataclass(frozen=True)
class Principal:
    role: Role
    user: str

    @property
    def is_business(self) -> bool:
        return self.role in BUSINESS


def parse_principal(authorization: str | None) -> Principal:
    if not authorization:
        raise ApiError(401, "UNAUTHENTICATED", "Missing Authorization header (expected 'Bearer dev:<role>:<user>').")
    m = _TOKEN.match(authorization.strip())
    if not m:
        raise ApiError(401, "UNAUTHENTICATED", "Malformed Authorization header (expected 'Bearer dev:<role>:<user>').")
    return Principal(role=m.group(1), user=m.group(2))  # type: ignore[arg-type]


def require(*roles: Role):
    def dep(authorization: str | None = Header(default=None)) -> Principal:
        p = parse_principal(authorization)
        if p.role not in roles:
            raise ApiError(403, "FORBIDDEN", f"Role '{p.role}' may not perform this action (allowed: {', '.join(roles)}).")
        return p

    return dep
