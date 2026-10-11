"""Audit records: who did what, to which resource and version, and whether it succeeded."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Optional

from sqlalchemy.orm import Session, sessionmaker

from app.auth import Principal
from app.db import utcnow
from app.models import AuditLogRow


def record(s: Session, who: Principal, operation: str, resource_type: str, resource_id: Optional[str] = None,
           version: Optional[str] = None, outcome: str = "SUCCESS", details: Optional[dict[str, Any]] = None) -> None:
    s.add(AuditLogRow(at=utcnow(), actor=who.user, role=who.role, operation=operation, resource_type=resource_type,
                      resource_id=resource_id, version=version, outcome=outcome, details=details or {}))


@contextmanager
def audited(factory: sessionmaker[Session], who: Principal, operation: str, resource_type: str,
            resource_id: Optional[str] = None) -> Iterator[Session]:
    """Run a mutation in one transaction; on failure, roll it back and write a FAILED audit row separately."""
    s = factory()
    try:
        yield s
        s.commit()
    except Exception as e:
        s.rollback()
        with factory() as s2:
            record(s2, who, operation, resource_type, resource_id, outcome="FAILED",
                   details={"error": getattr(e, "code", type(e).__name__), "message": str(e)[:500]})
            s2.commit()
        raise
    finally:
        s.close()
