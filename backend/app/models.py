"""ORM tables. Structured configuration is stored as JSON and validated by Pydantic on the way in and out."""

from __future__ import annotations

from typing import Any, Optional

from sqlalchemy import JSON, Boolean, CheckConstraint, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint, text
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base


class DomainRow(Base):
    __tablename__ = "domains"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(80))
    config: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_by: Mapped[str] = mapped_column(String(80))
    created_at: Mapped[str] = mapped_column(String(32))
    updated_at: Mapped[str] = mapped_column(String(32))


class StrategyDraftRow(Base):
    __tablename__ = "strategy_drafts"
    __table_args__ = (
        CheckConstraint("status IN ('DRAFT','VALIDATED')", name="ck_draft_status"),
        CheckConstraint("source IN ('manual','llm','mock')", name="ck_draft_source"),
    )
    domain_id: Mapped[str] = mapped_column(ForeignKey("domains.id", ondelete="CASCADE"), primary_key=True)
    config: Mapped[dict[str, Any]] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(16))
    content_hash: Mapped[str] = mapped_column(String(32))
    source: Mapped[str] = mapped_column(String(16))
    updated_by: Mapped[str] = mapped_column(String(80))
    updated_at: Mapped[str] = mapped_column(String(32))


class StrategyVersionRow(Base):
    __tablename__ = "strategy_versions"
    __table_args__ = (
        UniqueConstraint("domain_id", "number", name="uq_version_number"),
        CheckConstraint("status IN ('APPROVED','ACTIVE','RETIRED')", name="ck_version_status"),
        Index("uq_one_active", "domain_id", unique=True, sqlite_where=text("status = 'ACTIVE'"),
              postgresql_where=text("status = 'ACTIVE'")),
    )
    id: Mapped[str] = mapped_column(String(96), primary_key=True)
    domain_id: Mapped[str] = mapped_column(ForeignKey("domains.id", ondelete="CASCADE"), index=True)
    number: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16))
    config: Mapped[dict[str, Any]] = mapped_column(JSON)
    domain_snapshot: Mapped[dict[str, Any]] = mapped_column(JSON)
    content_hash: Mapped[str] = mapped_column(String(32))
    created_by: Mapped[str] = mapped_column(String(80))
    approved_by: Mapped[str] = mapped_column(String(80))
    approved_at: Mapped[str] = mapped_column(String(32))
    published_at: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    retired_at: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    notes: Mapped[str] = mapped_column(Text, default="")


class ValidationRunRow(Base):
    __tablename__ = "validation_runs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    domain_id: Mapped[str] = mapped_column(ForeignKey("domains.id", ondelete="CASCADE"), index=True)
    content_hash: Mapped[str] = mapped_column(String(32))
    ok: Mapped[bool] = mapped_column(Boolean)
    result: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_by: Mapped[str] = mapped_column(String(80))
    created_at: Mapped[str] = mapped_column(String(32))


class SimulationRunRow(Base):
    __tablename__ = "simulation_runs"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    domain_id: Mapped[str] = mapped_column(ForeignKey("domains.id", ondelete="CASCADE"), index=True)
    content_hash: Mapped[str] = mapped_column(String(32))
    version_id: Mapped[Optional[str]] = mapped_column(String(96), nullable=True)
    request: Mapped[dict[str, Any]] = mapped_column(JSON)
    result: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_by: Mapped[str] = mapped_column(String(80))
    created_at: Mapped[str] = mapped_column(String(32))


class DecisionRow(Base):
    __tablename__ = "decisions"
    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    domain_id: Mapped[str] = mapped_column(ForeignKey("domains.id", ondelete="CASCADE"), index=True)
    version_id: Mapped[str] = mapped_column(String(96), index=True)
    product_id: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(20))
    final_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    request: Mapped[dict[str, Any]] = mapped_column(JSON)
    result: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_by: Mapped[str] = mapped_column(String(80))
    created_at: Mapped[str] = mapped_column(String(32), index=True)


class CandidateEvaluationRow(Base):
    __tablename__ = "candidate_evaluations"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    decision_id: Mapped[str] = mapped_column(ForeignKey("decisions.id", ondelete="CASCADE"), index=True)
    price: Mapped[float] = mapped_column(Float)
    eligible: Mapped[bool] = mapped_column(Boolean)
    violations: Mapped[list[Any]] = mapped_column(JSON)
    p_buy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    expected_revenue: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    expected_profit: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    selection_status: Mapped[str] = mapped_column(String(32))


class ShadowValidationRow(Base):
    __tablename__ = "shadow_validations"
    __table_args__ = (CheckConstraint("owner_type IN ('simulation','decision')", name="ck_shadow_owner"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    owner_type: Mapped[str] = mapped_column(String(16))
    owner_id: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(20))
    result: Mapped[dict[str, Any]] = mapped_column(JSON)
    population_size: Mapped[int] = mapped_column(Integer)
    seed: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[str] = mapped_column(String(32))


class AuditLogRow(Base):
    __tablename__ = "audit_logs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    at: Mapped[str] = mapped_column(String(32), index=True)
    actor: Mapped[str] = mapped_column(String(80))
    role: Mapped[str] = mapped_column(String(16))
    operation: Mapped[str] = mapped_column(String(48))
    resource_type: Mapped[str] = mapped_column(String(32))
    resource_id: Mapped[Optional[str]] = mapped_column(String(96), nullable=True)
    version: Mapped[Optional[str]] = mapped_column(String(96), nullable=True)
    outcome: Mapped[str] = mapped_column(String(16))
    details: Mapped[dict[str, Any]] = mapped_column(JSON)


class CustomerRow(Base):
    __tablename__ = "customers"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    domain_id: Mapped[Optional[str]] = mapped_column(ForeignKey("domains.id", ondelete="CASCADE"), nullable=True)
    name: Mapped[str] = mapped_column(String(80))
    profile: Mapped[dict[str, Any]] = mapped_column(JSON)
