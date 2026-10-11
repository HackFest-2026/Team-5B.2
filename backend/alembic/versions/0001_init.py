"""Initial schema: domains, drafts, immutable versions, runs, decisions, shadow validations, audit.

Revision ID: 0001_init
Revises:
Create Date: 2026-10-11
"""

import sqlalchemy as sa
from alembic import op

revision = "0001_init"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "domains",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("name", sa.String(80), nullable=False),
        sa.Column("config", sa.JSON(), nullable=False),
        sa.Column("created_by", sa.String(80), nullable=False),
        sa.Column("created_at", sa.String(32), nullable=False),
        sa.Column("updated_at", sa.String(32), nullable=False),
    )
    op.create_table(
        "strategy_drafts",
        sa.Column("domain_id", sa.String(64), sa.ForeignKey("domains.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("config", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("content_hash", sa.String(32), nullable=False),
        sa.Column("source", sa.String(16), nullable=False),
        sa.Column("updated_by", sa.String(80), nullable=False),
        sa.Column("updated_at", sa.String(32), nullable=False),
        sa.CheckConstraint("status IN ('DRAFT','VALIDATED')", name="ck_draft_status"),
        sa.CheckConstraint("source IN ('manual','llm','mock')", name="ck_draft_source"),
    )
    op.create_table(
        "strategy_versions",
        sa.Column("id", sa.String(96), primary_key=True),
        sa.Column("domain_id", sa.String(64), sa.ForeignKey("domains.id", ondelete="CASCADE"), nullable=False),
        sa.Column("number", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("config", sa.JSON(), nullable=False),
        sa.Column("domain_snapshot", sa.JSON(), nullable=False),
        sa.Column("content_hash", sa.String(32), nullable=False),
        sa.Column("created_by", sa.String(80), nullable=False),
        sa.Column("approved_by", sa.String(80), nullable=False),
        sa.Column("approved_at", sa.String(32), nullable=False),
        sa.Column("published_at", sa.String(32), nullable=True),
        sa.Column("retired_at", sa.String(32), nullable=True),
        sa.Column("notes", sa.Text(), nullable=False),
        sa.UniqueConstraint("domain_id", "number", name="uq_version_number"),
        sa.CheckConstraint("status IN ('APPROVED','ACTIVE','RETIRED')", name="ck_version_status"),
    )
    op.create_index("ix_strategy_versions_domain_id", "strategy_versions", ["domain_id"])
    op.create_index("uq_one_active", "strategy_versions", ["domain_id"], unique=True,
                    sqlite_where=sa.text("status = 'ACTIVE'"), postgresql_where=sa.text("status = 'ACTIVE'"))
    op.create_table(
        "validation_runs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("domain_id", sa.String(64), sa.ForeignKey("domains.id", ondelete="CASCADE"), nullable=False),
        sa.Column("content_hash", sa.String(32), nullable=False),
        sa.Column("ok", sa.Boolean(), nullable=False),
        sa.Column("result", sa.JSON(), nullable=False),
        sa.Column("created_by", sa.String(80), nullable=False),
        sa.Column("created_at", sa.String(32), nullable=False),
    )
    op.create_index("ix_validation_runs_domain_id", "validation_runs", ["domain_id"])
    op.create_table(
        "simulation_runs",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("domain_id", sa.String(64), sa.ForeignKey("domains.id", ondelete="CASCADE"), nullable=False),
        sa.Column("content_hash", sa.String(32), nullable=False),
        sa.Column("version_id", sa.String(96), nullable=True),
        sa.Column("request", sa.JSON(), nullable=False),
        sa.Column("result", sa.JSON(), nullable=False),
        sa.Column("created_by", sa.String(80), nullable=False),
        sa.Column("created_at", sa.String(32), nullable=False),
    )
    op.create_index("ix_simulation_runs_domain_id", "simulation_runs", ["domain_id"])
    op.create_table(
        "decisions",
        sa.Column("id", sa.String(40), primary_key=True),
        sa.Column("domain_id", sa.String(64), sa.ForeignKey("domains.id", ondelete="CASCADE"), nullable=False),
        sa.Column("version_id", sa.String(96), nullable=False),
        sa.Column("product_id", sa.String(64), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("final_price", sa.Float(), nullable=True),
        sa.Column("request", sa.JSON(), nullable=False),
        sa.Column("result", sa.JSON(), nullable=False),
        sa.Column("created_by", sa.String(80), nullable=False),
        sa.Column("created_at", sa.String(32), nullable=False),
    )
    op.create_index("ix_decisions_domain_id", "decisions", ["domain_id"])
    op.create_index("ix_decisions_version_id", "decisions", ["version_id"])
    op.create_index("ix_decisions_created_at", "decisions", ["created_at"])
    op.create_table(
        "candidate_evaluations",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("decision_id", sa.String(40), sa.ForeignKey("decisions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("price", sa.Float(), nullable=False),
        sa.Column("eligible", sa.Boolean(), nullable=False),
        sa.Column("violations", sa.JSON(), nullable=False),
        sa.Column("p_buy", sa.Float(), nullable=True),
        sa.Column("expected_revenue", sa.Float(), nullable=True),
        sa.Column("expected_profit", sa.Float(), nullable=True),
        sa.Column("selection_status", sa.String(32), nullable=False),
    )
    op.create_index("ix_candidate_evaluations_decision_id", "candidate_evaluations", ["decision_id"])
    op.create_table(
        "shadow_validations",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("owner_type", sa.String(16), nullable=False),
        sa.Column("owner_id", sa.String(64), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("result", sa.JSON(), nullable=False),
        sa.Column("population_size", sa.Integer(), nullable=False),
        sa.Column("seed", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.String(32), nullable=False),
        sa.CheckConstraint("owner_type IN ('simulation','decision')", name="ck_shadow_owner"),
    )
    op.create_index("ix_shadow_validations_owner_id", "shadow_validations", ["owner_id"])
    op.create_table(
        "audit_logs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("at", sa.String(32), nullable=False),
        sa.Column("actor", sa.String(80), nullable=False),
        sa.Column("role", sa.String(16), nullable=False),
        sa.Column("operation", sa.String(48), nullable=False),
        sa.Column("resource_type", sa.String(32), nullable=False),
        sa.Column("resource_id", sa.String(96), nullable=True),
        sa.Column("version", sa.String(96), nullable=True),
        sa.Column("outcome", sa.String(16), nullable=False),
        sa.Column("details", sa.JSON(), nullable=False),
    )
    op.create_index("ix_audit_logs_at", "audit_logs", ["at"])
    op.create_table(
        "customers",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("domain_id", sa.String(64), sa.ForeignKey("domains.id", ondelete="CASCADE"), nullable=True),
        sa.Column("name", sa.String(80), nullable=False),
        sa.Column("profile", sa.JSON(), nullable=False),
    )


def downgrade() -> None:
    for table in ("customers", "audit_logs", "shadow_validations", "candidate_evaluations", "decisions",
                  "simulation_runs", "validation_runs", "strategy_versions", "strategy_drafts", "domains"):
        op.drop_table(table)
