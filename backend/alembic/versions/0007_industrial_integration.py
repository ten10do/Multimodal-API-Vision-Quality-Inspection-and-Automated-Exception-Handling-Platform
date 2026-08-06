"""phase7: industrial integration (plc_events + inspection industrial state)

Revision ID: 0007_industrial_integration
Revises: 0006_anomaly_columns
Create Date: 2026-08-05

- plc_events: persistent PLC command log (idempotency + audit)
- inspections: industrial_state / industrial_final_state / plc_command /
  plc_status / plc_latency_ms / mes_sync_status
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0007_industrial_integration"
down_revision = "0006_anomaly_columns"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "plc_events",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("command_id", sa.String(length=128), nullable=False),
        sa.Column("product_id", sa.String(length=64), nullable=False),
        sa.Column("inspection_id", sa.String(length=64), nullable=False),
        sa.Column("command", sa.String(length=16), nullable=False),
        sa.Column("desired_command", sa.String(length=16), nullable=False),
        sa.Column("execution_status", sa.String(length=32), nullable=False),
        sa.Column("industrial_state", sa.String(length=32), nullable=True),
        sa.Column("adapter_type", sa.String(length=16), nullable=True),
        sa.Column("request_payload", sa.JSON(), nullable=False),
        sa.Column("response", sa.JSON(), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="PENDING"),
        sa.Column("retry_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("latency_ms", sa.Float(), nullable=True),
        sa.Column("reason_code", sa.String(length=64), nullable=True),
        sa.Column("acknowledged_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_plc_events_command_id", "plc_events", ["command_id"])
    op.create_index("ix_plc_events_inspection_id", "plc_events", ["inspection_id"])
    op.create_index("ix_plc_events_product_id", "plc_events", ["product_id"])

    for col in ("industrial_state", "industrial_final_state", "plc_command", "plc_status", "mes_sync_status"):
        op.add_column("inspections", sa.Column(col, sa.String(length=32), nullable=True))
    op.add_column("inspections", sa.Column("plc_latency_ms", sa.Float(), nullable=True))
    op.add_column("inspections", sa.Column("desired_command", sa.String(length=16), nullable=True))
    op.add_column("inspections", sa.Column("execution_status", sa.String(length=32), nullable=True))
    op.add_column("inspections", sa.Column("plc_adapter_type", sa.String(length=16), nullable=True))
    op.add_column("inspections", sa.Column("plc_reason_code", sa.String(length=64), nullable=True))


def downgrade() -> None:
    op.drop_column("inspections", "plc_reason_code")
    op.drop_column("inspections", "plc_adapter_type")
    op.drop_column("inspections", "execution_status")
    op.drop_column("inspections", "desired_command")
    op.drop_column("inspections", "plc_latency_ms")
    for col in ("mes_sync_status", "plc_status", "plc_command", "industrial_final_state", "industrial_state"):
        op.drop_column("inspections", col)
    op.drop_table("plc_events")
