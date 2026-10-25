"""Initial persisted prediction history schema."""

import sqlalchemy as sa
from alembic import op

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "predictions",
        sa.Column("prediction_id", sa.String(length=36), nullable=False),
        sa.Column("player_hash", sa.String(length=64), nullable=False),
        sa.Column("model_version", sa.String(length=100), nullable=False),
        sa.Column("probability", sa.Float(), nullable=False),
        sa.Column("risk_band", sa.String(length=16), nullable=False),
        sa.Column("feature_values", sa.JSON(), nullable=False),
        sa.Column("feature_explanations", sa.JSON(), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("churned_label", sa.Boolean(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("outcome_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("latency_ms", sa.Float(), nullable=False),
        sa.PrimaryKeyConstraint("prediction_id"),
    )
    op.create_index("ix_predictions_created_at", "predictions", ["created_at"])
    op.create_index("ix_predictions_model_version", "predictions", ["model_version"])
    op.create_index("ix_predictions_player_hash", "predictions", ["player_hash"])
    op.create_index("ix_predictions_risk_band", "predictions", ["risk_band"])
    op.create_index("ix_predictions_source_label", "predictions", ["source", "churned_label"])


def downgrade() -> None:
    op.drop_table("predictions")
