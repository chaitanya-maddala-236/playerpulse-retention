from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, Index, String
from sqlalchemy.orm import Mapped, mapped_column

from playerpulse.database import Base


class PredictionRecord(Base):
    __tablename__ = "predictions"
    __table_args__ = (
        Index("ix_predictions_created_at", "created_at"),
        Index("ix_predictions_source_label", "source", "churned_label"),
    )

    prediction_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    player_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    model_version: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    probability: Mapped[float] = mapped_column(Float, nullable=False)
    risk_band: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    feature_values: Mapped[dict] = mapped_column(JSON, nullable=False)
    feature_explanations: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    source: Mapped[str] = mapped_column(String(32), nullable=False, default="api")
    churned_label: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    outcome_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
