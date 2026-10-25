from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from playerpulse.synthetic_data import FEATURE_COLUMNS


class PredictionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    player_ref: str | None = Field(default=None, min_length=1, max_length=200)
    sessions_7d: int = Field(ge=0, le=7_000)
    sessions_30d: int = Field(ge=0, le=30_000)
    playtime_7d_hours: float = Field(ge=0, le=10_000)
    playtime_30d_hours: float = Field(ge=0, le=50_000)
    days_since_last_session: int = Field(ge=0, le=30)

    @field_validator("player_ref")
    @classmethod
    def normalize_player_reference(cls, value: str | None) -> str | None:
        if value is None:
            return value
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("player_ref cannot be empty or whitespace.")
        return cleaned

    @field_validator("sessions_30d")
    @classmethod
    def enforce_session_window_order(cls, value: int, info):
        sessions_7d = info.data.get("sessions_7d")
        if sessions_7d is not None and value < sessions_7d:
            raise ValueError("sessions_30d must be at least sessions_7d.")
        return value

    @field_validator("playtime_30d_hours")
    @classmethod
    def enforce_playtime_window_order(cls, value: float, info):
        playtime_7d = info.data.get("playtime_7d_hours")
        if playtime_7d is not None and value < playtime_7d:
            raise ValueError("playtime_30d_hours must be at least playtime_7d_hours.")
        return value

    @model_validator(mode="after")
    def enforce_activity_consistency(self):
        if self.sessions_30d == 0:
            if self.days_since_last_session != 30:
                raise ValueError("No sessions in 30 days requires days_since_last_session=30.")
            if self.playtime_30d_hours > 0:
                raise ValueError("Positive 30-day playtime requires at least one session.")
        elif self.days_since_last_session == 30:
            raise ValueError("A player with a recent session must have activity age below 30 days.")
        if self.sessions_7d == 0 and self.playtime_7d_hours > 0:
            raise ValueError("Positive 7-day playtime requires at least one session.")
        if self.sessions_7d > 0 and self.days_since_last_session > 6:
            raise ValueError(
                "A session in the last 7 days requires an activity age no greater than 6."
            )
        return self

    def feature_values(self) -> dict[str, float]:
        values = self.model_dump(exclude={"player_ref"})
        return {name: float(values[name]) for name in FEATURE_COLUMNS}


class BatchPredictionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    predictions: list[PredictionInput] = Field(min_length=1, max_length=100)


class OutcomeInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    player_ref: str = Field(min_length=1, max_length=200)
    churned: bool

    @field_validator("player_ref")
    @classmethod
    def normalize_player_reference(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("player_ref cannot be empty or whitespace.")
        return cleaned
