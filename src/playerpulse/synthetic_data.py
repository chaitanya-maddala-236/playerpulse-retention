from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = (
    "sessions_7d",
    "sessions_30d",
    "playtime_7d_hours",
    "playtime_30d_hours",
    "days_since_last_session",
)
TARGET_COLUMN = "churned"
IDENTIFIER_COLUMN = "player_id"
TIMESTAMP_COLUMN = "prediction_at"
GENERATOR_VERSION = "synthetic-events-v1"
BASE_DATE = pd.Timestamp("2025-01-01", tz="UTC")


def generate_dataset(n_players: int = 3_000, seed: int = 42) -> pd.DataFrame:
    if n_players < 100:
        raise ValueError("At least 100 synthetic players are required for chronological splits.")

    random = np.random.default_rng(seed)
    low_engagement = random.random(n_players) < 0.44
    daily_session_probability = np.where(
        low_engagement,
        random.uniform(0.008, 0.075, size=n_players),
        random.uniform(0.17, 0.43, size=n_players),
    )
    past_sessions = random.random((n_players, 30)) < daily_session_probability[:, None]
    future_sessions = random.random((n_players, 14)) < daily_session_probability[:, None]
    session_hours = random.gamma(shape=2.0, scale=0.38, size=(n_players, 30))
    observed_hours = session_hours * past_sessions

    sessions_30d = past_sessions.sum(axis=1).astype(int)
    sessions_7d = past_sessions[:, -7:].sum(axis=1).astype(int)
    playtime_30d = observed_hours.sum(axis=1)
    playtime_7d = observed_hours[:, -7:].sum(axis=1)
    last_session_age = np.full(n_players, 30, dtype=int)
    has_session = sessions_30d > 0
    last_session_day = np.where(past_sessions, np.arange(30), -1).max(axis=1)
    last_session_age[has_session] = 29 - last_session_day[has_session]

    installed_day = random.integers(0, 550, size=n_players)
    install_timestamps = BASE_DATE + pd.to_timedelta(installed_day, unit="D")
    prediction_timestamps = install_timestamps + pd.Timedelta(days=30)
    churned = ~future_sessions.any(axis=1)

    dataset = (
        pd.DataFrame(
            {
                IDENTIFIER_COLUMN: [f"synthetic-player-{index:06d}" for index in range(n_players)],
                TIMESTAMP_COLUMN: prediction_timestamps,
                "sessions_7d": sessions_7d,
                "sessions_30d": sessions_30d,
                "playtime_7d_hours": np.round(playtime_7d, 4),
                "playtime_30d_hours": np.round(playtime_30d, 4),
                "days_since_last_session": last_session_age,
                TARGET_COLUMN: churned.astype(int),
            }
        )
        .sort_values(TIMESTAMP_COLUMN, kind="mergesort")
        .reset_index(drop=True)
    )
    validate_dataset(dataset)
    return dataset


def feature_frame(dataset: pd.DataFrame) -> pd.DataFrame:
    missing = set(FEATURE_COLUMNS) - set(dataset.columns)
    if missing:
        raise ValueError(f"Missing required feature columns: {sorted(missing)}")
    return dataset.loc[:, list(FEATURE_COLUMNS)].astype(float)


def validate_dataset(dataset: pd.DataFrame) -> None:
    required = set(FEATURE_COLUMNS) | {TARGET_COLUMN, IDENTIFIER_COLUMN, TIMESTAMP_COLUMN}
    missing = required - set(dataset.columns)
    if missing:
        raise ValueError(f"Missing required dataset columns: {sorted(missing)}")
    if dataset.empty:
        raise ValueError("Dataset must contain at least one row.")
    if dataset[IDENTIFIER_COLUMN].isna().any() or dataset[IDENTIFIER_COLUMN].duplicated().any():
        raise ValueError("Each dataset row must have a unique, non-empty player identifier.")
    parsed_timestamps = pd.to_datetime(
        dataset[TIMESTAMP_COLUMN], format="mixed", utc=True, errors="coerce"
    )
    if parsed_timestamps.isna().any():
        raise ValueError("Prediction timestamps must be valid and present.")
    features = feature_frame(dataset)
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError("Feature values must be finite and non-missing.")
    if (features.to_numpy() < 0).any():
        raise ValueError("Session, playtime, and activity gap features must be non-negative.")
    if (features["sessions_7d"] > features["sessions_30d"]).any():
        raise ValueError("Seven-day session counts cannot exceed 30-day counts.")
    if (features["playtime_7d_hours"] > features["playtime_30d_hours"] + 1e-6).any():
        raise ValueError("Seven-day playtime cannot exceed 30-day playtime.")
    if (features["days_since_last_session"] > 30).any():
        raise ValueError("Activity gaps cannot exceed the 30-day feature window.")
    if not dataset[TARGET_COLUMN].isin([0, 1]).all():
        raise ValueError("Churn labels must be binary 0/1 values.")


def split_chronologically(
    dataset: pd.DataFrame,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validate_dataset(dataset)
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("Split fractions must be between zero and one.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("A non-empty held-out test fraction is required.")

    ordered = dataset.sort_values(TIMESTAMP_COLUMN, kind="mergesort").reset_index(drop=True)
    training_end = int(len(ordered) * train_fraction)
    validation_end = int(len(ordered) * (train_fraction + validation_fraction))
    training = ordered.iloc[:training_end].copy()
    validation = ordered.iloc[training_end:validation_end].copy()
    test = ordered.iloc[validation_end:].copy()
    if any(part[TARGET_COLUMN].nunique() < 2 for part in (training, validation, test)):
        raise ValueError("Each chronological split must contain both churn classes.")
    player_sets = [set(part[IDENTIFIER_COLUMN]) for part in (training, validation, test)]
    if (
        player_sets[0] & player_sets[1]
        or player_sets[0] & player_sets[2]
        or player_sets[1] & player_sets[2]
    ):
        raise ValueError("A player appears in more than one split.")
    if training[TIMESTAMP_COLUMN].max() > validation[TIMESTAMP_COLUMN].min():
        raise ValueError("Training timestamps must precede validation timestamps.")
    if validation[TIMESTAMP_COLUMN].max() > test[TIMESTAMP_COLUMN].min():
        raise ValueError("Validation timestamps must precede test timestamps.")
    return training, validation, test
