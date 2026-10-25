import pandas as pd
import pytest

from playerpulse.synthetic_data import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    feature_frame,
    generate_dataset,
    split_chronologically,
    validate_dataset,
)


def test_generator_is_deterministic_and_uses_past_only_features():
    first = generate_dataset(n_players=240, seed=17)
    second = generate_dataset(n_players=240, seed=17)

    pd.testing.assert_frame_equal(first, second)
    assert tuple(feature_frame(first).columns) == FEATURE_COLUMNS
    assert not {"player_id", "prediction_at", TARGET_COLUMN}.intersection(FEATURE_COLUMNS)
    assert first[TARGET_COLUMN].nunique() == 2


def test_empty_dataset_and_missing_columns_are_rejected():
    dataset = generate_dataset(n_players=120, seed=18)

    with pytest.raises(ValueError, match="at least one row"):
        validate_dataset(dataset.iloc[:0])
    with pytest.raises(ValueError, match="Missing required"):
        validate_dataset(dataset.drop(columns=["sessions_7d"]))


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("sessions_7d", 99, "cannot exceed 30-day counts"),
        ("playtime_7d_hours", -1, "non-negative"),
        ("churned", 3, "binary"),
        ("prediction_at", "not-a-date", "timestamps must be valid"),
    ],
)
def test_dataset_validation_rejects_invalid_values(column, value, message):
    dataset = generate_dataset(n_players=120, seed=21)
    if column == "prediction_at":
        dataset[column] = dataset[column].astype(object)
    dataset.loc[0, column] = value

    with pytest.raises(ValueError, match=message):
        validate_dataset(dataset)


def test_chronological_splits_are_disjoint_and_ordered():
    dataset = generate_dataset(n_players=300, seed=11)
    training, validation, test = split_chronologically(dataset)

    assert training["prediction_at"].max() <= validation["prediction_at"].min()
    assert validation["prediction_at"].max() <= test["prediction_at"].min()
    assert set(training["player_id"]).isdisjoint(validation["player_id"])
    assert set(training["player_id"]).isdisjoint(test["player_id"])
    assert set(validation["player_id"]).isdisjoint(test["player_id"])


def test_duplicate_player_is_rejected():
    dataset = generate_dataset(n_players=120, seed=8)
    dataset.loc[1, "player_id"] = dataset.loc[0, "player_id"]

    with pytest.raises(ValueError, match="unique"):
        validate_dataset(dataset)
