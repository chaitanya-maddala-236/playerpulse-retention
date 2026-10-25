from datetime import UTC, datetime, timedelta

import pandas as pd
from fastapi.testclient import TestClient
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from playerpulse.api import create_app
from playerpulse.config import Settings
from playerpulse.database import Base
from playerpulse.db_models import PredictionRecord
from playerpulse.schemas import PredictionInput
from playerpulse.synthetic_data import FEATURE_COLUMNS


def demo_artifact():
    rows = pd.DataFrame(
        [
            [0, 0, 0.0, 0.0, 30],
            [1, 2, 0.5, 0.8, 3],
            [2, 6, 1.0, 2.0, 1],
            [5, 20, 3.0, 12.0, 0],
        ],
        columns=FEATURE_COLUMNS,
    )
    labels = [1, 1, 0, 0]
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("classifier", DummyClassifier(strategy="prior")),
        ]
    ).fit(rows, labels)
    curve = {"precision": [0.5, 1.0], "recall": [1.0, 0.0], "thresholds": [0.5]}
    evaluation = {
        "n_samples": 4,
        "n_churned": 2,
        "churn_prevalence": 0.5,
        "pr_auc": 0.5,
        "roc_auc": 0.5,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
        "brier_score": 0.25,
        "threshold": 0.5,
        "confusion_matrix": [[1, 1], [1, 1]],
        "precision_recall_curve": curve,
        "calibration": {"predicted_probability": [0.5], "observed_prevalence": [0.5]},
        "test_evaluated_at_utc": datetime.now(UTC).isoformat(),
    }
    return {
        "pipeline": pipeline,
        "model_version": "playerpulse-test-v1",
        "features": list(FEATURE_COLUMNS),
        "threshold": 0.5,
        "training_feature_medians": {name: float(rows[name].median()) for name in FEATURE_COLUMNS},
        "metrics": {
            "selected_model": "dummy_prior",
            "test_evaluated_at_utc": datetime.now(UTC).isoformat(),
            "test": evaluation,
            "validation": {**evaluation, "precision_recall_curve": curve},
            "candidates": [{"model": "dummy_prior", "validation": {"pr_auc": 0.5}}],
            "feature_importance_validation": [],
            "split": {"n_train": 2, "n_validation": 1, "n_test": 1},
        },
        "drift_reference": {name: {"edges": [], "proportions": [1.0]} for name in FEATURE_COLUMNS},
    }


def app_client(tmp_path):
    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'predictions.db'}",
        model_dir=tmp_path,
        prediction_api_key="test-api-key",
        player_id_hash_secret="test-only-hash-secret-long-enough",
        demo_mode=False,
        allowed_origins=("http://testserver",),
        environment="test",
    )
    app = create_app(settings=settings, artifact=demo_artifact(), migrate_on_start=False)
    Base.metadata.create_all(app.state.engine)
    return app


def valid_payload(player_ref="game-id-345"):
    return {
        "player_ref": player_ref,
        "sessions_7d": 2,
        "sessions_30d": 8,
        "playtime_7d_hours": 1.5,
        "playtime_30d_hours": 4.5,
        "days_since_last_session": 2,
    }


def test_health_and_readiness_include_model_and_database(tmp_path):
    app = app_client(tmp_path)
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        response = client.get("/ready")
        assert response.status_code == 200
        assert response.json()["components"] == {
            "model": "ready",
            "database": "ready",
            "security": "ready",
        }


def test_model_info_includes_evaluation_timestamp_for_dashboard(tmp_path):
    app = app_client(tmp_path)
    with TestClient(app) as client:
        response = client.get("/v1/model/info")

    assert response.status_code == 200
    assert response.json()["evaluation"]["test_evaluated_at_utc"]


def test_prediction_requires_key_and_persists_only_pseudonymous_id(tmp_path):
    app = app_client(tmp_path)
    with TestClient(app) as client:
        assert client.post("/v1/predict", json=valid_payload()).status_code == 401
        response = client.post(
            "/v1/predict", json=valid_payload(), headers={"X-API-Key": "test-api-key"}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["model_version"] == "playerpulse-test-v1"
        assert body["player_key"].startswith("pp_")
        assert "game-id-345" not in response.text
        assert set(body["feature_explanations"]) == set(FEATURE_COLUMNS)

        history = client.get("/v1/predictions", headers={"X-API-Key": "test-api-key"})
        assert history.status_code == 200
        assert history.json()["predictions"][0]["player_key"] == body["player_key"]
        assert history.json()["predictions"][0]["source"] == "api"
        assert "game-id-345" not in history.text


def test_batch_is_bounded_and_saves_each_prediction(tmp_path):
    app = app_client(tmp_path)
    headers = {"X-API-Key": "test-api-key"}
    with TestClient(app) as client:
        too_many = {"predictions": [valid_payload() for _ in range(101)]}
        assert client.post("/v1/predict/batch", json=too_many, headers=headers).status_code == 422
        response = client.post(
            "/v1/predict/batch",
            json={"predictions": [valid_payload("one"), valid_payload("two")]},
            headers=headers,
        )
        assert response.status_code == 200
        assert response.json()["count"] == 2
        assert client.get("/v1/predictions", headers=headers).json()["total"] == 2


def test_invalid_schema_is_clear_and_does_not_echo_unknown_sensitive_fields(tmp_path):
    app = app_client(tmp_path)
    invalid = valid_payload("private-player-reference")
    invalid.update({"sessions_30d": 1, "private_unknown_field": "do-not-return-this"})
    with TestClient(app) as client:
        response = client.post("/v1/predict", json=invalid, headers={"X-API-Key": "test-api-key"})
        assert response.status_code == 422
        assert "private-player-reference" not in response.text
        assert "do-not-return-this" not in response.text
        assert all("reason" in issue for issue in response.json()["detail"])


def test_metrics_are_empty_and_window_is_validated(tmp_path):
    app = app_client(tmp_path)
    with TestClient(app) as client:
        headers = {"X-API-Key": "test-api-key"}
        empty = client.get("/v1/metrics/summary?window_days=30", headers=headers)
        assert empty.status_code == 200
        assert empty.json()["prediction_count"] == 0
        assert empty.json()["observed_churn_rate"] is None
        assert empty.json()["reporting_window_days"] == 30
        assert client.get("/v1/metrics/summary?window_days=365", headers=headers).status_code == 422


def test_demo_cohort_populates_recent_reporting_window(tmp_path):
    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'demo.db'}",
        model_dir=tmp_path,
        prediction_api_key="test-api-key",
        player_id_hash_secret="test-only-hash-secret-long-enough",
        demo_mode=True,
        allowed_origins=("http://testserver",),
        environment="test",
    )
    app = create_app(settings=settings, artifact=demo_artifact(), migrate_on_start=False)
    Base.metadata.create_all(app.state.engine)
    with TestClient(app) as client:
        metrics = client.get(
            "/v1/metrics/summary?window_days=90", headers={"X-API-Key": "test-api-key"}
        ).json()

    assert metrics["prediction_count"] == 600
    assert metrics["eligible_player_count"] == 600
    assert metrics["matured_outcome_count"] == 600
    assert metrics["source_counts"] == {"synthetic_demo": 600}


def test_matured_outcome_updates_delayed_metrics(tmp_path):
    app = app_client(tmp_path)
    headers = {"X-API-Key": "test-api-key"}
    with TestClient(app) as client:
        prediction = client.post("/v1/predict", json=valid_payload(), headers=headers)
        assert prediction.status_code == 200
        premature = client.post(
            "/v1/outcomes",
            json={"player_ref": "game-id-345", "churned": True},
            headers=headers,
        )
        assert premature.status_code == 404

        with app.state.session_factory() as db:
            record = db.get(PredictionRecord, prediction.json()["prediction_id"])
            record.created_at = datetime.now(UTC) - timedelta(days=15)
            db.commit()

        outcome = client.post(
            "/v1/outcomes",
            json={"player_ref": "game-id-345", "churned": True},
            headers=headers,
        )
        assert outcome.status_code == 200
        assert outcome.json()["updated_predictions"] == 1

        metrics = client.get("/v1/metrics/summary", headers=headers).json()
        assert metrics["matured_outcome_count"] == 1
        assert metrics["observed_churn_rate"] == 1.0
        assert metrics["delayed_performance"]["sample_size"] == 1
        assert (
            client.post(
                "/v1/outcomes",
                json={"player_ref": "game-id-345", "churned": True},
                headers=headers,
            ).status_code
            == 404
        )


def test_pydantic_enforces_activity_window_consistency():
    assert PredictionInput.model_validate(
        {
            "sessions_7d": 0,
            "sessions_30d": 0,
            "playtime_7d_hours": 0,
            "playtime_30d_hours": 0,
            "days_since_last_session": 30,
        }
    )
    try:
        PredictionInput.model_validate(
            {
                "sessions_7d": 0,
                "sessions_30d": 0,
                "playtime_7d_hours": 0,
                "playtime_30d_hours": 0,
                "days_since_last_session": 10,
            }
        )
    except ValueError as error:
        assert "requires days_since_last_session=30" in str(error)
    else:
        raise AssertionError("Inconsistent activity windows should not validate.")
