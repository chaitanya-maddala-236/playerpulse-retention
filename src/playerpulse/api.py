from __future__ import annotations

import asyncio
import json
import logging
import secrets
import time
import uuid
from collections import Counter, deque
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from alembic import command
from alembic.config import Config
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from playerpulse.config import PROJECT_ROOT, Settings, get_settings
from playerpulse.database import make_engine, make_session_factory
from playerpulse.db_models import PredictionRecord
from playerpulse.monitoring import activity_segment, feature_psi, risk_band
from playerpulse.schemas import BatchPredictionInput, OutcomeInput, PredictionInput
from playerpulse.security import display_player_key, pseudonymize_player_reference
from playerpulse.synthetic_data import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    feature_frame,
    generate_dataset,
)

logger = logging.getLogger("playerpulse.api")
MAX_BODY_BYTES = 1_048_576
MAX_RECENT_LATENCIES = 5_000
PREDICTION_TIMEOUT_SECONDS = 20


def load_model_artifact(model_dir: Path) -> dict:
    artifact_path = model_dir / "model.joblib"
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Expected a trusted model artifact at {artifact_path}.")
    artifact = joblib.load(artifact_path)
    required = {"pipeline", "model_version", "features", "threshold", "metrics", "drift_reference"}
    if not isinstance(artifact, dict) or required - set(artifact):
        raise ValueError(
            "The configured model artifact does not match the expected PlayerPulse schema."
        )
    if tuple(artifact["features"]) != FEATURE_COLUMNS:
        raise ValueError("The configured model artifact feature schema does not match the API.")
    if not hasattr(artifact["pipeline"], "predict_proba"):
        raise ValueError("The configured model artifact has no probability prediction method.")
    if not 0 <= float(artifact["threshold"]) <= 1:
        raise ValueError("The configured model threshold must be between zero and one.")
    return artifact


def _migrate_database(engine) -> None:
    alembic_config = Config(str(PROJECT_ROOT / "alembic.ini"))
    alembic_config.set_main_option("script_location", str(PROJECT_ROOT / "migrations"))
    with engine.begin() as connection:
        alembic_config.attributes["connection"] = connection
        command.upgrade(alembic_config, "head")


def _require_secure_production_settings(settings: Settings) -> None:
    if settings.environment != "production":
        return
    if len(settings.prediction_api_key) < 32:
        raise ValueError("Production requires a prediction API key with at least 32 characters.")
    if (
        len(settings.player_id_hash_secret) < 32
        or settings.player_id_hash_secret == "local-demo-hash-key-change-before-deploy"
    ):
        raise ValueError("Production requires a unique player identifier hashing secret.")


def _iso_datetime(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat()


def _predict_with_explanations(artifact: dict, payloads: list[PredictionInput]) -> list[dict]:
    values = pd.DataFrame(
        [payload.feature_values() for payload in payloads], columns=FEATURE_COLUMNS
    )
    probabilities = artifact["pipeline"].predict_proba(values)[:, 1]
    medians = artifact.get("training_feature_medians", {name: 0.0 for name in FEATURE_COLUMNS})
    perturbed_rows = []
    explanations: list[list[float]] = [[] for _ in payloads]
    for row_index in range(len(payloads)):
        for feature in FEATURE_COLUMNS:
            perturbed = values.iloc[row_index].copy()
            perturbed[feature] = medians[feature]
            perturbed_rows.append(perturbed)
    if perturbed_rows:
        perturbed_probabilities = artifact["pipeline"].predict_proba(
            pd.DataFrame(perturbed_rows, columns=FEATURE_COLUMNS)
        )[:, 1]
        for row_index, probability in enumerate(probabilities):
            start = row_index * len(FEATURE_COLUMNS)
            explanations[row_index] = [
                float(probability - perturbed_probabilities[start + feature_index])
                for feature_index in range(len(FEATURE_COLUMNS))
            ]
    return [
        {
            "probability": float(probability),
            "risk_band": risk_band(float(probability)),
            "feature_explanations": dict(zip(FEATURE_COLUMNS, explanations[index], strict=True)),
        }
        for index, probability in enumerate(probabilities)
    ]


def _record_prediction(
    db: Session,
    payload: PredictionInput,
    result: dict,
    artifact: dict,
    hash_secret: str,
    source: str,
    label: bool | None = None,
    prediction_id: str | None = None,
) -> dict:
    prediction_id = prediction_id or str(uuid.uuid4())
    player_reference = payload.player_ref or f"unlinked:{prediction_id}"
    player_hash = pseudonymize_player_reference(player_reference, hash_secret)
    record = PredictionRecord(
        prediction_id=prediction_id,
        player_hash=player_hash,
        model_version=artifact["model_version"],
        probability=result["probability"],
        risk_band=result["risk_band"],
        feature_values=payload.feature_values(),
        feature_explanations=result["feature_explanations"],
        source=source,
        churned_label=label,
        created_at=datetime.now(UTC),
        latency_ms=0.0,
    )
    db.add(record)
    return {
        "prediction_id": prediction_id,
        "player_key": display_player_key(player_hash),
        "model_version": artifact["model_version"],
        "probability": result["probability"],
        "risk_band": result["risk_band"],
        "feature_explanations": result["feature_explanations"],
        "source": source,
        "predicted_at": _iso_datetime(record.created_at),
    }


def _seed_synthetic_demo(db: Session, artifact: dict, settings: Settings) -> None:
    if db.scalar(select(func.count()).select_from(PredictionRecord)):
        return
    dataset = generate_dataset(n_players=600, seed=2026)
    feature_rows = feature_frame(dataset)
    probabilities = artifact["pipeline"].predict_proba(feature_rows)[:, 1]
    created_records = []
    latest_prediction_at = datetime.now(UTC) - timedelta(days=15)
    demo_history_days = 70
    for index, row in dataset.iterrows():
        values = {feature: float(row[feature]) for feature in FEATURE_COLUMNS}
        probability = float(probabilities[index])
        player_hash = pseudonymize_player_reference(
            f"synthetic-demo:{row['player_id']}", settings.player_id_hash_secret
        )
        history_fraction = index / max(len(dataset) - 1, 1)
        prediction_at = latest_prediction_at - timedelta(
            days=demo_history_days * (1 - history_fraction)
        )
        created_records.append(
            PredictionRecord(
                prediction_id=str(uuid.uuid4()),
                player_hash=player_hash,
                model_version=artifact["model_version"],
                probability=probability,
                risk_band=risk_band(probability),
                feature_values=values,
                feature_explanations={},
                source="synthetic_demo",
                churned_label=bool(row[TARGET_COLUMN]),
                created_at=prediction_at,
                outcome_at=prediction_at + timedelta(days=14),
                latency_ms=0.0,
            )
        )
    db.add_all(created_records)
    db.commit()


def create_app(
    settings: Settings | None = None,
    artifact: dict | None = None,
    migrate_on_start: bool = True,
) -> FastAPI:
    app_settings = settings or get_settings()
    engine = make_engine(app_settings.database_url)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model_error = None
        app.state.database_error = None
        try:
            _require_secure_production_settings(app_settings)
        except ValueError as error:
            app.state.model_error = type(error).__name__
            logger.error(
                json.dumps({"event": "security_configuration_invalid", "error": str(error)})
            )
        if migrate_on_start:
            try:
                _migrate_database(engine)
            except Exception as error:
                app.state.database_error = type(error).__name__
                logger.exception(json.dumps({"event": "database_migration_failed"}))
        if artifact is not None:
            app.state.model_artifact = artifact
        else:
            try:
                app.state.model_artifact = load_model_artifact(app_settings.model_dir)
            except Exception as error:
                app.state.model_artifact = None
                app.state.model_error = type(error).__name__
                logger.exception(json.dumps({"event": "model_load_failed"}))
        if (
            app.state.database_error is None
            and app.state.model_artifact is not None
            and app_settings.demo_mode
        ):
            with app.state.session_factory() as db:
                try:
                    _seed_synthetic_demo(db, app.state.model_artifact, app_settings)
                except Exception as error:
                    db.rollback()
                    logger.exception(
                        json.dumps({"event": "demo_seed_failed", "error": type(error).__name__})
                    )
        yield
        engine.dispose()

    app = FastAPI(
        title="PlayerPulse API",
        description=(
            "Game-retention prediction and monitoring demo. The bundled model uses synthetic "
            "data and must not be interpreted as real-player evidence."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.settings = app_settings
    app.state.engine = engine
    app.state.session_factory = make_session_factory(engine)
    app.state.model_artifact = None
    app.state.model_error = None
    app.state.database_error = None
    app.state.telemetry = {
        "lock": asyncio.Lock(),
        "samples": deque(maxlen=MAX_RECENT_LATENCIES),
        "counts": Counter(),
        "missing_features": Counter(),
        "invalid_features": Counter(),
    }
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(app_settings.allowed_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-API-Key", "X-Request-ID"],
    )

    @app.exception_handler(RequestValidationError)
    async def sanitized_validation_error(request: Request, error: RequestValidationError):
        known_fields = set(FEATURE_COLUMNS) | {"player_ref", "predictions", "churned"}
        details = []
        telemetry = request.app.state.telemetry
        async with telemetry["lock"]:
            for issue in error.errors():
                location = issue.get("loc", ())
                field_name = location[-1] if location else "request"
                safe_field = field_name if field_name in known_fields else "other"
                if issue.get("type") == "missing" and safe_field in FEATURE_COLUMNS:
                    telemetry["missing_features"][safe_field] += 1
                elif safe_field in FEATURE_COLUMNS:
                    telemetry["invalid_features"][safe_field] += 1
                details.append(
                    {
                        "field": safe_field,
                        "reason": "Required value is missing."
                        if issue.get("type") == "missing"
                        else "Input value is invalid.",
                    }
                )
        return JSONResponse(
            status_code=422,
            content={"detail": details},
            headers={"X-Request-ID": getattr(request.state, "request_id", str(uuid.uuid4()))},
        )

    def get_db(request: Request):
        session = request.app.state.session_factory()
        try:
            yield session
        finally:
            session.close()

    def require_api_key(
        request: Request,
        api_key: str | None = Header(default=None, alias="X-API-Key"),
    ) -> None:
        expected = request.app.state.settings.prediction_api_key
        if expected and (api_key is None or not secrets.compare_digest(expected, api_key)):
            raise HTTPException(status_code=401, detail="A valid X-API-Key header is required.")

    @app.middleware("http")
    async def request_context_middleware(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", "")
        if (
            not request_id
            or len(request_id) > 80
            or not request_id.replace("-", "").replace("_", "").isalnum()
        ):
            request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        started_at = time.perf_counter()
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                if int(content_length) > MAX_BODY_BYTES:
                    return JSONResponse(
                        status_code=413,
                        content={"detail": "Request body exceeds the configured size limit."},
                        headers={"X-Request-ID": request_id},
                    )
            except ValueError:
                return JSONResponse(
                    status_code=400,
                    content={"detail": "Invalid Content-Length header."},
                    headers={"X-Request-ID": request_id},
                )
        response = None
        status_code = 500
        try:
            response = await asyncio.wait_for(
                call_next(request), timeout=PREDICTION_TIMEOUT_SECONDS
            )
            status_code = response.status_code
            return response
        except TimeoutError:
            response = JSONResponse(
                status_code=504,
                content={"detail": "Request exceeded the service time limit."},
            )
            status_code = 504
            return response
        finally:
            elapsed_ms = (time.perf_counter() - started_at) * 1_000
            telemetry = request.app.state.telemetry
            async with telemetry["lock"]:
                telemetry["samples"].append(
                    {"latency_ms": elapsed_ms, "status_code": status_code, "path": request.url.path}
                )
                telemetry["counts"]["requests"] += 1
                if status_code >= 400:
                    telemetry["counts"]["errors"] += 1
            logger.info(
                json.dumps(
                    {
                        "event": "http_request",
                        "request_id": request_id,
                        "method": request.method,
                        "path": request.url.path,
                        "status_code": status_code,
                        "latency_ms": round(elapsed_ms, 2),
                    },
                    separators=(",", ":"),
                )
            )
            if response is not None:
                response.headers["X-Request-ID"] = request_id

    @app.get("/health", tags=["health"])
    def health():
        return {"status": "ok"}

    @app.get("/ready", tags=["health"])
    def ready(request: Request):
        statuses = {"model": "ready", "database": "ready", "security": "ready"}
        if request.app.state.model_artifact is None or request.app.state.model_error:
            statuses["model"] = "unavailable"
        if request.app.state.database_error:
            statuses["database"] = "unavailable"
        else:
            try:
                with request.app.state.engine.connect() as connection:
                    connection.execute(select(1))
            except Exception:
                statuses["database"] = "unavailable"
        try:
            _require_secure_production_settings(request.app.state.settings)
        except ValueError:
            statuses["security"] = "misconfigured"
        status_code = 200 if set(statuses.values()) == {"ready"} else 503
        return JSONResponse(
            status_code=status_code,
            content={
                "status": "ready" if status_code == 200 else "not_ready",
                "components": statuses,
            },
        )

    @app.get("/v1/model/info", tags=["model"])
    def model_info(request: Request):
        loaded = request.app.state.model_artifact
        if loaded is None:
            raise HTTPException(status_code=503, detail="The model is not available.")
        metrics = loaded["metrics"]
        return {
            "model_version": loaded["model_version"],
            "features": loaded["features"],
            "threshold": loaded["threshold"],
            "selected_model": metrics["selected_model"],
            "evaluation": {
                **metrics["test"],
                "test_evaluated_at_utc": metrics["test_evaluated_at_utc"],
            },
            "validation": metrics["validation"],
            "candidates": metrics["candidates"],
            "feature_importance_validation": metrics["feature_importance_validation"],
            "training_split": metrics["split"],
            "dataset_is_synthetic": True,
            "warning": "Synthetic development data only; not evidence about real players.",
        }

    @app.post("/v1/predict", tags=["predictions"], dependencies=[Depends(require_api_key)])
    def predict(payload: PredictionInput, request: Request, db: Session = Depends(get_db)):
        artifact = request.app.state.model_artifact
        if artifact is None:
            raise HTTPException(status_code=503, detail="The model is not available.")
        started = time.perf_counter()
        result = _predict_with_explanations(artifact, [payload])[0]
        response = _record_prediction(
            db,
            payload,
            result,
            artifact,
            request.app.state.settings.player_id_hash_secret,
            source="api",
        )
        response["latency_ms"] = round((time.perf_counter() - started) * 1_000, 2)
        db.query(PredictionRecord).filter(
            PredictionRecord.prediction_id == response["prediction_id"]
        ).update({PredictionRecord.latency_ms: response["latency_ms"]})
        db.commit()
        return response

    @app.post("/v1/predict/batch", tags=["predictions"], dependencies=[Depends(require_api_key)])
    def predict_batch(
        payload: BatchPredictionInput, request: Request, db: Session = Depends(get_db)
    ):
        artifact = request.app.state.model_artifact
        if artifact is None:
            raise HTTPException(status_code=503, detail="The model is not available.")
        started = time.perf_counter()
        results = _predict_with_explanations(artifact, payload.predictions)
        elapsed_ms = (time.perf_counter() - started) * 1_000
        predictions = [
            _record_prediction(
                db,
                item,
                result,
                artifact,
                request.app.state.settings.player_id_hash_secret,
                source="api",
            )
            for item, result in zip(payload.predictions, results, strict=True)
        ]
        per_record_latency = round(elapsed_ms / max(len(predictions), 1), 2)
        prediction_ids = [item["prediction_id"] for item in predictions]
        db.query(PredictionRecord).filter(
            PredictionRecord.prediction_id.in_(prediction_ids)
        ).update({PredictionRecord.latency_ms: per_record_latency}, synchronize_session=False)
        db.commit()
        return {
            "model_version": artifact["model_version"],
            "count": len(predictions),
            "batch_latency_ms": round(elapsed_ms, 2),
            "predictions": predictions,
        }

    @app.get("/v1/predictions", tags=["predictions"], dependencies=[Depends(require_api_key)])
    def list_predictions(
        limit: int = Query(default=100, ge=1, le=500),
        offset: int = Query(default=0, ge=0, le=1_000_000),
        db: Session = Depends(get_db),
    ):
        total = db.scalar(select(func.count()).select_from(PredictionRecord)) or 0
        rows = db.scalars(
            select(PredictionRecord)
            .order_by(PredictionRecord.created_at.desc())
            .offset(offset)
            .limit(limit)
        ).all()
        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "data_label": "Synthetic demo fixtures are explicitly labeled.",
            "predictions": [
                {
                    "prediction_id": row.prediction_id,
                    "player_key": display_player_key(row.player_hash),
                    "model_version": row.model_version,
                    "probability": row.probability,
                    "risk_band": row.risk_band,
                    "features": row.feature_values,
                    "feature_explanations": row.feature_explanations,
                    "source": row.source,
                    "churned_label": row.churned_label,
                    "created_at": _iso_datetime(row.created_at),
                    "latency_ms": row.latency_ms,
                }
                for row in rows
            ],
        }

    @app.post("/v1/outcomes", tags=["monitoring"], dependencies=[Depends(require_api_key)])
    def add_outcome(payload: OutcomeInput, request: Request, db: Session = Depends(get_db)):
        player_hash = pseudonymize_player_reference(
            payload.player_ref, request.app.state.settings.player_id_hash_secret
        )
        cutoff = datetime.now(UTC) - timedelta(days=14)
        matching = db.scalars(
            select(PredictionRecord).where(
                PredictionRecord.player_hash == player_hash,
                PredictionRecord.source == "api",
                PredictionRecord.churned_label.is_(None),
                PredictionRecord.created_at <= cutoff,
            )
        ).all()
        if not matching:
            raise HTTPException(
                status_code=404,
                detail="No unlabeled prediction with a complete 14-day outcome window was found.",
            )
        outcome_time = datetime.now(UTC)
        for row in matching:
            row.churned_label = payload.churned
            row.outcome_at = outcome_time
        db.commit()
        return {"updated_predictions": len(matching), "outcome_at": _iso_datetime(outcome_time)}

    @app.get(
        "/v1/metrics/summary",
        tags=["monitoring"],
        dependencies=[Depends(require_api_key)],
    )
    async def metrics_summary(
        request: Request,
        db: Session = Depends(get_db),
        window_days: int = Query(default=90, ge=7, le=90),
    ):
        artifact = request.app.state.model_artifact
        if artifact is None:
            raise HTTPException(status_code=503, detail="The model is not available.")
        earliest = datetime.now(UTC) - timedelta(days=window_days)
        total = (
            db.scalar(
                select(func.count())
                .select_from(PredictionRecord)
                .where(PredictionRecord.created_at >= earliest)
            )
            or 0
        )
        latest_times = (
            select(
                PredictionRecord.player_hash,
                func.max(PredictionRecord.created_at).label("latest_at"),
            )
            .where(PredictionRecord.created_at >= earliest)
            .group_by(PredictionRecord.player_hash)
            .subquery()
        )
        latest_join = PredictionRecord.player_hash == latest_times.c.player_hash
        latest_at_join = PredictionRecord.created_at == latest_times.c.latest_at
        distinct_players = db.scalar(select(func.count()).select_from(latest_times)) or 0
        high_risk_players = (
            db.scalar(
                select(func.count(func.distinct(PredictionRecord.player_hash)))
                .join(latest_times, latest_join & latest_at_join)
                .where(PredictionRecord.risk_band == "high")
            )
            or 0
        )
        risk_rows = db.execute(
            select(
                PredictionRecord.risk_band, func.count(func.distinct(PredictionRecord.player_hash))
            )
            .join(latest_times, latest_join & latest_at_join)
            .group_by(PredictionRecord.risk_band)
        ).all()
        source_rows = db.execute(
            select(PredictionRecord.source, func.count())
            .where(PredictionRecord.created_at >= earliest)
            .group_by(PredictionRecord.source)
        ).all()
        model_version_rows = db.execute(
            select(PredictionRecord.model_version, PredictionRecord.source, func.count())
            .where(PredictionRecord.created_at >= earliest)
            .group_by(PredictionRecord.model_version, PredictionRecord.source)
        ).all()
        outcome_rows = db.scalars(
            select(PredictionRecord)
            .where(PredictionRecord.created_at >= earliest)
            .where(PredictionRecord.churned_label.is_not(None))
            .order_by(PredictionRecord.created_at.desc())
            .limit(5_000)
        ).all()
        labeled_count = len(outcome_rows)
        observed_churn_rate = (
            float(np.mean([row.churned_label for row in outcome_rows])) if labeled_count else None
        )
        delayed_performance = None
        subgroup_values = {}
        if labeled_count:
            labels = np.asarray([row.churned_label for row in outcome_rows], dtype=int)
            probabilities = np.asarray([row.probability for row in outcome_rows], dtype=float)
            predictions = probabilities >= artifact["threshold"]
            true_positive = int(np.sum(predictions & (labels == 1)))
            false_positive = int(np.sum(predictions & (labels == 0)))
            false_negative = int(np.sum(~predictions & (labels == 1)))
            precision_value = (
                true_positive / (true_positive + false_positive)
                if true_positive + false_positive
                else 0.0
            )
            recall_value = (
                true_positive / (true_positive + false_negative)
                if true_positive + false_negative
                else 0.0
            )
            f1_value = (
                2 * precision_value * recall_value / (precision_value + recall_value)
                if precision_value + recall_value
                else 0.0
            )
            delayed_performance = {
                "sample_size": labeled_count,
                "precision": precision_value,
                "recall": recall_value,
                "f1": f1_value,
                "churn_prevalence": observed_churn_rate,
                "evaluation_threshold": artifact["threshold"],
                "source_counts": dict(Counter(row.source for row in outcome_rows)),
            }
            for index, row in enumerate(outcome_rows):
                segment = activity_segment(row.feature_values)
                summary = subgroup_values.setdefault(
                    segment,
                    {"indices": [], "sources": Counter()},
                )
                summary["indices"].append(index)
                summary["sources"][row.source] += 1
            for segment, summary in sorted(subgroup_values.items()):
                indices = summary["indices"]
                segment_labels = labels[indices]
                subgroup_values[segment] = {
                    "activity_segment": segment,
                    "sample_size": len(indices),
                    "churn_prevalence": float(segment_labels.mean()),
                    "mean_predicted_probability": float(probabilities[indices].mean()),
                    "source_counts": dict(summary["sources"]),
                }

        recent_rows = db.scalars(
            select(PredictionRecord)
            .where(PredictionRecord.created_at >= earliest)
            .order_by(PredictionRecord.created_at.desc())
            .limit(5_000)
        ).all()
        trend: dict[str, dict] = {}
        for row in recent_rows:
            date_key = row.created_at.date().isoformat()
            entry = trend.setdefault(
                date_key,
                {"date": date_key, "predictions": 0, "labeled": 0, "churned": 0},
            )
            entry["predictions"] += 1
            if row.churned_label is not None:
                entry["labeled"] += 1
                entry["churned"] += int(row.churned_label)
        risk_distribution = {band: int(count) for band, count in risk_rows}
        for band in ("low", "medium", "high"):
            risk_distribution.setdefault(band, 0)

        drift_results = {}
        if recent_rows:
            reference = artifact["drift_reference"]
            for feature in FEATURE_COLUMNS:
                observed_values = [float(row.feature_values[feature]) for row in recent_rows]
                drift_results[feature] = feature_psi(observed_values, reference[feature])
        telemetry = request.app.state.telemetry
        async with telemetry["lock"]:
            latency_samples = list(telemetry["samples"])
            request_count = telemetry["counts"]["requests"]
            error_count = telemetry["counts"]["errors"]
            missing_features = dict(telemetry["missing_features"])
            invalid_features = dict(telemetry["invalid_features"])
        latency_values = np.asarray(
            [
                sample["latency_ms"]
                for sample in latency_samples
                if sample["path"].startswith("/v1/")
            ],
            dtype=float,
        )
        latency = None
        if latency_values.size >= 10:
            latency = {
                "sample_size": int(latency_values.size),
                "p50_ms": float(np.percentile(latency_values, 50)),
                "p95_ms": float(np.percentile(latency_values, 95)),
                "p99_ms": float(np.percentile(latency_values, 99)),
            }
        return {
            "generated_at": datetime.now(UTC).isoformat(),
            "dataset_is_synthetic": True,
            "warning": "Synthetic demo and submitted matured labels are reported by source.",
            "model_version": artifact["model_version"],
            "prediction_count": int(total),
            "eligible_player_count": int(distinct_players),
            "high_risk_player_count": int(high_risk_players),
            "observed_churn_rate": observed_churn_rate,
            "matured_outcome_count": labeled_count,
            "risk_distribution": risk_distribution,
            "source_counts": {name: int(count) for name, count in source_rows},
            "model_versions": [
                {"model_version": model_version, "source": source, "prediction_count": int(count)}
                for model_version, source, count in model_version_rows
            ],
            "trend_90d": [trend[key] for key in sorted(trend)],
            "drift_psi": drift_results,
            "delayed_performance": delayed_performance,
            "activity_segments": list(subgroup_values.values()),
            "api": {
                "availability": "available",
                "request_count_since_start": request_count,
                "error_count_since_start": error_count,
                "error_rate_since_start": error_count / request_count if request_count else 0.0,
                "missing_feature_errors_since_start": missing_features,
                "invalid_feature_errors_since_start": invalid_features,
                "latency": latency,
            },
            "reporting_window_days": window_days,
            "labels": {
                "synthetic_demo": "simulated future events",
                "api": "matured outcomes only",
            },
        }

    return app


app = create_app()
