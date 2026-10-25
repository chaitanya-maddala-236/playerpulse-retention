# PlayerPulse

**Retention-risk analytics for game teams — from session events to follow-up.**

PlayerPulse is an end-to-end portfolio demo for exploring how a game team could spot players showing signs of disengagement, prioritize retention outreach, and measure what happens afterward. It combines a leakage-aware ML workflow, a protected prediction API, persistent prediction history, and an API-backed dashboard.

> **Important:** The included dataset and every reported model score are synthetic. They demonstrate the engineering workflow, not real player behavior, business impact, or production readiness.

## Live demo

- **[PlayerPulse dashboard](https://playerpulse-dashboard.onrender.com)** — protected by the Render `DASHBOARD_ACCESS_KEY` environment secret; the key is not published here.
- **[API readiness](https://playerpulse-api-ckg3.onrender.com/ready)** · [OpenAPI docs](https://playerpulse-api-ckg3.onrender.com/docs)

**Current status (October 26, 2026):** The API readiness check returned ready, including its model and Neon database. The dashboard and its access-key screen respond, but the dashboard currently cannot resolve the API host configured for it, so data-backed pages may show **API unavailable**. A complete dashboard-to-API prediction and persistence flow has not been verified. Render free services may also spin down between visits and take around 50 seconds or longer to wake.

## The business problem

Retention teams need to decide **who may be disengaging, when to act, and whether an intervention helped**. Aggregate churn reports arrive late, while ad hoc player lists can be noisy and difficult to evaluate. A useful workflow should turn recent activity into an explainable risk signal, focus limited outreach, then attach outcomes after enough time has passed.

PlayerPulse demonstrates that workflow end to end: it builds features only from activity before a prediction timestamp, defines churn from a future inactivity window, scores player snapshots, records pseudonymous history, and supports delayed outcome and drift review. **Because the data is synthetic, this project does not establish that the model can identify real at-risk players or improve retention.**

## Results at a glance

The checked-in report evaluates a balanced logistic-regression model on a chronological synthetic cohort. The operating threshold was selected on validation data, then applied to the held-out test partition.

| Measure | Result |
| --- | ---: |
| Synthetic players | **3,000** |
| Chronological split | **2,100 train / 450 validation / 450 test** |
| Prediction target | Inactivity in the following **14 days** |
| Test PR-AUC | **0.580** |
| Test ROC-AUC | **0.879** |
| Test F1 | **0.654** at threshold **0.428** |
| Test Brier score | **0.154** |

These are measurements on generated data only. They are not evidence of real-world predictive performance, retention lift, or revenue impact. See the [experiment report](docs/experiment-report.md) and checked-in [metrics](artifacts/models/playerpulse-synthetic-v1/metrics.json) for details.

A small local API load smoke test also completed **100 requests at concurrency 5 with 0 errors**, at **28.16 requests/second** (p95 **425 ms**) on an 8-logical-CPU Windows host. It is a single-machine check, not a capacity benchmark; see the [load-test script](scripts/load_test.py).

## What it includes

- **Leakage-aware ML:** event validation, past-only 7-/30-day features, a future 14-day inactivity label, chronological splits, candidate comparison, validation-only model/threshold selection, and held-out evaluation.
- **FastAPI service:** health and readiness checks, single and bounded batch scoring, model metadata, paginated history, monitoring summaries, and delayed outcome capture.
- **Privacy-conscious identifiers:** keyed HMAC pseudonyms for player references; raw player IDs are neither stored nor returned.
- **Monitoring:** latency and prediction history, feature drift review, and matured-label summaries; drift alerts never promote a model automatically.
- **Streamlit dashboard:** Retention Overview, Player Risk Explorer, Model Performance, Model Monitoring, and Data and Model Information.
- **Persistence and operations:** PostgreSQL in cloud deployment, SQLite for Python-only local development, Alembic migrations, Docker Compose, health checks, and GitHub Actions CI.

## Architecture

```text
Session events ──> validation ──> past-only features ──> chronological evaluation
                                                        │
                                                        v
Dashboard ──> FastAPI + API key ──> model scoring ──> pseudonymous Postgres history
      ^                                  │                      │
      └──────── history, monitoring, and matured outcomes ──────┘
```

See the [architecture](docs/architecture.md), [dataset card](docs/dataset-card.md), [model card](docs/model-card.md), [experiment report](docs/experiment-report.md), and [deployment status](docs/deployment-status.md).

## Run locally

Requirements: Python 3.11+, Git, and Docker Desktop with Compose.

```powershell
Copy-Item .env.example .env
docker compose up --build
```

Open the dashboard at `http://localhost:8501`, the API at `http://localhost:8000`, and interactive API docs at `http://localhost:8000/docs`. Compose starts a local Postgres volume and clearly labeled synthetic demo predictions when the database is empty. Local dashboard access has no sign-in by default; set `DASHBOARD_ACCESS_KEY` before exposing it outside your machine.

For Python-only development, install the pinned environment and train the synthetic model:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt
python -m playerpulse.train --output-dir artifacts/models/playerpulse-synthetic-v1
alembic upgrade head
uvicorn playerpulse.api:app --reload
```

In another terminal, set `API_BASE_URL=http://127.0.0.1:8000`, then run `streamlit run dashboard/app.py`. Without Docker, the API uses local SQLite by default. The values in `.env.example` are local demo defaults; replace them before any deployment and never commit real secrets.

## API overview

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Process health |
| `GET` | `/ready` | Model, database, and production-security readiness |
| `GET` | `/v1/model/info` | Active model and evaluation metadata |
| `POST` | `/v1/predict` | Score one player snapshot |
| `POST` | `/v1/predict/batch` | Score 1–100 snapshots |
| `GET` | `/v1/predictions` | Pseudonymous, paginated prediction history |
| `GET` | `/v1/metrics/summary` | History, latency, drift, and matured-label summary |
| `POST` | `/v1/outcomes` | Attach a label after the 14-day window matures |

Prediction, history, monitoring, and outcome routes require `X-API-Key` when `PREDICTION_API_KEY` is configured. Batch inference and payload size are bounded; invalid player IDs and unknown features are rejected.

## Data and evaluation

The deterministic generator creates synthetic session events and labels a player as churned based on a future 14-day inactivity window. Features use only activity before the prediction timestamp. Training compares a dummy baseline, logistic regression, class-weighted logistic regression, random forest variants, and histogram gradient boosting. Validation PR-AUC selects the model and validation F1 selects the operating threshold; the later chronological test partition is reserved for evaluation.

No independently verifiable public game-telemetry dataset with an established reuse license was available for this build. The IEEE Game Data Mining Competition describes commercial-game logs, but its raw-data links are group-gated. An inspected 300-player sample had a churn flag independently drawn from its activity fields and was excluded. Full provenance and known limitations are in the [dataset card](docs/dataset-card.md).

## Deploy

`render.yaml` describes the API and dashboard services on Render; the deployed API uses Neon Postgres. Cloud credentials are configured in the provider and are not stored in this repository. The current dashboard-to-API connectivity issue is recorded in [deployment status](docs/deployment-status.md). The local Docker stack uses its own Postgres container.

Run the checks locally with:

```powershell
python -m pytest
ruff check .
ruff format --check .
docker compose up --build
```

## Limitations and responsible use

- The hand-designed generator is not representative of commercial game telemetry; there is no licensed production dataset or demonstrated predictive value for actual players.
- No protected characteristics are inferred or included, and subgroup performance is not established.
- Explanations are perturbation associations, not causal effects. A score is an estimate, not an intervention recommendation.
- Drift alerts call for investigation; they do not automatically promote or replace a model.
- Do not store sensitive player data on free demo hosting. Validate privacy, security, calibration, fairness, and intervention outcomes before considering real-world use.
