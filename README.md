# PlayerPulse

PlayerPulse is a portfolio demo for game-retention analysis: leakage-aware training, model explanations, a FastAPI prediction service, a Streamlit dashboard, persistent history, and drift and outcome monitoring.

**Data status:** no independently verifiable public game-telemetry file with an established reuse license was available in this build. The built-in data and model metrics are synthetic. They are labeled throughout the product and are not evidence of real player behavior or business outcomes.

## Architecture

```text
synthetic session events → validation and past-only snapshot features → chronological splits
                                                              → model and measured demo report
Streamlit dashboard → API key protected FastAPI → PostgreSQL prediction history / monitoring
```

See [architecture](docs/architecture.md), [dataset card](docs/dataset-card.md), [experiment report](docs/experiment-report.md), [model card](docs/model-card.md), [implementation plan](docs/implementation-plan.md), and [deployment status](docs/deployment-status.md).

## Local setup

Requirements: Python 3.11+, Docker Desktop with Compose, and Git. Direct Python dependencies are pinned in `pyproject.toml`; `requirements-lock.txt` records the resolved Python 3.11 development environment.

```powershell
Copy-Item .env.example .env
docker compose up --build
```

Open the dashboard at `http://localhost:8501`; the API is at `http://localhost:8000`, with interactive OpenAPI docs at `/docs`. Compose starts a persistent local Postgres volume and creates clearly labeled synthetic demo predictions if the database is empty. Local dashboard access has no sign-in by default; set `DASHBOARD_ACCESS_KEY` before exposing the dashboard outside your machine.

For Python-only development, install dependencies, train the artifact, and start the API and dashboard in separate terminals:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt
python -m playerpulse.train --output-dir artifacts/models/playerpulse-synthetic-v1
alembic upgrade head
uvicorn playerpulse.api:app --reload
```

In another terminal, set `API_BASE_URL=http://127.0.0.1:8000`, then run `streamlit run dashboard/app.py`. Without Docker, the API uses local SQLite by default.

## Data and training

The IEEE Game Data Mining Competition describes real commercial-game logs, but its raw-data links are group-gated; this environment has no access or verified reuse license. An inspected 300-player sample had a churn flag independently drawn from its activity fields and cannot support model development. See [dataset card](docs/dataset-card.md) for provenance and exclusions.

The deterministic synthetic generator creates player session events and derives the label from the future 14-day inactivity window. Features use only days before the prediction timestamp.

```powershell
python -m playerpulse.train --output-dir artifacts/models/playerpulse-synthetic-v1
```

The script writes one fitted preprocessing and model pipeline, test and validation metrics, model metadata, and training-reference distributions. It compares a dummy baseline, logistic regression, class-weighted logistic regression, random forest, class-weighted random forest, and histogram gradient boosting. Validation PR-AUC selects the model and validation F1 selects the threshold. A later chronological test partition is used only for final evaluation. Results are simulation-only.

## API

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Process health |
| `GET` | `/ready` | Model, database, and production security readiness |
| `GET` | `/v1/model/info` | Active model and evaluation metadata |
| `POST` | `/v1/predict` | Score one snapshot |
| `POST` | `/v1/predict/batch` | Score 1–100 snapshots |
| `GET` | `/v1/predictions` | Pseudonymous, paginated prediction history |
| `GET` | `/v1/metrics/summary` | History, latency, drift, and matured-label summary |
| `POST` | `/v1/outcomes` | Attach a label after the 14-day window matures |

Prediction, history, monitoring, and outcome routes require `X-API-Key` whenever `PREDICTION_API_KEY` is configured. Raw player references are not stored or returned; matching uses a keyed HMAC pseudonym. Batch inference and payload size are bounded; player IDs and unknown features are rejected.

## Dashboard

The dashboard includes Retention Overview, Player Risk Explorer, Model Performance, Model Monitoring, and Data and Model Information. It fetches prediction, history, and monitoring data from the API; it does not independently load the model. It shows explicit synthetic-data, error, and empty states and exposes only pseudonymous CSV downloads.

## Docker, CI, and cloud

```powershell
docker compose up --build
```

Compose runs API, dashboard, and Postgres, with service health checks and a persistent database volume. `.github/workflows/ci.yml` runs Ruff, pytest, and Docker image builds.

`render.yaml` defines a Render Blueprint for the API and dashboard. The API uses Neon Postgres: create a Neon project, copy its pooled connection string (including TLS settings), and provide it as `DATABASE_URL` when creating the Blueprint. Keep this value private and out of Git. Render generates the API, pseudonymization, and dashboard-access keys; the dashboard receives the shared API key but not the player-ID hashing secret. The local Compose stack uses its own Postgres container. No cloud resource or verified live URL is claimed until deployment is smoke-tested.

## Evaluation and load test

Run training to generate `artifacts/models/playerpulse-synthetic-v1/metrics.json`; these are real measurements of the synthetic held-out cohort, not game telemetry. No live deployment or prior benchmark is claimed.

```powershell
python scripts/load_test.py --url http://localhost:8000 --requests 100 --concurrency 5
```

The load-test report includes request count, concurrency, host, throughput, p50/p95/p99 latency, error rate, and available memory. Results apply only to the machine and model printed in that run. A local 100-request, concurrency-5 smoke run completed with 0 errors at 28.16 requests/second (p50 131 ms, p95 425 ms, p99 530 ms) on an 8-logical-CPU Windows host; process memory was unavailable because `psutil` was not installed. This is a small local check, not a capacity claim.

## Limitations

- The generator’s hand-chosen session process is not representative of real game telemetry.
- There is no validation on a licensed production dataset or evidence of predictive value for actual players.
- Subgroup performance by protected characteristics is unsupported; none are inferred or included.
- Explanations are perturbation associations, not causal effects; predictions are estimates, not intervention recommendations.
- Drift alerts prompt investigation; they never automatically promote or replace a model.
- Do not store sensitive player data on free demo hosting.
