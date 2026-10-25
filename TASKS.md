# PlayerPulse build checklist

- [x] Inspect workspace, Python versions, Docker CLI/daemon, Git identity, dataset candidates, and provider credentials.
- [x] Preserve the unrelated GameGuard telemetry project and create a clean PlayerPulse churn repository.
- [x] Record data source exclusions, model definition, architecture, and deployment limits.
- [x] Add deterministic synthetic event generation, data validation, future inactivity labels, past-only features, and chronological splits.
- [x] Implement candidate training, validation-only selection, held-out evaluation, and artifact generation.
- [x] Implement FastAPI routes, API key checks, HMAC pseudonyms, Postgres/SQLite persistence, migrations, and operational metrics.
- [x] Add five API-backed dashboard pages with real API data and synthetic-demo labeling.
- [x] Add delayed outcome, drift review, monitoring visuals, and dashboard access controls.
- [x] Add targeted tests, Ruff config, Compose, Dockerfiles, health checks, CI, and Render blueprint.
- [x] Install the pinned Python stack and run training, tests, lint, Docker builds, API smoke tests, and load test where the daemon permits.
- [ ] Publish to the authenticated GitHub account and deploy if available authorization supports it.
- [ ] Record only verified repository, deployment, and measurement results.
