# Deployment status

Last checked: 2026-10-26.

## Workspace and access

- A separate PlayerPulse churn repository was created at `playerpulse-retention/`; the workspace's unrelated GameGuard project and raw telemetry remain separate.
- Python 3.11 and 3.13 are installed. Docker CLI exists; the Docker Engine did not return a server version.
- Git Credential Manager lists account `chaitanya-maddala-236`. No GitHub CLI or Render API token is present in environment variables.
- Local Docker Compose API, dashboard, and Postgres containers passed their health checks. The API reports 600 explicitly synthetic demo predictions; the dashboard returned HTTP 200.
- No live hosted service, URL, secret, or Neon database has been provisioned or verified.

## Hosting path

`render.yaml` declares the API and dashboard and expects a Neon pooled Postgres connection string as `DATABASE_URL`. Render generates the API key, dashboard access key, and player pseudonym secret. Neon credentials are not stored in this repository. No cloud resource was provisioned.

## Verification record

Update this file only after repository publication, Neon database/API/dashboard deployment, live health checks, a valid and invalid prediction smoke test, and observed persistence. Record actual URLs and results; currently no live URL is verified.
