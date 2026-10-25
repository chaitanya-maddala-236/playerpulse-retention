# Deployment status

Last checked: 2026-10-26.

## Render and Neon

- The Render Blueprint `PlayerPulse` provisions two free web services in Oregon: `playerpulse-api` and `playerpulse-dashboard`.
- API: [https://playerpulse-api-ckg3.onrender.com](https://playerpulse-api-ckg3.onrender.com). The `/ready` endpoint returned ready, including model and database checks against the Neon Postgres database. The check ran against commit `8b8b409`.
- Dashboard: [https://playerpulse-dashboard.onrender.com](https://playerpulse-dashboard.onrender.com). The Streamlit health endpoint returned HTTP 200, and the configured access-key gate loaded and unlocked.
- The dashboard currently cannot reach the API: its configured API host resolves to `playerpulse-api-ckg3`, which failed DNS resolution from the dashboard service. The API is healthy on its public URL, but data-backed dashboard views may show “API unavailable.”
- A request to `/v1/predict` without `X-API-Key` returned HTTP 401 as expected. A valid authenticated prediction and resulting Neon persistence were not verified.
- Render's free services may spin down when idle, so the first request can take around 50 seconds or longer. No paid service plan is configured.

## Credentials and data

The pooled Neon `DATABASE_URL` and generated API, dashboard-access, and player-hashing secrets are configured in provider settings only. They are not stored in this repository. The deployed model and all evaluation data are synthetic; do not use the demo with sensitive player data.

## Follow-up

The remaining deployment issue is the dashboard's API base URL configuration in `render.yaml`. It needs a public API URL (or a correctly configured private-network hostname), followed by a new deploy and authenticated prediction/history smoke test. No such change or redeploy has been made as part of the README cleanup.
