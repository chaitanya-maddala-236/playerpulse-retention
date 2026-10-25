# Implementation plan

1. Inspect runtime, workspace, dataset provenance, and provider access.
2. Record architecture, limitations, data contract, and hosting assumptions.
3. Implement a synthetic fallback with future inactivity labels, feature-window validation, and chronological player-disjoint partitions.
4. Train and compare a dummy classifier, logistic regression variants, random forest variants, and a tree-boosting alternative.
5. Select a model and classification threshold using validation only; evaluate on a later test window; package preprocessing and model together.
6. Implement the API, trusted artifact loading, authentication, HMAC player references, migrations, database history, and monitoring.
7. Build the API-backed overview, player explorer, model performance, monitoring, and data information pages.
8. Add edge-case tests, formatting/lint, Docker, CI, and a deployment blueprint.
9. Run checks, publish and deploy only with verified access, and record live endpoint results only after smoke tests.

Progress and the remaining work are tracked in `TASKS.md`. A suitable public real-game dataset was not acquired, so synthetic model measurements are clearly marked and do not claim real player behavior.
