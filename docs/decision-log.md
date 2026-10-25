# Decision log

| Date | Decision | Reason |
| --- | --- | --- |
| 2026-10-26 | Place churn prediction in `playerpulse-retention/`. | A separate folder in the workspace was receiving an unrelated GameGuard telemetry project and contained raw data; isolation preserves both tasks. |
| 2026-10-26 | Use synthetic session events until a real dataset and license are verified. | The IEEE Blade & Soul competition gates files behind group access, and a checked sample has a label unrelated to its features. |
| 2026-10-26 | Define churn as no event in a fully observed 14-day period after a 30-day feature window. | Creates a leakage-aware, reproducible future outcome label. |
| 2026-10-26 | Split one player snapshot by timestamp into chronological train/validation/test partitions. | Prevents player overlap and avoids future-to-past leakage. |
| 2026-10-26 | Compare logistic regression, random forest, and histogram gradient boosting, with a dummy and weighting controls. | Measures the class-weighting tradeoff without assuming resampling improves results. |
| 2026-10-26 | Explain individual predictions with single-feature median perturbations. | Provides model-agnostic local comparisons without making causal claims. |
| 2026-10-26 | Use SQLite locally and Postgres in Compose and deployment. | Keeps developer setup simple and hosted prediction history persistent. |
| 2026-10-26 | Offer Render Blueprint free demo resources with explicit expiration limits. | API, dashboard, and managed Postgres fit a simple deploy path; this is not durable production hosting. |
| 2026-10-26 | Create a private GitHub repo by default if account auth works. | The user asked for GitHub storage; the private default avoids disclosing source without an explicit visibility choice. |
