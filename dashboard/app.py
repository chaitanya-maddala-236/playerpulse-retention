from __future__ import annotations

import hmac
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from api_client import (
    ApiUnavailable,
    get_health,
    get_metrics,
    get_model_info,
    get_predictions,
    post_prediction,
)

COLORS = {"low": "#71E5C0", "medium": "#FFB86B", "high": "#FF727A"}
FEATURES = (
    "sessions_7d",
    "sessions_30d",
    "playtime_7d_hours",
    "playtime_30d_hours",
    "days_since_last_session",
)

st.set_page_config(page_title="PlayerPulse · Retention Intelligence", page_icon="◈", layout="wide")
st.markdown(
    """
    <style>
      :root { color-scheme: dark; }
      .stApp { background: #101827; color: #F2F5FA; }
      [data-testid="stSidebar"] { background: #121D2E; border-right: 1px solid #29384C; }
      [data-testid="stHeader"] { background: rgba(16,24,39,0.88); }
      [data-testid="stMetric"] { background: #1A2536; border: 1px solid #2A384A; padding: 18px 20px; border-radius: 12px; }
      [data-testid="stMetricLabel"] { color: #A8B5C8; }
      h1, h2, h3 { color: #F2F5FA; letter-spacing: -0.02em; }
      .pp-kicker { color: #71E5C0; font-weight: 700; text-transform: uppercase; letter-spacing: .16em; font-size: .74rem; }
      .pp-banner { background: #182B34; border: 1px solid #35685F; color: #D9FFF2; padding: 12px 16px; border-radius: 10px; margin: .8rem 0 1.4rem; }
      div[data-testid="stDataFrame"] { border: 1px solid #29384C; border-radius: 10px; }
      .stButton > button, .stDownloadButton > button { border: 1px solid #3C786C; border-radius: 9px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def api_data(loader, *args):
    try:
        return loader(*args), None
    except ApiUnavailable as error:
        return None, str(error)


def percent(value):
    return "—" if value is None else f"{value:.1%}"


def header(title: str, subtitle: str):
    st.markdown(
        '<div class="pp-kicker">PlayerPulse · Game retention intelligence</div>',
        unsafe_allow_html=True,
    )
    st.title(title)
    st.caption(subtitle)
    st.markdown(
        '<div class="pp-banner"><strong>SYNTHETIC DEMO MODEL</strong> · The bundled model and seeded cohort use simulated sessions. Scores are estimates, not evidence about real players.</div>',
        unsafe_allow_html=True,
    )


def plot_layout(figure, height: int = 330):
    figure.update_layout(
        height=height,
        paper_bgcolor="#1A2536",
        plot_bgcolor="#1A2536",
        font_color="#F2F5FA",
        margin={"l": 18, "r": 18, "t": 34, "b": 20},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    figure.update_xaxes(gridcolor="#29384C", zerolinecolor="#29384C")
    figure.update_yaxes(gridcolor="#29384C", zerolinecolor="#29384C")
    return figure


def show_api_error(message: str):
    st.error(f"{message} Check that the API is ready and the dashboard key is configured.")


def retention_overview():
    header(
        "Retention overview",
        "Persisted player history and matured outcomes in a defined reporting window.",
    )
    window = st.selectbox(
        "Reporting window", [30, 60, 90], index=2, format_func=lambda days: f"Last {days} days"
    )
    metrics, error = api_data(get_metrics, window)
    if error:
        show_api_error(error)
        return
    st.caption(
        f"Reporting window: {metrics['reporting_window_days']} days · Sources: {metrics['source_counts']}"
    )
    columns = st.columns(4)
    columns[0].metric("Eligible players", f"{metrics['eligible_player_count']:,}")
    columns[1].metric(
        "Observed churn",
        percent(metrics["observed_churn_rate"]),
        help="Matured labels only; built-in demo outcomes are synthetic.",
    )
    columns[2].metric("High-risk players", f"{metrics['high_risk_player_count']:,}")
    columns[3].metric("Prediction records", f"{metrics['prediction_count']:,}")
    st.caption(
        f"Active model: `{metrics['model_version']}` · matured outcomes: {metrics['matured_outcome_count']:,}"
    )

    trend = pd.DataFrame(metrics["trend_90d"])
    risk = pd.DataFrame(
        [
            {"Risk band": key.title(), "Players": value}
            for key, value in metrics["risk_distribution"].items()
        ]
    )
    left, right = st.columns([1.5, 1])
    if trend.empty:
        left.info("No predictions were recorded in this window yet.")
    else:
        chart = px.bar(
            trend,
            x="date",
            y="predictions",
            title="Predictions by day",
            labels={"date": "Prediction date", "predictions": "Records"},
            color_discrete_sequence=["#71E5C0"],
        )
        left.plotly_chart(plot_layout(chart), use_container_width=True)
    chart = px.bar(
        risk,
        x="Risk band",
        y="Players",
        color="Risk band",
        color_discrete_map={key.title(): value for key, value in COLORS.items()},
        title="Risk distribution",
    )
    right.plotly_chart(plot_layout(chart), use_container_width=True)

    segments = pd.DataFrame(metrics["activity_segments"])
    if not segments.empty:
        st.subheader("Matured outcomes by activity segment")
        st.caption(
            "Synthetic and submitted outcomes are identified by source; subgroup counts provide context."
        )
        st.dataframe(
            segments[["activity_segment", "sample_size", "churn_prevalence", "source_counts"]],
            use_container_width=True,
            hide_index=True,
        )
        chart = px.bar(
            segments,
            x="activity_segment",
            y="churn_prevalence",
            text="sample_size",
            color="activity_segment",
            labels={"activity_segment": "Activity segment", "churn_prevalence": "Churn prevalence"},
        )
        chart.update_yaxes(tickformat=".0%")
        st.plotly_chart(plot_layout(chart, 300), use_container_width=True)
    if not metrics["prediction_count"]:
        st.info(
            "History is empty. Score a snapshot in Player Risk Explorer to create the first record."
        )


def player_risk_explorer():
    header(
        "Player risk explorer",
        "Search pseudonymous history, inspect feature summaries, and score a snapshot.",
    )
    filters = st.columns([2, 1])
    search = filters[0].text_input("Search pseudonymous player key", placeholder="pp_…")
    page_size = filters[1].selectbox("Rows per page", [25, 50, 100, 200], index=1)
    try:
        predictions, total = get_predictions(5_000)
    except ApiUnavailable as error:
        show_api_error(str(error))
        return
    if total > 5_000:
        st.caption(
            "The explorer shows the newest 5,000 records. Use the API offset to inspect older rows."
        )
    rows = [
        {
            "Player": record["player_key"],
            "Risk": record["risk_band"].title(),
            "Churn probability": record["probability"],
            "Sessions · 30d": record["features"]["sessions_30d"],
            "Days since session": record["features"]["days_since_last_session"],
            "Model": record["model_version"],
            "Source": record["source"].replace("_", " ").title(),
            "Created": record["created_at"],
        }
        for record in predictions
    ]
    table = pd.DataFrame(rows)
    if search and not table.empty:
        table = table[table["Player"].str.contains(search.strip(), case=False, regex=False)]
    if table.empty:
        st.info("No matching rows. Try a different pseudonymous key or score a snapshot below.")
    else:
        pages = max(1, (len(table) + page_size - 1) // page_size)
        page = st.number_input("Page", min_value=1, max_value=pages, value=1, step=1)
        start = (page - 1) * page_size
        shown = table.iloc[start : start + page_size].copy()
        shown["Created"] = pd.to_datetime(shown["Created"], utc=True)
        st.dataframe(
            shown,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Churn probability": st.column_config.ProgressColumn(
                    "Churn probability", min_value=0, max_value=1, format="%.1%%"
                ),
                "Created": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
            },
        )
        visible_keys = set(table["Player"])
        by_key = {
            row["player_key"]: row for row in predictions if row["player_key"] in visible_keys
        }
        key = st.selectbox("Player detail", list(by_key))
        record = by_key[key]
        columns = st.columns(3)
        columns[0].metric("Churn probability", percent(record["probability"]))
        columns[1].metric("Risk band", record["risk_band"].title())
        columns[2].metric("Model version", record["model_version"])
        st.caption(
            f"Pseudonymous key `{key}` · source: {record['source'].replace('_', ' ').title()} · {record['created_at']}"
        )
        feature, explanation = st.columns(2)
        feature.subheader("Snapshot activity")
        feature.dataframe(
            pd.DataFrame(
                [{"Feature": name, "Value": value} for name, value in record["features"].items()]
            ),
            use_container_width=True,
            hide_index=True,
        )
        explanation.subheader("Local feature explanation")
        contributions = record.get("feature_explanations", {})
        if contributions:
            contribution_frame = pd.DataFrame(
                [
                    {"Feature": name, "Probability change vs training median": value}
                    for name, value in contributions.items()
                ]
            ).sort_values("Probability change vs training median")
            chart = px.bar(
                contribution_frame,
                x="Probability change vs training median",
                y="Feature",
                orientation="h",
                color="Probability change vs training median",
                color_continuous_scale=["#71E5C0", "#A8B5C8", "#FF727A"],
            )
            explanation.plotly_chart(plot_layout(chart, 280), use_container_width=True)
            explanation.caption("Median perturbations show associations, not causal effects.")
        else:
            explanation.info("The synthetic seed stores scores but no per-player explanation.")
        csv = table.drop(columns=["Created"]).to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download filtered pseudonymous CSV", csv, "playerpulse_predictions.csv", "text/csv"
        )

    st.divider()
    st.subheader("Score a player snapshot")
    st.caption(
        "The API stores a keyed pseudonym, numeric features, and model version. Do not send personal or protected information."
    )
    with st.form("player_snapshot"):
        reference = st.text_input("Optional player reference", max_chars=200)
        first = st.columns(3)
        sessions_7d = first[0].number_input("Sessions · 7 days", 0, 7_000, 2)
        sessions_30d = first[1].number_input("Sessions · 30 days", 0, 30_000, 8)
        days_since = first[2].number_input("Days since last session", 0, 30, 2)
        second = st.columns(2)
        play_7d = second[0].number_input("Playtime · 7 days (hours)", 0.0, 10_000.0, 1.5, step=0.5)
        play_30d = second[1].number_input(
            "Playtime · 30 days (hours)", 0.0, 50_000.0, 4.5, step=0.5
        )
        submit = st.form_submit_button("Score snapshot", type="primary")
    if submit:
        payload = {
            "player_ref": reference.strip() or None,
            "sessions_7d": sessions_7d,
            "sessions_30d": sessions_30d,
            "playtime_7d_hours": play_7d,
            "playtime_30d_hours": play_30d,
            "days_since_last_session": days_since,
        }
        try:
            result = post_prediction(payload)
            st.success(
                f"Saved {result['risk_band']} risk estimate · {percent(result['probability'])} · `{result['player_key']}`"
            )
            st.caption("Synthetic training model. Treat this estimate as a demo only.")
            st.json(result["feature_explanations"])
        except ApiUnavailable as error:
            show_api_error(str(error))


def model_performance():
    header(
        "Model performance",
        "Candidate comparison, validation trade-offs, and final held-out test results.",
    )
    info, error = api_data(get_model_info)
    if error:
        show_api_error(error)
        return
    validation, evaluation = info["validation"], info["evaluation"]
    st.caption(
        f"Selected on validation PR-AUC: `{info['selected_model']}` · test n={evaluation['n_samples']:,} · prevalence {percent(evaluation['churn_prevalence'])}"
    )
    st.caption(
        f"Validation-selected threshold: {info['threshold']:.3f} · the test set did not tune it."
    )
    cards = st.columns(6)
    for cell, key in zip(
        cards,
        ("pr_auc", "roc_auc", "precision", "recall", "f1", "brier_score"),
        strict=True,
    ):
        cell.metric(key.upper().replace("_", " "), f"{evaluation[key]:.3f}")

    comparison = pd.DataFrame(
        [{"Model": item["model"], **item["validation"]} for item in info["candidates"]]
    ).sort_values("pr_auc", ascending=False)
    st.subheader("Validation candidate comparison")
    st.dataframe(comparison, use_container_width=True, hide_index=True)
    st.caption("Weighting variants are measured directly. No resampling is assumed to help.")
    left, right = st.columns(2)
    curve = validation["precision_recall_curve"]
    tradeoff = pd.DataFrame(
        {
            "threshold": curve["thresholds"],
            "precision": curve["precision"][:-1],
            "recall": curve["recall"][:-1],
        }
    )
    tradeoff_chart = px.line(
        tradeoff,
        x="threshold",
        y=["precision", "recall"],
        title="Threshold trade-offs · validation",
    )
    tradeoff_chart.add_vline(x=info["threshold"], line_dash="dash", line_color="#FFB86B")
    tradeoff_chart.update_yaxes(range=[0, 1])
    st.plotly_chart(plot_layout(tradeoff_chart), use_container_width=True)
    chart = go.Figure(
        go.Scatter(
            x=curve["recall"],
            y=curve["precision"],
            mode="lines",
            name="Validation",
            line={"color": "#71E5C0", "width": 3},
        )
    )
    chart.update_layout(
        title="Precision–recall · validation", xaxis_title="Recall", yaxis_title="Precision"
    )
    chart.update_xaxes(range=[0, 1])
    chart.update_yaxes(range=[0, 1])
    left.plotly_chart(plot_layout(chart), use_container_width=True)

    confusion = go.Figure(
        go.Heatmap(
            z=evaluation["confusion_matrix"],
            x=["Predicted active", "Predicted churn"],
            y=["Observed active", "Observed churn"],
            text=evaluation["confusion_matrix"],
            texttemplate="%{text}",
            colorscale=[[0, "#1A2536"], [1, "#71E5C0"]],
        )
    )
    confusion.update_layout(
        title=f"Held-out test confusion matrix · n={evaluation['n_samples']}",
        xaxis_title="Prediction",
        yaxis_title="Observed label",
    )
    right.plotly_chart(plot_layout(confusion), use_container_width=True)

    calibration = evaluation["calibration"]
    chart = go.Figure()
    chart.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line={"dash": "dash", "color": "#8291A7"},
            name="Perfect calibration",
        )
    )
    chart.add_trace(
        go.Scatter(
            x=calibration["predicted_probability"],
            y=calibration["observed_prevalence"],
            mode="lines+markers",
            marker={"color": "#FFB86B"},
            name="Held-out test",
        )
    )
    chart.update_layout(
        title="Calibration · held-out test",
        xaxis_title="Mean predicted probability",
        yaxis_title="Observed prevalence",
    )
    chart.update_xaxes(range=[0, 1])
    chart.update_yaxes(range=[0, 1])
    st.plotly_chart(plot_layout(chart), use_container_width=True)

    importance = pd.DataFrame(info["feature_importance_validation"])
    if not importance.empty:
        chart = px.bar(
            importance.sort_values("mean_pr_auc_decrease"),
            x="mean_pr_auc_decrease",
            y="feature",
            orientation="h",
            error_x="std_pr_auc_decrease",
            title="Permutation importance · validation",
            labels={"mean_pr_auc_decrease": "Validation PR-AUC decrease", "feature": "Feature"},
            color_discrete_sequence=["#71E5C0"],
        )
        st.plotly_chart(plot_layout(chart, 285), use_container_width=True)
    st.caption(
        "Local median perturbations are available for scored API predictions in Player Risk Explorer."
    )


def model_monitoring():
    header(
        "Model monitoring",
        "Service telemetry, feature drift, and outcomes after the 14-day maturity window.",
    )
    metrics, error = api_data(get_metrics, 90)
    if error:
        show_api_error(error)
        return
    try:
        health = get_health()
    except ApiUnavailable as problem:
        health = {"status": "unavailable"}
        st.warning(str(problem))
    api = metrics["api"]
    latency = api["latency"]
    cells = st.columns(4)
    cells[0].metric("API readiness", health.get("status", "unknown").title())
    cells[1].metric("Requests since start", f"{api['request_count_since_start']:,}")
    cells[2].metric("Error rate since start", percent(api["error_rate_since_start"]))
    cells[3].metric("p95 latency", "—" if latency is None else f"{latency['p95_ms']:.1f} ms")
    if latency is None:
        st.caption("p50/p95/p99 appear after at least 10 recent versioned API observations.")
    else:
        percentile_cells = st.columns(3)
        percentile_cells[0].metric("p50 latency", f"{latency['p50_ms']:.1f} ms")
        percentile_cells[1].metric("p95 latency", f"{latency['p95_ms']:.1f} ms")
        percentile_cells[2].metric("p99 latency", f"{latency['p99_ms']:.1f} ms")
    validation_errors = {
        "missing": api["missing_feature_errors_since_start"],
        "invalid": api["invalid_feature_errors_since_start"],
    }
    st.caption(f"Feature validation errors since process start: {validation_errors}")
    st.caption(
        f"Model: `{metrics['model_version']}` · latency and error counters reset when the API process restarts."
    )

    st.subheader("Feature drift · Population Stability Index")
    drift_rows, review_items = [], []
    for feature, score in metrics["drift_psi"].items():
        if score is None:
            label = "Need 20+ observations"
        elif score >= 0.20:
            label = "Review requested"
            review_items.append(f"{feature} PSI {score:.2f}")
        elif score >= 0.10:
            label = "Watch"
        else:
            label = "No strong shift in this window"
        drift_rows.append({"Feature": feature, "PSI": score, "Status": label})
    if review_items:
        st.warning(
            "Investigate drift signals before making a model decision: " + ", ".join(review_items)
        )
    else:
        st.info(
            "No configured PSI review threshold was crossed. This does not prove the model is healthy."
        )
    if drift_rows:
        st.dataframe(pd.DataFrame(drift_rows), use_container_width=True, hide_index=True)

    trend = pd.DataFrame(metrics["trend_90d"])
    if not trend.empty:
        chart = px.line(
            trend,
            x="date",
            y=["predictions", "labeled"],
            markers=True,
            title="Predictions and matured labels by day",
        )
        st.plotly_chart(plot_layout(chart), use_container_width=True)
    st.subheader("Model version history")
    st.dataframe(pd.DataFrame(metrics["model_versions"]), use_container_width=True, hide_index=True)
    st.subheader("Delayed-label performance")
    delayed = metrics["delayed_performance"]
    if delayed is None:
        st.info("No matured outcomes yet. Predictions are eligible for outcomes after 14 days.")
    else:
        st.caption(
            f"{delayed['sample_size']:,} labeled rows · sources: {delayed['source_counts']} · selected threshold {delayed['evaluation_threshold']:.3f}"
        )
        if delayed["sample_size"] < 50:
            st.warning("Small sample: investigate these early metrics before drawing conclusions.")
        cells = st.columns(4)
        cells[0].metric("Matured labels", f"{delayed['sample_size']:,}")
        cells[1].metric("Precision", f"{delayed['precision']:.3f}")
        cells[2].metric("Recall", f"{delayed['recall']:.3f}")
        cells[3].metric("F1", f"{delayed['f1']:.3f}")
    st.caption(
        "Review drift, label coverage, and subgroup context before retraining. Alerts never promote a model automatically."
    )


def data_model_information():
    header(
        "Data and model information",
        "Provenance, feature schema, intended use, and the human review path.",
    )
    info, error = api_data(get_model_info)
    if error:
        show_api_error(error)
        return
    st.subheader("Dataset source and license")
    st.markdown(
        "The active model uses the in-repository `synthetic-events-v1` demo generator; no real game telemetry was downloaded. "
        "The IEEE Game Data Mining Competition describes commercial Blade & Soul logs, but the rules gate raw-file access behind a group and this build could not verify reuse permission. "
        "A checked 300-player sample was excluded because its churn flag is unrelated to activity. See `docs/dataset-card.md`."
    )
    schema = [
        {"Feature": "sessions_7d", "Window": "Past 7 days", "Meaning": "Session count"},
        {"Feature": "sessions_30d", "Window": "Past 30 days", "Meaning": "Session count"},
        {"Feature": "playtime_7d_hours", "Window": "Past 7 days", "Meaning": "Playtime hours"},
        {"Feature": "playtime_30d_hours", "Window": "Past 30 days", "Meaning": "Playtime hours"},
        {
            "Feature": "days_since_last_session",
            "Window": "At cutoff",
            "Meaning": "Activity age; 30 if no session was observed",
        },
    ]
    st.subheader("Feature schema")
    st.dataframe(pd.DataFrame(schema), use_container_width=True, hide_index=True)
    cards = st.columns(4)
    cards[0].metric("Active version", info["model_version"])
    cards[1].metric("Selected method", info["selected_model"].replace("_", " ").title())
    cards[2].metric("Validation threshold", f"{info['threshold']:.3f}")
    cards[3].metric("Held-out test sample", f"{info['evaluation']['n_samples']:,}")
    st.caption(
        f"Test evaluated {info['evaluation']['test_evaluated_at_utc']} · all reported training measures are synthetic."
    )
    st.subheader("Label and review contract")
    st.markdown(
        "Churn means no session in the fully observed 14 days after a prediction timestamp. API outcomes are accepted only after 14 days. "
        "Validate coverage and labels, calculate matured metrics, investigate drift, train a candidate, and review future test results before any promotion. "
        "No protected characteristics or manipulative targeting logic are used. A risk estimate is not a certainty or causal claim."
    )


def authenticate_dashboard():
    access_key = os.getenv("DASHBOARD_ACCESS_KEY", "")
    environment = os.getenv("PLAYERPULSE_ENV", "development").lower()
    if environment == "production" and len(access_key) < 32:
        st.error("The production dashboard is not configured with a strong access key.")
        st.stop()
    if not access_key:
        return
    if st.session_state.get("dashboard_authenticated"):
        with st.sidebar:
            if st.button("Sign out"):
                st.session_state["dashboard_authenticated"] = False
                st.rerun()
        return
    st.title("PlayerPulse access")
    with st.form("dashboard_login"):
        entered = st.text_input("Dashboard access key", type="password")
        submitted = st.form_submit_button("Continue")
    if submitted and hmac.compare_digest(access_key, entered):
        st.session_state["dashboard_authenticated"] = True
        st.rerun()
    if submitted:
        st.error("The access key was not accepted.")
    st.stop()


def main():
    authenticate_dashboard()
    with st.sidebar:
        st.markdown('<div class="pp-kicker">PLAYERPULSE</div>', unsafe_allow_html=True)
        st.markdown("## Retention studio")
        st.caption("MODEL · SYNTHETIC DEMO")
        page = st.radio(
            "Navigate",
            [
                "Retention Overview",
                "Player Risk Explorer",
                "Model Performance",
                "Model Monitoring",
                "Data and Model Information",
            ],
            label_visibility="collapsed",
        )
        st.divider()
        try:
            status = get_health()["status"]
            st.success("API ready" if status == "ready" else "API not ready")
        except ApiUnavailable:
            st.error("API unavailable")
        st.caption("Portfolio demo · no real-game claims")
    pages = {
        "Retention Overview": retention_overview,
        "Player Risk Explorer": player_risk_explorer,
        "Model Performance": model_performance,
        "Model Monitoring": model_monitoring,
        "Data and Model Information": data_model_information,
    }
    pages[page]()


if __name__ == "__main__":
    main()
