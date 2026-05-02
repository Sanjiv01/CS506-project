"""Render an interactive Plotly bar chart of model metrics.

Reads ``results/summary.csv`` (produced by ``summarize_results.py``) and
writes a self-contained HTML file at ``figures/results_interactive.html``.
The HTML loads Plotly from a CDN so it opens in any browser without extra
setup.
"""
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

RESULTS_CSV = Path("results/summary.csv")
OUT_HTML = Path("figures/results_interactive.html")


def main() -> None:
    if not RESULTS_CSV.exists():
        raise SystemExit(
            f"{RESULTS_CSV} missing. Run `make summarize` first."
        )

    df = pd.read_csv(RESULTS_CSV)
    metric_cols = [c for c in df.columns if c not in ("model", "tag")]

    fig = go.Figure()
    for metric in metric_cols:
        fig.add_trace(go.Bar(
            x=df["model"],
            y=df[metric],
            name=metric,
            hovertemplate="<b>%{x}</b><br>" + metric + ": %{y:.4f}<extra></extra>",
        ))

    fig.update_layout(
        title="Spotify Playlist Completion — Model Comparison",
        xaxis_title="Model",
        yaxis_title="Score",
        barmode="group",
        legend_title="Metric",
        template="plotly_white",
        height=500,
    )

    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(OUT_HTML, include_plotlyjs="cdn")
    print(f"saved {OUT_HTML}")


if __name__ == "__main__":
    main()
