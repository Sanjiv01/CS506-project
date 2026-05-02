"""Aggregate per-model metric JSONs into a single summary table.

Reads every ``results/*_metrics.json`` written by the ``rec_*.py`` scripts
and produces ``results/summary.csv`` plus a markdown table at
``results/summary.md``. The script is robust to missing models — it just
includes whoever has run.
"""
import json
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path("results")

# Friendly display names keyed by the prefix of MODEL_TAG (rec_*.py sets these).
PRETTY_NAMES = {
    "pop": "Popularity",
    "cooc": "Co-occurrence",
    "bm25": "BM25 Co-occurrence",
    "als": "ALS",
    "knn_advanced": "KNN (advanced)",
    "knn": "KNN",
}


def pretty(tag: str) -> str:
    for prefix, name in PRETTY_NAMES.items():
        if tag.startswith(prefix):
            return name
    return tag


def main() -> None:
    metric_files = sorted(RESULTS_DIR.glob("*_metrics.json"))
    if not metric_files:
        raise SystemExit(
            f"No metrics found in {RESULTS_DIR}/. Run `make models` first."
        )

    rows = []
    for path in metric_files:
        with open(path) as f:
            payload = json.load(f)
        tag = payload.get("model_tag", path.stem.replace("_metrics", ""))
        row = {"model": pretty(tag), "tag": tag}
        row.update(payload.get("metrics", {}))
        rows.append(row)

    df = pd.DataFrame(rows)
    metric_cols = [c for c in df.columns if c not in ("model", "tag")]
    df = df[["model", "tag", *metric_cols]].sort_values("HitRate@10", ascending=False).reset_index(drop=True)

    csv_path = RESULTS_DIR / "summary.csv"
    md_path = RESULTS_DIR / "summary.md"
    df.to_csv(csv_path, index=False)

    md = _to_markdown(df)
    md_path.write_text(md)
    print(md)
    print(f"\nsaved {csv_path}")
    print(f"saved {md_path}")


def _to_markdown(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, row in df.iterrows():
        cells = [f"{row[h]:.4f}" if isinstance(row[h], float) else str(row[h])
                 for h in headers]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
