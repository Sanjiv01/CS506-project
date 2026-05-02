"""Smoke tests for the recommendation pipeline.

These tests build a tiny Spotify-shaped fixture in a temp directory and
execute the standalone rec_*.py scripts via runpy. They confirm that each
script reads the expected ``data/spotify/*.csv`` inputs, runs end-to-end,
and writes a valid ``results/<tag>_metrics.json`` with a numeric
``HitRate@10``.

Only the popularity baseline and co-occurrence model are smoke-tested —
those are the cheapest to run and exercise both the simple ranking path
and the sparse-matrix co-occurrence path.
"""
import json
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def fixture_workdir(tmp_path, monkeypatch):
    """Build a tiny Spotify-shaped dataset under tmp_path/data/spotify and chdir there."""
    rng = np.random.default_rng(0)
    # rec_cooc.py uses np.argpartition(scores, -TOP_N) with TOP_N=500, so the
    # fixture needs at least TOP_N + 1 distinct songs.
    n_songs = 600
    n_train_pl = 50

    train_rows = []
    for pl in range(n_train_pl):
        base = int(rng.integers(0, n_songs - 50))
        size = int(rng.integers(5, 11))
        for s in rng.choice(np.arange(base, base + 50), size=size, replace=False):
            train_rows.append((pl, int(s)))

    test_seen_rows, test_hidden_rows = [], []
    for pl in range(n_train_pl, n_train_pl + 5):
        base = int(rng.integers(0, n_songs - 50))
        songs = rng.choice(np.arange(base, base + 50), size=5, replace=False)
        for s in songs[:3]:
            test_seen_rows.append((pl, int(s)))
        for s in songs[3:]:
            test_hidden_rows.append((pl, int(s)))

    data_dir = tmp_path / "data" / "spotify"
    data_dir.mkdir(parents=True)
    pd.DataFrame(train_rows, columns=["pl_id", "song_id"]).to_csv(
        data_dir / "playlists_train.csv", index=False)
    pd.DataFrame(test_seen_rows, columns=["pl_id", "song_id"]).to_csv(
        data_dir / "playlists_test_seen.csv", index=False)
    pd.DataFrame(test_hidden_rows, columns=["pl_id", "song_id"]).to_csv(
        data_dir / "playlists_test_hidden.csv", index=False)
    pd.DataFrame({
        "song_id": np.arange(n_songs),
        "artist_name": [f"a{i}" for i in range(n_songs)],
        "track_name": [f"t{i}" for i in range(n_songs)],
    }).to_csv(data_dir / "song_meta_no_duplicates.csv", index=False)

    monkeypatch.chdir(tmp_path)
    return tmp_path


def _smoke(script: str, workdir: Path, tag: str) -> None:
    runpy.run_path(str(REPO_ROOT / script), run_name="__main__")
    metrics_path = workdir / "results" / f"{tag}_metrics.json"
    assert metrics_path.exists(), f"{metrics_path} not written"
    payload = json.loads(metrics_path.read_text())
    assert payload["model_tag"] == tag
    assert "HitRate@10" in payload["metrics"]
    assert isinstance(payload["metrics"]["HitRate@10"], float)


def test_rec_pop_runs_end_to_end(fixture_workdir):
    _smoke("rec_pop.py", fixture_workdir, "pop")


def test_rec_cooc_runs_end_to_end(fixture_workdir):
    _smoke("rec_cooc.py", fixture_workdir, "cooc")
