"""Popularity baseline.

For every test playlist, recommend the globally most-played songs (ranked by
the number of distinct training playlists each song appears in), excluding
songs already present in the test playlist's seed.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

# --- Config -----------------------------------------------------------------
DATA_DIR = Path("data/spotify")
RESULTS_DIR = Path("results")
TOP_N = 500
HIT_RATE_KS = (10, 20, 40)
MODEL_TAG = "pop"

# --- Load -------------------------------------------------------------------
train       = pd.read_csv(DATA_DIR / "playlists_train.csv")
test_seen   = pd.read_csv(DATA_DIR / "playlists_test_seen.csv")
test_hidden = pd.read_csv(DATA_DIR / "playlists_test_hidden.csv")
song_meta   = pd.read_csv(DATA_DIR / "song_meta_no_duplicates.csv", keep_default_na=False)
print(f"train:       {train['pl_id'].nunique():>7,} pl, {len(train):>10,} rows")
print(f"test_seen:   {test_seen['pl_id'].nunique():>7,} pl, {len(test_seen):>10,} rows")
print(f"test_hidden: {test_hidden['pl_id'].nunique():>7,} pl, {len(test_hidden):>10,} rows")
print(f"songs:       {len(song_meta):>7,}")

# --- Recode ids to contiguous 0-based integers ------------------------------
all_song_ids = song_meta["song_id"].values
song_code_map = {sid: i for i, sid in enumerate(all_song_ids)}
song_decode_map = np.asarray(all_song_ids)
n_songs = len(all_song_ids)

train["song_code"] = train["song_id"].map(song_code_map)
test_seen["song_code"] = test_seen["song_id"].map(song_code_map)
test_hidden["song_code"] = test_hidden["song_id"].map(song_code_map)
test_seen = test_seen.dropna(subset=["song_code"]).copy()
test_hidden = test_hidden.dropna(subset=["song_code"]).copy()
test_seen["song_code"] = test_seen["song_code"].astype(np.int32)
test_hidden["song_code"] = test_hidden["song_code"].astype(np.int32)

train_pl_ids = train["pl_id"].unique()
pl_code_map = {pid: i for i, pid in enumerate(train_pl_ids)}
n_train_pl = len(train_pl_ids)
train["pl_code"] = train["pl_id"].map(pl_code_map).astype(np.int32)

test_pl_ids = test_seen["pl_id"].unique()
test_pl_code_map = {pid: i for i, pid in enumerate(test_pl_ids)}
test_pl_decode_map = test_pl_ids
n_test_pl = len(test_pl_ids)
test_seen["pl_code"] = test_seen["pl_id"].map(test_pl_code_map).astype(np.int32)
test_hidden["pl_code"] = test_hidden["pl_id"].map(test_pl_code_map).astype(np.int32)

# --- Popularity score: count of distinct training playlists per song --------
pop = (
    train.drop_duplicates(subset=["pl_code", "song_code"])["song_code"]
    .value_counts()
)
pop_scores = np.zeros(n_songs, dtype=np.float64)
pop_scores[pop.index.values] = pop.values
print(f"songs with non-zero training popularity: {(pop_scores > 0).sum():,}")

# --- Pre-sort the global popularity ranking once ---------------------------
seen_by_plcode = test_seen.groupby("pl_code")["song_code"].apply(set).to_dict()
seen_arrays = {pc: np.array(list(s), dtype=np.int32) for pc, s in seen_by_plcode.items()}

order_global = np.argsort(pop_scores)[::-1]
recs = []
for pl_code in range(n_test_pl):
    seen = seen_arrays.get(pl_code)
    if seen is None or len(seen) == 0:
        ranked = order_global[:TOP_N]
    else:
        mask = np.ones(n_songs, dtype=bool)
        mask[seen] = False
        ranked = order_global[mask[order_global]][:TOP_N]
    recs.append(pd.DataFrame({
        "pl_code": pl_code,
        "song_code": ranked,
        "rank": np.arange(len(ranked)),
    }))

recs_df = pd.concat(recs, ignore_index=True)
print(f"generated {len(recs_df):,} ranking entries for {n_test_pl:,} test playlists")

# --- Evaluate ---------------------------------------------------------------
hidden_by_plcode = test_hidden.groupby("pl_code")["song_code"].apply(set).to_dict()
recs_by_plcode = (
    recs_df.sort_values(["pl_code", "rank"])
    .groupby("pl_code")["song_code"].apply(list).to_dict()
)
r_precisions = []
hit_rates = {k: [] for k in HIT_RATE_KS}
for pl_code, relevant in hidden_by_plcode.items():
    R = len(relevant)
    predicted = recs_by_plcode.get(pl_code, [])
    if R == 0:
        continue
    hits_at_R = sum(1 for s in predicted[:R] if s in relevant)
    r_precisions.append(hits_at_R / R)
    for k in HIT_RATE_KS:
        hit_rates[k].append(1.0 if any(s in relevant for s in predicted[:k]) else 0.0)

metrics = {
    "R-Precision": float(np.mean(r_precisions)),
    **{f"HitRate@{k}": float(np.mean(hit_rates[k])) for k in HIT_RATE_KS},
}
print(f"\n=== {MODEL_TAG} ===")
print(f"playlists evaluated: {len(r_precisions)}")
for name, val in metrics.items():
    print(f"  {name:<14}: {val:.4f}")

# --- Save -------------------------------------------------------------------
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
out = recs_df.copy()
out["pl_id"] = test_pl_decode_map[out["pl_code"].values]
out["song_id"] = song_decode_map[out["song_code"].values]
recs_path = RESULTS_DIR / f"{MODEL_TAG}_recs.csv"
out[["pl_id", "song_id", "rank"]].to_csv(recs_path, index=False)
print(f"saved {recs_path}")

metrics_path = RESULTS_DIR / f"{MODEL_TAG}_metrics.json"
with open(metrics_path, "w") as f:
    json.dump({"model_tag": MODEL_TAG, "metrics": metrics,
               "config": {"top_n": TOP_N, "hit_rate_ks": list(HIT_RATE_KS)}},
              f, indent=2)
print(f"saved {metrics_path}")
