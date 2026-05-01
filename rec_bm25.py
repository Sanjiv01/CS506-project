"""BM25-weighted co-occurrence.

Each training playlist is treated as a "document" and each seed song in a
test playlist as a query token. Per-(playlist, song) BM25 weight:

    weight[p, s] = B[p] * IDF(s)
    B[p]   = (k1 + 1) / (1 + k1 * (1 - b + b * |p| / avg_p_len))
    IDF(s) = log((N - df(s) + 0.5) / (df(s) + 0.5) + 1)

The query vector is the binary seed indicator. The neighbor-aggregation
step then converts similar-train-playlist matches into song scores via
top-K kNN with min-max-square reweighting (mirrors recommendation_final.py).
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as spl

# --- Config -----------------------------------------------------------------
DATA_DIR = Path("data/spotify")
RESULTS_DIR = Path("results")
TOP_N = 500
HIT_RATE_KS = (10, 20, 40)
BATCH_SIZE = 200
N_WORKERS = 4

KNN_K   = int(os.environ.get("KNN_K", 500))
BM25_K1 = float(os.environ.get("BM25_K1", 1.5))
BM25_B  = float(os.environ.get("BM25_B", 0.75))

MODEL_TAG = f"bm25_k1_{BM25_K1:g}_b_{BM25_B:g}"
print(f"BM25  k1={BM25_K1}, b={BM25_B}, KNN_K={KNN_K}")

# --- Load -------------------------------------------------------------------
train       = pd.read_csv(DATA_DIR / "playlists_train.csv")
test_seen   = pd.read_csv(DATA_DIR / "playlists_test_seen.csv")
test_hidden = pd.read_csv(DATA_DIR / "playlists_test_hidden.csv")
song_meta   = pd.read_csv(DATA_DIR / "song_meta_no_duplicates.csv", keep_default_na=False)
print(f"train:       {train['pl_id'].nunique():>7,} pl, {len(train):>10,} rows")
print(f"test_seen:   {test_seen['pl_id'].nunique():>7,} pl, {len(test_seen):>10,} rows")
print(f"test_hidden: {test_hidden['pl_id'].nunique():>7,} pl, {len(test_hidden):>10,} rows")
print(f"songs:       {len(song_meta):>7,}")

# --- Recode -----------------------------------------------------------------
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

# --- BM25 weight per (train_playlist, song) --------------------------------
train_dedup = train.drop_duplicates(subset=["pl_code", "song_code"]).copy()
pl_lens = train_dedup.groupby("pl_code").size()
avg_len = pl_lens.mean()
df_song = train_dedup.groupby("song_code").size()

B = (BM25_K1 + 1) / (1 + BM25_K1 * (1 - BM25_B + BM25_B * pl_lens / avg_len))
IDF = np.log((n_train_pl - df_song + 0.5) / (df_song + 0.5) + 1)
train_dedup["weight"] = (
    train_dedup["pl_code"].map(B).values
    * train_dedup["song_code"].map(IDF).values
)

sp_A = spl.coo_matrix(  # (n_train_pl, n_songs) BM25-weighted
    (train_dedup["weight"].values.astype(np.float32),
     (train_dedup["pl_code"].values, train_dedup["song_code"].values)),
    shape=(n_train_pl, n_songs),
).tocsr()
sp_A_const_t = spl.coo_matrix(  # (n_songs, n_train_pl) binary, for aggregation
    (np.ones(len(train_dedup), dtype=np.float32),
     (train_dedup["song_code"].values, train_dedup["pl_code"].values)),
    shape=(n_songs, n_train_pl),
).tocsr()

# --- Test seed: binary query (BM25 IDF lives in sp_A) ----------------------
test_dedup = test_seen.drop_duplicates(subset=["pl_code", "song_code"])
sp_test_seed = spl.coo_matrix(
    (np.ones(len(test_dedup), dtype=np.float32),
     (test_dedup["pl_code"].values, test_dedup["song_code"].values)),
    shape=(n_test_pl, n_songs),
).tocsr()
print(f"sp_A: {sp_A.shape} nnz={sp_A.nnz:,}")

seen_by_plcode = test_seen.groupby("pl_code")["song_code"].apply(set).to_dict()
seen_arrays = {pc: np.array(list(s), dtype=np.int32) for pc, s in seen_by_plcode.items()}

# --- Threaded batched neighbor-kNN scoring ---------------------------------
def process_batch(b):
    start = b * BATCH_SIZE
    end = min(start + BATCH_SIZE, n_test_pl)
    batch_codes = np.arange(start, end)
    bs = end - start

    sim = sp_A.dot(sp_test_seed[batch_codes].T).toarray()
    sim_T = sim.T  # (bs, n_train_pl)

    if n_train_pl <= KNN_K:
        top_k_inds = np.tile(np.arange(n_train_pl), (bs, 1))
    else:
        top_k_inds = np.argpartition(sim_T, -KNN_K, axis=1)[:, -KNN_K:]
    top_k_vals = np.take_along_axis(sim_T, top_k_inds, axis=1)

    vmax = top_k_vals.max(axis=1, keepdims=True)
    vmax = np.where(vmax == 0, 0.01, vmax)
    vmin = top_k_vals.min(axis=1, keepdims=True)
    weights = ((top_k_vals - vmin) / vmax) ** 2

    rows = top_k_inds.ravel()
    cols = np.repeat(np.arange(bs), top_k_inds.shape[1])
    W = spl.coo_matrix(
        (weights.ravel(), (rows, cols)),
        shape=(n_train_pl, bs),
    ).tocsc()
    song_scores = sp_A_const_t.dot(W).toarray()  # (n_songs, bs)

    batch_recs = []
    for i, pl_code in enumerate(batch_codes):
        scores = song_scores[:, i]
        seen = seen_arrays.get(int(pl_code))
        if seen is not None and len(seen) > 0:
            scores[seen] = -1.0
        idx = np.argpartition(scores, -TOP_N)[-TOP_N:]
        idx = idx[np.argsort(scores[idx])[::-1]]
        batch_recs.append(pd.DataFrame({
            "pl_code": pl_code,
            "song_code": idx,
            "rank": np.arange(TOP_N),
        }))
    return b, batch_recs


print(f"running {MODEL_TAG}: {BATCH_SIZE} pls/batch x {N_WORKERS} workers")
n_batches = int(np.ceil(n_test_pl / BATCH_SIZE))
all_recs = [None] * n_batches
with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
    futures = [ex.submit(process_batch, b) for b in range(n_batches)]
    done = 0
    for fut in futures:
        b, batch_recs = fut.result()
        all_recs[b] = batch_recs
        done += 1
        if done % 20 == 0 or done == n_batches:
            print(f"  {done}/{n_batches} batches done")

recs_df = pd.concat([df for batch in all_recs for df in batch], ignore_index=True)
print(f"generated {len(recs_df):,} ranking entries")

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
               "config": {"BM25_K1": BM25_K1, "BM25_B": BM25_B, "KNN_K": KNN_K,
                          "top_n": TOP_N, "batch_size": BATCH_SIZE, "n_workers": N_WORKERS,
                          "hit_rate_ks": list(HIT_RATE_KS)}},
              f, indent=2)
print(f"saved {metrics_path}")
