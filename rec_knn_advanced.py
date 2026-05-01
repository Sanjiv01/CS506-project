"""kNN advanced — cosine similarity with inverse-item-frequency (IIF) reweighting.

Track-weight coefficient applied to the seed song:

    iif(s) = ((f_s - 1) ** rho + 1) ** -1

where f_s is the number of *training* playlists containing song s. Two
playlists that share rare songs are treated as more similar than two that
share popular songs. The kNN scoring proceeds exactly as in rec_knn.py
(threaded, batched, min-max-square reweight on top-K neighbors).

We sweep rho over {0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60} and report
metrics for each value. The best rho on R-Precision is reported separately.
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

KNN_K = int(os.environ.get("KNN_K", 500))
RHOS = tuple(float(x) for x in os.environ.get(
    "RHOS", "0.30,0.35,0.40,0.45,0.50,0.55,0.60"
).split(","))
print(f"kNN advanced (IIF cosine), K={KNN_K}, rhos={RHOS}")

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

# --- Train cosine matrix (rho-independent) ---------------------------------
train_dedup = train.drop_duplicates(subset=["pl_code", "song_code"]).copy()
train_dedup["w"] = 1.0
train_dedup["w"] = train_dedup.groupby("pl_code")["w"].transform(
    lambda x: x / np.linalg.norm(x.values)
)
sp_A = spl.coo_matrix(
    (train_dedup["w"].values.astype(np.float32),
     (train_dedup["pl_code"].values, train_dedup["song_code"].values)),
    shape=(n_train_pl, n_songs),
).tocsr()
sp_A_const_t = spl.coo_matrix(
    (np.ones(len(train_dedup), dtype=np.float32),
     (train_dedup["song_code"].values, train_dedup["pl_code"].values)),
    shape=(n_songs, n_train_pl),
).tocsr()
print(f"sp_A: {sp_A.shape} nnz={sp_A.nnz:,}")

# Song frequency f_s in training playlists.
song_freq = np.zeros(n_songs, dtype=np.float64)
freq_series = train_dedup["song_code"].value_counts()
song_freq[freq_series.index.values] = freq_series.values

seen_by_plcode = test_seen.groupby("pl_code")["song_code"].apply(set).to_dict()
seen_arrays = {pc: np.array(list(s), dtype=np.int32) for pc, s in seen_by_plcode.items()}
hidden_by_plcode = test_hidden.groupby("pl_code")["song_code"].apply(set).to_dict()

# --- Recommend / evaluate / save loop over rho -----------------------------
def recommend(sp_test_seed):
    def process_batch(b):
        start = b * BATCH_SIZE
        end = min(start + BATCH_SIZE, n_test_pl)
        batch_codes = np.arange(start, end)
        bs = end - start

        sim = sp_A.dot(sp_test_seed[batch_codes].T).toarray()
        sim_T = sim.T
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
        song_scores = sp_A_const_t.dot(W).toarray()

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
    return pd.concat([df for batch in all_recs for df in batch], ignore_index=True)


def evaluate(recs_df):
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
    return {
        "R-Precision": float(np.mean(r_precisions)),
        **{f"HitRate@{k}": float(np.mean(hit_rates[k])) for k in HIT_RATE_KS},
    }


RESULTS_DIR.mkdir(parents=True, exist_ok=True)
results = {}
for rho in RHOS:
    print(f"\n--- rho = {rho:.2f} ---")
    iif = 1.0 / ((song_freq - 1.0).clip(min=0.0) ** rho + 1.0)

    test_dedup = test_seen.drop_duplicates(subset=["pl_code", "song_code"]).copy()
    test_dedup["w"] = 1.0
    test_dedup["w"] = test_dedup.groupby("pl_code")["w"].transform(
        lambda x: x / np.linalg.norm(x.values)
    )
    test_dedup["w"] = test_dedup["w"].values * iif[test_dedup["song_code"].values]
    sp_test_seed = spl.coo_matrix(
        (test_dedup["w"].values.astype(np.float32),
         (test_dedup["pl_code"].values, test_dedup["song_code"].values)),
        shape=(n_test_pl, n_songs),
    ).tocsr()

    print(f"running rho={rho:.2f}: {BATCH_SIZE} pls/batch x {N_WORKERS} workers")
    recs_df = recommend(sp_test_seed)
    metrics = evaluate(recs_df)
    tag = f"knn_iif_rho_{rho:.2f}"
    print(f"=== {tag} ===")
    for name, val in metrics.items():
        print(f"  {name:<14}: {val:.4f}")

    out = recs_df.copy()
    out["pl_id"] = test_pl_decode_map[out["pl_code"].values]
    out["song_id"] = song_decode_map[out["song_code"].values]
    out[["pl_id", "song_id", "rank"]].to_csv(RESULTS_DIR / f"{tag}_recs.csv", index=False)
    with open(RESULTS_DIR / f"{tag}_metrics.json", "w") as f:
        json.dump({"model_tag": tag, "metrics": metrics,
                   "config": {"rho": rho, "KNN_K": KNN_K,
                              "top_n": TOP_N, "batch_size": BATCH_SIZE,
                              "n_workers": N_WORKERS, "hit_rate_ks": list(HIT_RATE_KS)}},
                  f, indent=2)
    print(f"  saved {tag}_recs.csv and {tag}_metrics.json")
    results[rho] = metrics

# --- Sweep summary ---------------------------------------------------------
print("\n=== kNN advanced rho sweep ===")
header = f"{'rho':<6}" + "".join(f"{m:>14}" for m in ["R-Precision", *[f"HitRate@{k}" for k in HIT_RATE_KS]])
print(header)
print("-" * len(header))
for rho, m in results.items():
    row = f"{rho:<6.2f}{m['R-Precision']:>14.4f}"
    for k in HIT_RATE_KS:
        row += f"{m[f'HitRate@{k}']:>14.4f}"
    print(row)

best_rho = max(results, key=lambda r: results[r]["R-Precision"])
print(f"\nbest rho on R-Precision: {best_rho:.2f}")
