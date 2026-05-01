"""ALS matrix factorization (Hu, Koren, Volinsky 2008 - implicit feedback).

Factorize a binary playlist x song matrix R into low-rank embeddings:

    R ~ X * Y^T,   X in R^{n_pl x f},   Y in R^{n_songs x f}

with confidence weighting c_ui = 1 + alpha * r_ui. We use the `implicit`
library, which provides a fast Cython implementation of the closed-form
ALS update.

For test playlists, we fold-in the seed: solve for x_test using the
held-in songs only, then score every song with x_test * Y^T.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as spl

import implicit
from implicit.als import AlternatingLeastSquares

# --- Config -----------------------------------------------------------------
DATA_DIR = Path("data/spotify")
RESULTS_DIR = Path("results")
TOP_N = 500
HIT_RATE_KS = (10, 20, 40)
BATCH_SIZE = 200

N_FACTORS = int(os.environ.get("N_FACTORS", 64))
N_ITER    = int(os.environ.get("N_ITER", 15))
ALS_ALPHA = float(os.environ.get("ALS_ALPHA", 40.0))
ALS_REG   = float(os.environ.get("ALS_REG", 0.01))

MODEL_TAG = f"als_f{N_FACTORS}_it{N_ITER}_a{ALS_ALPHA:g}_r{ALS_REG:g}"
print(f"ALS  factors={N_FACTORS}, iter={N_ITER}, alpha={ALS_ALPHA}, reg={ALS_REG}")

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

# --- Train R: binary user-item matrix --------------------------------------
train_dedup = train.drop_duplicates(subset=["pl_code", "song_code"])
R = spl.coo_matrix(
    (np.ones(len(train_dedup), dtype=np.float32),
     (train_dedup["pl_code"].values, train_dedup["song_code"].values)),
    shape=(n_train_pl, n_songs),
).tocsr()
print(f"R: {R.shape} nnz={R.nnz:,}")

# --- Fit ALS ---------------------------------------------------------------
model = AlternatingLeastSquares(
    factors=N_FACTORS,
    regularization=ALS_REG,
    alpha=ALS_ALPHA,
    iterations=N_ITER,
    use_gpu=False,
    random_state=42,
)
print("fitting ALS ...")
model.fit(R, show_progress=True)
Y = np.asarray(model.item_factors)  # (n_songs, f)
print(f"Y: {Y.shape}")

# --- Fold-in test playlists ------------------------------------------------
# x_u = (Y^T C^u Y + reg I)^{-1} Y^T C^u p(u)
# For binary R: A = Y^T Y + alpha * Yi^T Yi + reg I,   b = (1+alpha) * sum_i Yi
YtY = Y.T @ Y
reg_I = ALS_REG * np.eye(N_FACTORS)

seed_by_plcode = test_seen.groupby("pl_code")["song_code"].apply(set).to_dict()
seed_arrays = {pc: np.array(list(s), dtype=np.int32) for pc, s in seed_by_plcode.items()}

X_test = np.zeros((n_test_pl, N_FACTORS), dtype=np.float32)
for pl_code in range(n_test_pl):
    seed = seed_arrays.get(pl_code)
    if seed is None or len(seed) == 0:
        continue
    Yi = Y[seed]
    A = YtY + ALS_ALPHA * (Yi.T @ Yi) + reg_I
    b = (1.0 + ALS_ALPHA) * Yi.sum(axis=0)
    X_test[pl_code] = np.linalg.solve(A, b).astype(np.float32)
print(f"folded in {n_test_pl:,} test playlists into latent space")

# --- Score and recommend ----------------------------------------------------
recs = []
for s in range(0, n_test_pl, BATCH_SIZE):
    e = min(s + BATCH_SIZE, n_test_pl)
    scores_batch = X_test[s:e] @ Y.T
    for i in range(e - s):
        pl_code = s + i
        scores = scores_batch[i].astype(np.float64)
        seed = seed_arrays.get(pl_code)
        if seed is not None and len(seed) > 0:
            scores[seed] = -np.inf
        idx = np.argpartition(scores, -TOP_N)[-TOP_N:]
        idx = idx[np.argsort(scores[idx])[::-1]]
        recs.append(pd.DataFrame({
            "pl_code": pl_code,
            "song_code": idx,
            "rank": np.arange(TOP_N),
        }))
    batch_idx = (s // BATCH_SIZE) + 1
    n_batches_total = (n_test_pl + BATCH_SIZE - 1) // BATCH_SIZE
    if batch_idx % 20 == 0 or e == n_test_pl:
        print(f"  scored {e}/{n_test_pl}")

recs_df = pd.concat(recs, ignore_index=True)

# --- Evaluate ---------------------------------------------------------------
hidden_by_plcode = test_hidden.groupby("pl_code")["song_code"].apply(set).to_dict()
recs_by_plcode = (
    recs_df.sort_values(["pl_code", "rank"])
    .groupby("pl_code")["song_code"].apply(list).to_dict()
)
r_precisions = []
hit_rates = {k: [] for k in HIT_RATE_KS}
for pl_code, relevant in hidden_by_plcode.items():
    R_n = len(relevant)
    predicted = recs_by_plcode.get(pl_code, [])
    if R_n == 0:
        continue
    hits_at_R = sum(1 for s in predicted[:R_n] if s in relevant)
    r_precisions.append(hits_at_R / R_n)
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
               "config": {"N_FACTORS": N_FACTORS, "N_ITER": N_ITER,
                          "ALS_ALPHA": ALS_ALPHA, "ALS_REG": ALS_REG,
                          "top_n": TOP_N, "batch_size": BATCH_SIZE,
                          "hit_rate_ks": list(HIT_RATE_KS)}},
              f, indent=2)
print(f"saved {metrics_path}")
