import pandas as pd
import numpy as np
import scipy.sparse as spl

#%%
# --- Config ---
DATA_DIR = "data/spotify"
KNN_K = 500      # nearest neighbor playlists
POWB = 0.4       # popularity discount exponent (from 7_rest.py)
TOP_N = 500      # recommendations per playlist
BATCH_SIZE = 200 # test playlists per batch for matrix multiply

#%%
# --- Load data ---
train = pd.read_csv(f"{DATA_DIR}/playlists_train.csv")
test_seen = pd.read_csv(f"{DATA_DIR}/playlists_test_seen.csv")
test_hidden = pd.read_csv(f"{DATA_DIR}/playlists_test_hidden.csv")
song_meta = pd.read_csv(f"{DATA_DIR}/song_meta_no_duplicates.csv")

print(f"Train:       {train['pl_id'].nunique()} playlists, {len(train)} rows")
print(f"Test seen:   {test_seen['pl_id'].nunique()} playlists, {len(test_seen)} rows")
print(f"Test hidden: {test_hidden['pl_id'].nunique()} playlists, {len(test_hidden)} rows")
print(f"Songs:       {len(song_meta)}")

#%%
# --- Recode ids to contiguous 0-based integers ---
all_song_ids = song_meta["song_id"].values
song_code_map = {sid: i for i, sid in enumerate(all_song_ids)}
song_decode_map = np.array(all_song_ids)  # song_decode_map[code] = original song_id
n_songs = len(all_song_ids)

train_pl_ids = train["pl_id"].unique()
pl_code_map = {pid: i for i, pid in enumerate(train_pl_ids)}
n_train_pl = len(train_pl_ids)

train["pl_code"] = train["pl_id"].map(pl_code_map)
train["song_code"] = train["song_id"].map(song_code_map)

test_seen["song_code"] = test_seen["song_id"].map(song_code_map)
test_hidden["song_code"] = test_hidden["song_id"].map(song_code_map)
test_seen = test_seen.dropna(subset=["song_code"])
test_hidden = test_hidden.dropna(subset=["song_code"])
test_seen["song_code"] = test_seen["song_code"].astype(int)
test_hidden["song_code"] = test_hidden["song_code"].astype(int)

# Recode test pl_id to contiguous integers
test_pl_ids = test_seen["pl_id"].unique()
test_pl_code_map = {pid: i for i, pid in enumerate(test_pl_ids)}
test_pl_decode_map = test_pl_ids  # index -> original pl_id
n_test_pl = len(test_pl_ids)
test_seen["pl_code"] = test_seen["pl_id"].map(test_pl_code_map)
test_hidden["pl_code"] = test_hidden["pl_id"].map(test_pl_code_map)

#%%
# --- Build training sparse matrices ---
train_agg = train.drop_duplicates(subset=["pl_code", "song_code"]).copy()
train_agg["val"] = 1.0
train_agg["val_stoch"] = train_agg.groupby("pl_code")["val"].transform(
    lambda x: x / np.linalg.norm(x)
)

# sp_A: normalized, shape (n_train_pl, n_songs)
sp_A = spl.coo_matrix(
    (train_agg["val_stoch"].values,
     (train_agg["pl_code"].values, train_agg["song_code"].values)),
    shape=(n_train_pl, n_songs)
).tocsr()

# sp_A_const: binary, for aggregating neighbor song scores
sp_A_const_t = spl.coo_matrix(
    (train_agg["val"].values,
     (train_agg["song_code"].values, train_agg["pl_code"].values)),
    shape=(n_songs, n_train_pl)
).tocsr()

# Song popularity from training set
song_pop = train["song_code"].value_counts()

print(f"sp_A shape: {sp_A.shape}")

#%%
# --- Build test seed matrix ---
test_agg = test_seen.drop_duplicates(subset=["pl_code", "song_code"]).copy()
test_agg["val"] = 1.0
test_agg["val_stoch"] = test_agg.groupby("pl_code")["val"].transform(
    lambda x: x / np.linalg.norm(x)
)
test_agg["pop"] = test_agg["song_code"].map(song_pop).fillna(1)

# Popularity-discounted normalized values
test_agg["val_disc"] = test_agg["val_stoch"] / ((test_agg["pop"] - 1) ** POWB + 1)

# Sparse test seed matrix: (n_test_pl, n_songs)
sp_test = spl.coo_matrix(
    (test_agg["val_disc"].values,
     (test_agg["pl_code"].values, test_agg["song_code"].values)),
    shape=(n_test_pl, n_songs)
).tocsr()

# Lookup: test pl_code -> set of seen song codes (to exclude from recs)
seen_by_plcode = test_agg.groupby("pl_code")["song_code"].apply(set).to_dict()

print(f"sp_test shape: {sp_test.shape}")

#%%
# --- Batched KNN recommendation ---
# For each batch of test playlists:
#   1. similarities = sp_A.dot(test_batch.T)  -> (n_train, batch_size)
#   2. For each test playlist, take top-KNN_K neighbors
#   3. Aggregate songs from those neighbors (sp_A_const_t[:, neighbors].dot(weights))
#   4. Exclude seed songs, take top TOP_N

print(f"Running batched KNN ({BATCH_SIZE} playlists/batch)...")

all_recs = []
n_batches = int(np.ceil(n_test_pl / BATCH_SIZE))

for b in range(n_batches):
    start = b * BATCH_SIZE
    end = min(start + BATCH_SIZE, n_test_pl)
    batch_codes = np.arange(start, end)
    batch_size = len(batch_codes)

    # similarities: (n_train, batch_size)
    sim = sp_A.dot(sp_test[batch_codes].T).toarray()

    for i, pl_code in enumerate(batch_codes):
        col = sim[:, i]
        top_inds = col.argsort()[-KNN_K:][::-1]
        vals = col[top_inds]

        m = vals.max()
        if m == 0:
            m = 0.01
        weights = ((vals - vals.min()) / m) ** 2  # shape (KNN_K,)

        # Aggregate song scores from top neighbors
        song_scores = sp_A_const_t[:, top_inds].dot(weights)  # (n_songs,)

        # Exclude already-seen songs
        seen = np.array(list(seen_by_plcode.get(pl_code, [])), dtype=int)
        candidates = np.arange(n_songs)
        if len(seen) > 0:
            mask = np.ones(n_songs, dtype=bool)
            mask[seen] = False
            candidates = candidates[mask]
            song_scores_filtered = song_scores[candidates]
        else:
            song_scores_filtered = song_scores

        # Top TOP_N
        top_local = song_scores_filtered.argsort()[-TOP_N:][::-1]
        top_song_codes = candidates[top_local]

        all_recs.append(pd.DataFrame({
            "pl_code": pl_code,
            "song_code": top_song_codes,
            "rank": np.arange(len(top_song_codes)),
        }))

    if (b + 1) % 20 == 0 or b == n_batches - 1:
        print(f"  batch {b+1}/{n_batches} done ({end} playlists)")

recs = pd.concat(all_recs, ignore_index=True)
recs["pl_id"] = test_pl_decode_map[recs["pl_code"]]
recs["song_id"] = song_decode_map[recs["song_code"]]
print(f"Recommendations generated: {len(recs)} rows")

#%%
# --- Evaluation ---
hidden_by_plcode = test_hidden.groupby("pl_code")["song_code"].apply(set).to_dict()
recs_by_plcode = recs.sort_values(["pl_code", "rank"]).groupby("pl_code")["song_code"].apply(list).to_dict()

def dcg(hits, n_relevant):
    if n_relevant == 0:
        return 0.0, 0.0
    positions = np.arange(1, len(hits) + 1)
    gains = np.array(hits, dtype=float)
    actual = np.sum(gains / np.log2(positions + 1))
    ideal = np.sum(1.0 / np.log2(np.arange(1, n_relevant + 1) + 1))
    return actual, ideal

r_precisions, ndcgs = [], []
recalls_at = {10: [], 50: [], 100: [], 500: []}

for pl_code in hidden_by_plcode:
    relevant = hidden_by_plcode[pl_code]
    R = len(relevant)
    predicted = recs_by_plcode.get(pl_code, [])

    # R-precision
    hits_at_R = sum(1 for s in predicted[:R] if s in relevant)
    r_precisions.append(hits_at_R / R)

    # NDCG
    hits = [1 if s in relevant else 0 for s in predicted]
    actual_dcg, ideal_dcg = dcg(hits, R)
    ndcgs.append(actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0)

    # Recall@K
    for k in recalls_at:
        hits_k = sum(1 for s in predicted[:k] if s in relevant)
        recalls_at[k].append(hits_k / R)

print("\n=== KNN Results ===")
print(f"Playlists evaluated : {len(r_precisions)}")
print(f"R-Precision         : {np.mean(r_precisions):.4f}")
print(f"NDCG                : {np.mean(ndcgs):.4f}")
for k, vals in recalls_at.items():
    print(f"Recall@{k:<4}        : {np.mean(vals):.4f}")

#%%
# --- Popularity baseline ---
# Recommend the globally most popular songs (by training frequency),
# excluding songs already seen in each test playlist's seed set.
print("\nRunning popularity baseline...")

pop_ranked_codes = song_pop.reindex(range(n_songs), fill_value=0).sort_values(ascending=False).index.to_numpy()

pop_r_precisions, pop_ndcgs = [], []
pop_recalls_at = {10: [], 50: [], 100: [], 500: []}

for pl_code in hidden_by_plcode:
    relevant = hidden_by_plcode[pl_code]
    R = len(relevant)
    seen = seen_by_plcode.get(pl_code, set())

    # Vectorized: boolean mask over the ranked list, then slice
    seen_mask = np.zeros(n_songs, dtype=bool)
    if seen:
        seen_mask[np.fromiter(seen, dtype=int)] = True
    predicted = pop_ranked_codes[~seen_mask[pop_ranked_codes]][:TOP_N]

    relevant_arr = np.fromiter(relevant, dtype=int)
    predicted_set = set(predicted.tolist())

    hits_at_R = sum(1 for s in predicted[:R] if s in relevant)
    pop_r_precisions.append(hits_at_R / R)

    hits = np.isin(predicted, relevant_arr).tolist()
    actual_dcg, ideal_dcg = dcg(hits, R)
    pop_ndcgs.append(actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0)

    for k in pop_recalls_at:
        hits_k = int(np.isin(predicted[:k], relevant_arr).sum())
        pop_recalls_at[k].append(hits_k / R)

print("\n=== Popularity Baseline Results ===")
print(f"Playlists evaluated : {len(pop_r_precisions)}")
print(f"R-Precision         : {np.mean(pop_r_precisions):.4f}")
print(f"NDCG                : {np.mean(pop_ndcgs):.4f}")
for k, vals in pop_recalls_at.items():
    print(f"Recall@{k:<4}        : {np.mean(vals):.4f}")

#%%
# --- Side-by-side comparison ---
print("\n=== Comparison ===")
print(f"{'Metric':<18} {'KNN':>8} {'Popularity':>12} {'Lift':>8}")
print("-" * 50)
metrics = [
    ("R-Precision",  np.mean(r_precisions),    np.mean(pop_r_precisions)),
    ("NDCG",         np.mean(ndcgs),           np.mean(pop_ndcgs)),
    ("Recall@10",    np.mean(recalls_at[10]),   np.mean(pop_recalls_at[10])),
    ("Recall@50",    np.mean(recalls_at[50]),   np.mean(pop_recalls_at[50])),
    ("Recall@100",   np.mean(recalls_at[100]),  np.mean(pop_recalls_at[100])),
    ("Recall@500",   np.mean(recalls_at[500]),  np.mean(pop_recalls_at[500])),
]
for name, knn_val, pop_val in metrics:
    lift = (knn_val - pop_val) / pop_val * 100 if pop_val > 0 else float("nan")
    print(f"{name:<18} {knn_val:>8.4f} {pop_val:>12.4f} {lift:>+7.1f}%")
