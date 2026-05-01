#%% Imports
import os
from pathlib import Path

import numpy as np
import pandas as pd

#%% Config
CLEAN_PATH = Path("data/spotify_dataset_clean.csv")
OUT_DIR = Path("data/spotify")  # consumed by recommendation_final.py

# A song must appear in at least this many playlists across the dataset to be
# kept. Below this, the (artist, track) pair is most likely a typo / parsing
# artifact and provides no usable co-occurrence signal anyway.
MIN_SONG_FREQ = 3

# A playlist needs at least this many tracks (after song filtering) to be
# eligible for the test set. With HIDDEN_SONG_FRAC = 0.2, that yields at
# least ceil(6 * 0.2) = 2 hidden tracks and at least 4 (here, 5) seen tracks
# left as the prediction seed.
MIN_PLAYLIST_SIZE = 6

# Fraction of *all* playlists that go into the test set (sampled from the
# eligible pool defined above).
TEST_PLAYLIST_FRAC = 0.2

# Fraction of each test playlist's tracks that are masked as the held-out set.
HIDDEN_SONG_FRAC = 0.2

RANDOM_SEED = 42

OUT_DIR.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(RANDOM_SEED)

#%% Load cleaned dataset
# keep_default_na=False: the cleaned data contains literal strings like
# "None", "N/A", "NULL" as real artist / track / playlist names. Without
# this flag pandas would silently coerce them to NaN.
df = pd.read_csv(CLEAN_PATH, dtype=str, keep_default_na=False)
print(f"loaded: {df.shape}")
print("--- head ---")
print(df.head().to_string())

#%% Sanity checks (the cleaning step should already guarantee these)
print("--- null counts (must all be 0) ---")
print(df.isna().sum().to_string())
assert df.isna().sum().sum() == 0, "cleaned data has nulls -- re-run data_cleaning.py"

empty_counts = {c: int(df[c].eq("").sum()) for c in df.columns}
print(f"--- empty-string counts (must all be 0): {empty_counts} ---")
assert sum(empty_counts.values()) == 0, "cleaned data has empty-string fields -- re-run data_cleaning.py"

# Schema sanity
assert set(df.columns) == {"user_id", "artist_name", "track_name", "pl_name"}
print(f"sanity ok: {len(df):,} rows, no nulls, no empty fields")

#%% Filter low-frequency songs
# A "song" is identified by the composite key (artist_name, track_name).
song_freq = df.groupby(["artist_name", "track_name"]).size()
print(f"songs (artist+track) before filter: {len(song_freq):,}")
print("--- song-frequency distribution ---")
print(song_freq.describe(percentiles=[0.5, 0.9, 0.99]).to_string())
print(f"songs appearing exactly once:                {(song_freq == 1).sum():,}")
print(f"songs appearing < {MIN_SONG_FREQ} times (will be dropped): {(song_freq < MIN_SONG_FREQ).sum():,}")

# Examples of low-frequency songs that we are about to drop
low_freq = song_freq[song_freq < MIN_SONG_FREQ].sort_values()
print(f"--- {min(5, len(low_freq))} example songs being dropped (each appears {low_freq.iloc[0]} time) ---")
print(low_freq.head(5).reset_index().to_string(index=False))

valid_songs = song_freq[song_freq >= MIN_SONG_FREQ].index
before = len(df)
df = df.set_index(["artist_name", "track_name"])
df = df.loc[df.index.isin(valid_songs)].reset_index()
print(f"dropped {before - len(df):,} rows from low-freq songs -> {df.shape}")

#%% Compute per-playlist size after song filtering, and identify test-eligible playlists
playlist_size = df.groupby(["user_id", "pl_name"]).size()
print(f"distinct playlists after song filter: {len(playlist_size):,}")
print("--- playlist size distribution after song filter ---")
print(playlist_size.describe(percentiles=[0.5, 0.9, 0.99]).to_string())
print(f"playlists with >= {MIN_PLAYLIST_SIZE} tracks (eligible for test): "
      f"{(playlist_size >= MIN_PLAYLIST_SIZE).sum():,}")
print(f"playlists with <  {MIN_PLAYLIST_SIZE} tracks (forced into train): "
      f"{(playlist_size <  MIN_PLAYLIST_SIZE).sum():,}")

eligible_keys = playlist_size[playlist_size >= MIN_PLAYLIST_SIZE].index

#%% Assign pl_id (per (user_id, pl_name)) and song_id (per (artist_name, track_name))
pl_keys = (
    df[["user_id", "pl_name"]].drop_duplicates().reset_index(drop=True)
)
pl_keys["pl_id"] = pl_keys.index.astype(np.int32)

song_keys = (
    df[["artist_name", "track_name"]].drop_duplicates().reset_index(drop=True)
)
song_keys["song_id"] = song_keys.index.astype(np.int32)

df = df.merge(pl_keys, on=["user_id", "pl_name"]).merge(
    song_keys, on=["artist_name", "track_name"]
)
n_total_pl = pl_keys.shape[0]
n_songs = song_keys.shape[0]
print(f"assigned ids: {n_total_pl:,} playlists, {n_songs:,} songs")

# Eligible pl_ids for test sampling
eligible_pl_ids = (
    pl_keys.set_index(["user_id", "pl_name"]).loc[eligible_keys, "pl_id"].values
)
print(f"eligible test playlists: {len(eligible_pl_ids):,}")

#%% Train / test playlist split
# Sample TEST_PLAYLIST_FRAC * n_total_pl from the eligible pool.
n_test = int(round(n_total_pl * TEST_PLAYLIST_FRAC))
n_test = min(n_test, len(eligible_pl_ids))
test_pl_ids = rng.choice(eligible_pl_ids, size=n_test, replace=False)
test_pl_set = set(test_pl_ids.tolist())

train_mask = ~df["pl_id"].isin(test_pl_set)
train_df = df[train_mask].copy()
test_full_df = df[~train_mask].copy()

print(f"split: train rows={len(train_df):,} ({train_df['pl_id'].nunique():,} playlists), "
      f"test rows={len(test_full_df):,} ({test_full_df['pl_id'].nunique():,} playlists)")

# Quick look at the size distribution of train vs. test playlists
train_pl_size = train_df.groupby("pl_id").size()
test_pl_size  = test_full_df.groupby("pl_id").size()
print("--- train playlist sizes (no minimum applied) ---")
print(train_pl_size.describe(percentiles=[0.5, 0.9]).to_string())
print(f"train playlists with < {MIN_PLAYLIST_SIZE} tracks (kept anyway): "
      f"{(train_pl_size < MIN_PLAYLIST_SIZE).sum():,}")
print("--- test playlist sizes (must be >= MIN_PLAYLIST_SIZE) ---")
print(test_pl_size.describe(percentiles=[0.5, 0.9]).to_string())
assert test_pl_size.min() >= MIN_PLAYLIST_SIZE, "a test playlist is below MIN_PLAYLIST_SIZE"

#%% Within each test playlist, mask HIDDEN_SONG_FRAC of its tracks as "hidden"
hidden_frames = []
seen_frames = []
for pid, group in test_full_df.groupby("pl_id"):
    n_hidden = max(1, int(round(len(group) * HIDDEN_SONG_FRAC)))
    hidden_idx = rng.choice(group.index.values, size=n_hidden, replace=False)
    hidden_mask = group.index.isin(hidden_idx)
    hidden_frames.append(group[hidden_mask])
    seen_frames.append(group[~hidden_mask])

test_seen_df = pd.concat(seen_frames, ignore_index=True)
test_hidden_df = pd.concat(hidden_frames, ignore_index=True)

# Sanity: every test playlist has >=1 hidden and >=1 seen track
seen_per_pl = test_seen_df.groupby("pl_id").size()
hidden_per_pl = test_hidden_df.groupby("pl_id").size()
assert seen_per_pl.min() >= 1 and hidden_per_pl.min() >= 1, "test split left an empty side"
print(f"hidden tracks per test playlist: min={hidden_per_pl.min()}, "
      f"median={int(hidden_per_pl.median())}, max={hidden_per_pl.max()}")
print(f"seen tracks per test playlist:   min={seen_per_pl.min()}, "
      f"median={int(seen_per_pl.median())}, max={seen_per_pl.max()}")
print(f"total hidden rows: {len(test_hidden_df):,}, total seen rows: {len(test_seen_df):,}")

#%% Show one concrete example: a test playlist with its seen / hidden split
example_pid = int(test_pl_ids[0])
ex_pl = pl_keys.loc[pl_keys["pl_id"] == example_pid, ["user_id", "pl_name"]].iloc[0]
print(f"--- example test playlist (pl_id={example_pid}, name={ex_pl['pl_name']!r}) ---")
print("seen:")
print(test_seen_df[test_seen_df["pl_id"] == example_pid]
      [["artist_name", "track_name"]].to_string(index=False))
print("hidden:")
print(test_hidden_df[test_hidden_df["pl_id"] == example_pid]
      [["artist_name", "track_name"]].to_string(index=False))

#%% Persist artifacts consumed by recommendation_final.py
def save_pl_song(sub_df, label):
    out = sub_df[["pl_id", "song_id"]].drop_duplicates()
    path = OUT_DIR / f"playlists_{label}.csv"
    out.to_csv(path, index=False)
    print(f"saved {path}  ({len(out):,} rows)")

save_pl_song(train_df, "train")
save_pl_song(test_seen_df, "test_seen")
save_pl_song(test_hidden_df, "test_hidden")

# song_meta: id <-> (artist, track) lookup. recommendation_final.py uses
# song_meta["song_id"] as the canonical universe of songs.
song_meta = song_keys[["song_id", "artist_name", "track_name"]].copy()
song_meta_path = OUT_DIR / "song_meta_no_duplicates.csv"
song_meta.to_csv(song_meta_path, index=False)
print(f"saved {song_meta_path}  ({len(song_meta):,} rows)")

#%% Persist playlist meta (id <-> (user, name) lookup) for train and test
def save_pl_meta(sub_df, label):
    meta = (
        sub_df.groupby(["pl_id", "user_id", "pl_name"])
              .agg(num_artists=("artist_name", "nunique"),
                   num_tracks=("track_name", "nunique"))
              .reset_index()
    )
    path = OUT_DIR / f"playlist_meta_{label}.csv"
    meta.to_csv(path, index=False)
    print(f"saved {path}  ({len(meta):,} rows)")
    return meta

save_pl_meta(train_df, "train")
save_pl_meta(test_full_df, "test")

print("done.")
