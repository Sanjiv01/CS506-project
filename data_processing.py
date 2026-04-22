#%%
import pandas as pd
import numpy as np
import os

#%%
# --- Config ---
# DATA_PATH = "dataset/archive/spotify_dataset.csv"
DATA_PATH = "spotify_dataset.csv"
OUT_DIR = "data/spotify"
MIN_SONG_FREQ = 3
MIN_PLAYLIST_SIZE = 6
TEST_PLAYLIST_FRAC = 0.2
HIDDEN_SONG_FRAC = 0.2
RANDOM_SEED = 42

os.makedirs(OUT_DIR, exist_ok=True)

#%%
# --- Load & rename columns ---
df = pd.read_csv(DATA_PATH, on_bad_lines='skip')
df.columns = df.columns.str.strip().str.strip('"')
df = df.rename(columns={
    "user_id": "user_id",
    "artistname": "artist_name",
    "trackname": "track_name",
    "playlistname": "pl_name",
})
print("Loaded:", df.shape)

#%%
# --- Drop nulls ---
before = len(df)
df.dropna(inplace=True)
print(f"Dropped {before - len(df)} null rows, remaining: {len(df)}")

#%%
# --- Drop empty strings ---
before = len(df)
mask = (df == "").any(axis=1)
df = df[~mask]
print(f"Dropped {before - len(df)} empty-string rows, remaining: {len(df)}")

#%%
# --- Filter songs: remove songs below MIN_SONG_FREQ ---
# A song is identified by (artist_name, track_name)
song_freq = df.groupby(["artist_name", "track_name"])["user_id"].count()
valid_songs = song_freq[song_freq >= MIN_SONG_FREQ].index
before = len(df)
df = df.set_index(["artist_name", "track_name"])
df = df[df.index.isin(valid_songs)].reset_index()
print(f"Dropped {before - len(df)} rows from low-freq songs (<{MIN_SONG_FREQ}), remaining: {len(df)}")

#%%
# --- Filter playlists: remove playlists with fewer than MIN_PLAYLIST_SIZE songs ---
# A playlist is identified by (user_id, pl_name)
playlist_size = df.groupby(["user_id", "pl_name"])["track_name"].count()
valid_playlists = playlist_size[playlist_size >= MIN_PLAYLIST_SIZE].index
before = len(df)
df = df.set_index(["user_id", "pl_name"])
df = df[df.index.isin(valid_playlists)].reset_index()
print(f"Dropped {before - len(df)} rows from small playlists (<{MIN_PLAYLIST_SIZE}), remaining: {len(df)}")

#%%
# --- Re-check: ensure all remaining songs still appear >= MIN_SONG_FREQ ---
song_freq2 = df.groupby(["artist_name", "track_name"])["user_id"].count()
valid_songs2 = song_freq2[song_freq2 >= MIN_SONG_FREQ].index
before = len(df)
df = df.set_index(["artist_name", "track_name"])
df = df[df.index.isin(valid_songs2)].reset_index()
print(f"Re-check: dropped {before - len(df)} rows, remaining: {len(df)}")

#%%
# --- Assign pl_id and song_id ---
# pid: unique integer per (user_id, pl_name)
pl_keys = df[["user_id", "pl_name"]].drop_duplicates().reset_index(drop=True)
pl_keys["pl_id"] = pl_keys.index
df = df.merge(pl_keys, on=["user_id", "pl_name"])

# song_id: unique integer per (artist_name, track_name)
song_keys = df[["artist_name", "track_name"]].drop_duplicates().reset_index(drop=True)
song_keys["song_id"] = song_keys.index
df = df.merge(song_keys, on=["artist_name", "track_name"])

print(f"Playlists: {df['pl_id'].nunique()}, Songs: {df['song_id'].nunique()}")

#%%
display(df.head())

#%%
# --- Train/test split ---
# 20% of playlists go to test; within each test playlist, hide 20% of songs
rng = np.random.default_rng(RANDOM_SEED)
all_pids = df["pl_id"].unique()
n_test = int(len(all_pids) * TEST_PLAYLIST_FRAC)
test_pids = rng.choice(all_pids, size=n_test, replace=False)
train_pids = np.setdiff1d(all_pids, test_pids)

train_df = df[df["pl_id"].isin(train_pids)].copy()
test_full_df = df[df["pl_id"].isin(test_pids)].copy()

# For each test playlist, hide round(20% of songs)
hidden_rows = []
seen_rows = []
for pid, group in test_full_df.groupby("pl_id"):
    n_hidden = round(len(group) * HIDDEN_SONG_FRAC)
    n_hidden = max(n_hidden, 1)  # hide at least 1
    hidden_idx = rng.choice(group.index, size=n_hidden, replace=False)
    hidden_rows.append(group.loc[hidden_idx])
    seen_rows.append(group.drop(hidden_idx))

test_seen_df = pd.concat(seen_rows)
test_hidden_df = pd.concat(hidden_rows)

print(f"Train playlists: {len(train_pids)}, Test playlists: {len(test_pids)}")
print(f"Test seen rows: {len(test_seen_df)}, Test hidden rows: {len(test_hidden_df)}")

#%%
display(group.loc[hidden_idx].head())

#%%
# --- Save playlist_meta.csv ---
def make_playlist_meta(sub_df, label):
    meta = sub_df.groupby(["pl_id", "user_id", "pl_name"]).agg(
        num_artists=("artist_name", "nunique"),
        num_tracks=("track_name", "nunique"),
    ).reset_index()
    path = os.path.join(OUT_DIR, f"playlist_meta_{label}.csv")
    meta.to_csv(path, index=False)
    print(f"Saved {path} ({len(meta)} rows)")
    return meta

train_pl_meta = make_playlist_meta(train_df, "train")
test_pl_meta = make_playlist_meta(test_full_df, "test")

#%%
display(train_pl_meta.head())
display(test_pl_meta.head())

#%%
# --- Save song_meta_no_duplicates.csv ---
song_meta = song_keys[["song_id", "artist_name", "track_name"]].copy()
song_meta.to_csv(os.path.join(OUT_DIR, "song_meta_no_duplicates.csv"), index=False)
print(f"Saved song_meta_no_duplicates.csv ({len(song_meta)} rows)")

#%%
display(song_meta.head())

#%%
# --- Save playlists.csv (pl_id, song_id — no position) ---
def save_playlists(sub_df, label):
    pl_songs = sub_df[["pl_id", "song_id"]].drop_duplicates()
    path = os.path.join(OUT_DIR, f"playlists_{label}.csv")
    pl_songs.to_csv(path, index=False)
    print(f"Saved {path} ({len(pl_songs)} rows)")

save_playlists(train_df, "train")
save_playlists(test_seen_df, "test_seen")
save_playlists(test_hidden_df, "test_hidden")
