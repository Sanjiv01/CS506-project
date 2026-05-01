#%% Imports
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CLEAN_PATH = Path("data/spotify_dataset_clean.csv")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

#%% Load cleaned data
# keep_default_na=False: the dataset contains literal strings like "None",
# "N/A", "NULL", "null" as real artist / track / playlist names. Without
# this flag pandas would silently coerce them to NaN.
df = pd.read_csv(CLEAN_PATH, dtype=str, keep_default_na=False)
print(f"loaded: {df.shape}")
print(df.dtypes)
print("--- head(5) ---")
print(df.head().to_string())

#%% Sanity checks (should all be zero / clean)
print("--- null counts ---")
print(df.isna().sum())
dup = df.duplicated().sum()
print(f"exact duplicate rows: {dup:,}")

#%% Cardinalities (using correct identifiers)
# - user_id is a hash and stands alone as a user identifier
# - a playlist is identified by (user_id, pl_name): two users may both have a
#   playlist called "chill" but those are different playlists
# - a track is identified by (artist_name, track_name): two distinct artists
#   can have songs with the same title (e.g. covers, common titles)
n_users      = df["user_id"].nunique()
n_artists    = df["artist_name"].nunique()
n_track_titles = df["track_name"].nunique()
n_pl_names   = df["pl_name"].nunique()
n_tracks     = df.groupby(["artist_name", "track_name"]).ngroups
n_playlists  = df.groupby(["user_id", "pl_name"]).ngroups

print("--- cardinalities ---")
print(f"unique users:                          {n_users:,}")
print(f"unique artists (by artist_name):       {n_artists:,}")
print(f"unique track *titles* (by name only):  {n_track_titles:,}")
print(f"unique tracks (artist + track name):   {n_tracks:,}")
print(f"unique playlist *names*:               {n_pl_names:,}")
print(f"unique playlists (user + pl_name):     {n_playlists:,}")

# Demonstrate why title-only / name-only counts mislead.
print("--- example: track titles attached to multiple artists ---")
title_artists = (
    df.groupby("track_name")["artist_name"].nunique().sort_values(ascending=False)
)
print(title_artists.head(5).to_string())

print("--- example: playlist names shared across many users ---")
plname_users = (
    df.groupby("pl_name")["user_id"].nunique().sort_values(ascending=False)
)
print(plname_users.head(5).to_string())

#%% Top artists (by row count)
top_artists = df["artist_name"].value_counts().head(20)
print("--- top 20 artists by row count ---")
print(top_artists.to_string())

fig, ax = plt.subplots(figsize=(8, 6))
top_artists[::-1].plot(kind="barh", ax=ax)
ax.set_title("Top 20 artists by row count")
ax.set_xlabel("rows")
fig.tight_layout()
fig.savefig(FIG_DIR / "top_artists.png", dpi=120)
plt.close(fig)

#%% Top tracks (artist + track name pair)
top_tracks = (
    df.groupby(["artist_name", "track_name"])
      .size()
      .sort_values(ascending=False)
      .head(20)
)
print("--- top 20 tracks (artist, track) by row count ---")
print(top_tracks.to_string())

#%% Top playlist names (which names many users independently use)
top_pl_names = df.groupby("pl_name")["user_id"].nunique().sort_values(ascending=False).head(20)
print("--- top 20 playlist names by number of distinct users using them ---")
print(top_pl_names.to_string())

#%% Per-playlist size distribution
playlist_size = df.groupby(["user_id", "pl_name"]).size()
print("--- playlist size (tracks per playlist) ---")
print(playlist_size.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_string())
print(f"min: {playlist_size.min():,}    max: {playlist_size.max():,}")
print(f"playlists with exactly 1 track:           {(playlist_size == 1).sum():,}")
print(f"playlists with >= 10 tracks:              {(playlist_size >= 10).sum():,}")
print(f"playlists with >= 100 tracks:             {(playlist_size >= 100).sum():,}")
print(f"playlists with >= 1000 tracks:            {(playlist_size >= 1000).sum():,}")

fig, ax = plt.subplots(figsize=(8, 4))
playlist_size.plot(kind="hist", bins=100, log=True, ax=ax)
ax.set_title("Tracks per playlist (log y)")
ax.set_xlabel("tracks per playlist")
fig.tight_layout()
fig.savefig(FIG_DIR / "playlist_size.png", dpi=120)
plt.close(fig)

#%% Playlists per user
user_playlist_counts = df.groupby("user_id")["pl_name"].nunique()
print("--- playlists per user ---")
print(user_playlist_counts.describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]).to_string())
print(f"min: {user_playlist_counts.min():,}    max: {user_playlist_counts.max():,}")

fig, ax = plt.subplots(figsize=(8, 4))
user_playlist_counts.plot(kind="hist", bins=50, log=True, ax=ax)
ax.set_title("Playlists per user (log y)")
ax.set_xlabel("playlists per user")
fig.tight_layout()
fig.savefig(FIG_DIR / "playlists_per_user.png", dpi=120)
plt.close(fig)

#%% Tracks per user (rows per user)
user_track_counts = df.groupby("user_id").size()
print("--- tracks per user (rows per user) ---")
print(user_track_counts.describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]).to_string())

#%% Artists per playlist + concentration (is a playlist focused on 1-2 artists?)
artists_per_pl = df.groupby(["user_id", "pl_name"])["artist_name"].nunique()
print("--- distinct artists per playlist ---")
print(artists_per_pl.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95]).to_string())

# Concentration: share of the most frequent artist within each playlist.
artist_share = (
    df.groupby(["user_id", "pl_name", "artist_name"]).size()
      .groupby(level=[0, 1]).apply(lambda s: s.max() / s.sum())
)
print("--- top-artist share per playlist (max artist count / playlist size) ---")
print(artist_share.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95]).to_string())
print(f"playlists where top artist >= 50% of tracks: {(artist_share >= 0.5).sum():,} "
      f"({(artist_share >= 0.5).mean():.1%})")
print(f"playlists where top artist >= 80% of tracks: {(artist_share >= 0.8).sum():,} "
      f"({(artist_share >= 0.8).mean():.1%})")
print(f"playlists where top artist == 100% (single-artist): {(artist_share == 1.0).sum():,} "
      f"({(artist_share == 1.0).mean():.1%})")

#%% Sample a few playlists to eyeball
sample_keys = (
    df.groupby(["user_id", "pl_name"]).size().sample(3, random_state=0).index
)
print("--- 3 sample playlists ---")
for uid, pl in sample_keys:
    print(f"\n=== {uid} :: {pl} ===")
    print(df[(df["user_id"] == uid) & (df["pl_name"] == pl)][
        ["artist_name", "track_name"]
    ].head(15).to_string())

#%% Recommendation signal: do co-occurring tracks look related?
# Build a track id (artist, track), then for each playlist record the set of
# tracks. For a few seed tracks, list the tracks that co-occur most often in
# the same playlists. If the recommendation hypothesis holds, the top
# co-occurring tracks should look semantically related (same artist, same
# genre, similar style).
df["_track"] = df["artist_name"] + " :: " + df["track_name"]
df["_pl"] = df["user_id"] + " :: " + df["pl_name"]

# Track popularity (in how many distinct playlists each track appears).
track_pl_count = df.groupby("_track")["_pl"].nunique().sort_values(ascending=False)
print("--- track popularity: number of distinct playlists containing the track ---")
print(track_pl_count.head(10).to_string())

def cooccurring(seed, k=10, min_cooc=5):
    """Return the top-k tracks most frequently sharing a playlist with `seed`."""
    seed_pls = set(df.loc[df["_track"] == seed, "_pl"].unique())
    if not seed_pls:
        return None
    sub = df[df["_pl"].isin(seed_pls) & (df["_track"] != seed)]
    counts = sub["_track"].value_counts()
    return counts[counts >= min_cooc].head(k)

# Pick a small panel of seeds spanning genres so the result is interpretable.
seeds = [
    "Coldplay :: The Scientist",
    "Daft Punk :: Get Lucky - Radio Edit",
    "Eminem :: Lose Yourself",
    "Bob Marley & The Wailers :: No Woman, No Cry",
]
print("--- co-occurrence neighbors for a small panel of seed tracks ---")
for seed in seeds:
    print(f"\nSeed: {seed}    (in {track_pl_count.get(seed, 0):,} playlists)")
    neigh = cooccurring(seed, k=10)
    if neigh is None or len(neigh) == 0:
        print("  (no neighbors found)")
        continue
    print(neigh.to_string())

#%% Also: distribution of "how many playlists does a typical track live in?"
print("--- playlists-per-track distribution ---")
print(track_pl_count.describe(percentiles=[0.5, 0.9, 0.95, 0.99, 0.999]).to_string())
print(f"tracks appearing in only 1 playlist: {(track_pl_count == 1).sum():,} "
      f"({(track_pl_count == 1).mean():.1%})")
print(f"tracks appearing in >= 10 playlists: {(track_pl_count >= 10).sum():,}")
print(f"tracks appearing in >= 100 playlists: {(track_pl_count >= 100).sum():,}")

print("done.")
