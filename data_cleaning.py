#%% Imports
from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/spotify_dataset.csv")
CLEAN_PATH = Path("data/spotify_dataset_clean.csv")

COLS = ["user_id", "artist_name", "track_name", "pl_name"]
PEEK = 5

#%%
##! Load data
def load_lenient(path):
    # Format is rigid: every line is "f1","f2","f3","f4". Splitting on the
    # literal "," separator bypasses CSV escaping rules and recovers rows
    # that pandas drops due to unescaped inner quotes.
    rows, bad = [], []
    with open(path, encoding="utf-8") as f:
        next(f)  # skip header
        for line in f:
            line = line.rstrip("\r\n")
            assert line.startswith('"') and line.endswith('"'), "line is not wrapped by quotes"
            line = line[1:-1]
            parts = line.split('","')
            if len(parts) == 4 :
                rows.append(parts)
            else:
                bad.append(parts)
    print(f"unrecoverable rows: {len(bad):,}")
    return pd.DataFrame(
        rows, columns=["user_id", "artist_name", "track_name", "pl_name"]
    )

df = load_lenient(DATA_PATH)
print(f"loaded: {df.shape}")
print("--- head(5) after load ---")
print(df.head(PEEK).to_string())

#%%
##! [DIAG] Drop nulls
print("--- null counts per column ---")
print(df.isna().sum())

#%%
##! [DIAG] Strip Whitespace
def has_padding(s):
    s = s.astype("string")
    stripped = s.str.strip()
    return s.notna() & (s != stripped)

pad_mask = pd.Series(False, index=df.index)
print("--- examples of leading/trailing whitespace ---")
for col in COLS:
    col_pad_mask = has_padding(df[col])
    print(f"[{col}] entries with leading/trailing whitespace: {col_pad_mask.sum():,}")
    for i in df[col_pad_mask][col].head(PEEK):
        print(f"  before: {repr(i)}  ->  after: {repr(i.strip())}")
    pad_mask |= col_pad_mask

print(f"rows with leading/trailing whitespace in any field: {pad_mask.sum():,}")

#%%
##! [FIX] Strip whitespace
for col in COLS:
    df[col] = df[col].astype("string").str.strip()


#%%
##! [DIAG] Drop empty strings - must be done after stripping whitespace
print("--- empty-string counts per column (post-strip) ---")
for col in COLS:
    print(f"[{col}] empty-string entries: {df[col].eq('').sum():,}")

#%%
##! [FIX] Treat empty strings as NA, drop rows with any missing key field
df[COLS] = df[COLS].replace({"": pd.NA})
missing_before = df[COLS].isna().any(axis=1).sum()
df = df.dropna(subset=COLS).reset_index(drop=True)
print(f"dropped {missing_before:,} rows with missing key fields -> {df.shape}")

#%%
##! [DIAG] Internal whitespace inconsistency (e.g., "The  Beatles")
ws_mask = pd.Series(False, index=df.index)
print("--- examples of multiple-space runs inside a field ---")
for col in ["artist_name", "track_name", "pl_name"]:
    col_ws_mask = df[col].str.contains(r"\s{2,}", regex=True, na=False)
    print(f"[{col}] entries with multiple-space runs: {col_ws_mask.sum():,}")
    import re as _re
    for i in df[col_ws_mask][col].head(PEEK):
        collapsed = _re.sub(r"\s+", " ", i)
        print(f"  before: {repr(i)}  ->  after: {repr(collapsed)}")
    ws_mask |= col_ws_mask

print(f"rows with multiple-space runs in any field: {ws_mask.sum():,}")

#%%
##! [FIX] Collapse internal whitespace
for col in ["artist_name", "track_name", "pl_name"]:
    df[col] = df[col].str.replace(r"\s+", " ", regex=True)

#%% [DIAG] Exact duplicate rows
dup_mask = df.duplicated(subset=COLS, keep=False)
print(f"rows participating in exact duplicates: {dup_mask.sum():,}")
print("--- examples of exact-duplicate rows ---")
print(df[dup_mask].sort_values(COLS).head(PEEK).to_string())

#%% [FIX] Drop exact duplicates
before = len(df)
df = df.drop_duplicates(subset=COLS).reset_index(drop=True)
print(f"dropped {before - len(df):,} exact duplicate rows -> {df.shape}")

#%% Save
CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(CLEAN_PATH, index=False)
print(f"saved cleaned data -> {CLEAN_PATH} {df.shape}")

#%% Final summary
print("--- final null counts ---")
print(df.isna().sum())
print("--- final head(5) ---")
print(df.head().to_string())
