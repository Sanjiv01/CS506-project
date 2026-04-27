# CS506 Final Project — Spotify Playlist Completion

## Overview

Given a partially-observed playlist, predict which songs are missing from it.
We frame this as a **ranking problem**: given a set of observed songs, return a ranked list of candidates and evaluate using HitRate, Recall, NDCG, and MRR at K.

---

## Results

| Model | HitRate@10 | Recall@10 | NDCG@10 | MRR@10 | Coverage |
|---|---|---|---|---|---|
| Popularity Baseline | 0.028 | 0.008 | 0.007 | 0.014 | 0.09% |
| Co-occurrence | 0.345 | 0.174 | 0.172 | 0.229 | 8.1% |
| **BM25 Co-occurrence** | **0.421** | **0.206** | **0.206** | **0.276** | **10.9%** |
| ALS (factors=32) | 0.099 | 0.028 | 0.029 | 0.054 | 3.9% |
| KNN (K=500) | 0.314 | 0.172 | 0.178 | 0.225 | 6.3% |

**BM25 is the best-performing model**, achieving a 15x improvement over the popularity baseline and a +22% relative gain over raw co-occurrence.

---

## Dataset

**Source:** [Spotify Playlists — Kaggle](https://www.kaggle.com/datasets/andrewmvd/spotify-playlists)
**Raw size:** 617K rows of `(user_id, artistname, trackname, playlistname)` tuples scraped from Spotify's public playlist API.
**After filtering:** ~337K rows / 86.6K songs / 9,296 playlists.

---

## How to Run

### On Google Colab (recommended)

1. Upload `main.ipynb` to Colab
2. Upload `spotify_dataset.csv` via the **Files panel** (left sidebar)
3. **Runtime → Run all**

The notebook auto-detects `/content/spotify_dataset.csv`. No other setup needed.

### Locally

```bash
pip install -r requirements.txt
jupyter notebook main.ipynb
```

Set `COLAB_MODE = False` in the config cell and ensure `spotify_dataset.csv` is in the same directory as the notebook.

> **Note:** `implicit` has been removed from dependencies. ALS is now implemented with `numpy` + `scipy` only, compatible with Python 3.14+.

---

## Configuration

Two key constants at the top of `main.ipynb` (cell 3):

| Constant | Default | Description |
|---|---|---|
| `SAMPLE_FRACTION` | `0.60` | Fraction of playlists to sample. Set to `1.0` for full dataset (~4x more data, better results) |
| `COLAB_MODE` | `True` | `True` = load from `/content/`, `False` = load from local `DATA_PATH` |

---

## Models

### 1. Popularity Baseline
Recommends globally most-popular songs to every user. Null model — any useful recommender must beat it.

### 2. Item-Item Co-occurrence
For each observed song, finds all training playlists containing it and aggregates co-occurring songs by raw count. Simple but strong on human-curated playlists.

### 3. BM25 Co-occurrence *(best model)*
Applies BM25 (k1=1.5, b=0.75) weighting to co-occurrence: penalizes songs from very large playlists, rewards songs supported by many independent small playlists. Removes the playlist-length bias in raw co-occurrence.

### 4. ALS Matrix Factorization
Vectorised Alternating Least Squares with SVD warm-start. Learns 32-dimensional latent item embeddings. Inference uses inverse-popularity weighted mean of observed song embeddings. Underperforms neighbourhood methods at this dataset scale (0.18% matrix density) but improves significantly on the full dataset.

- `factors=32`, `iterations=10`, `regularization=0.1`
- Pure `numpy` + `scipy` + `sklearn` — no compiled extensions

### 5. KNN Playlist Similarity
Finds the K=500 most cosine-similar training playlists to the query, aggregates their songs with softmax-sharpened neighbour weights and IDF-style query weighting. Includes a mild popularity penalty on output scores to improve catalog diversity.

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| **HitRate@K** | Fraction of test cases with at least 1 hidden song in top-K |
| **Recall@K** | Average fraction of hidden songs recovered in top-K |
| **NDCG@K** | Normalized Discounted Cumulative Gain — rewards hits ranked higher |
| **MRR@K** | Mean Reciprocal Rank of the first correct hit |

Evaluated at K = 5, 10, 20, 50. Test set: 20% of playlists with ~20% of each playlist's songs hidden.

---

## Hypotheses & Verdicts

| Hypothesis | Verdict |
|---|---|
| H1: Popularity is a weak baseline | Confirmed — 2.8% vs 42.1% HitRate@10 |
| H2: Co-occurrence captures genre context | Confirmed — 12x gain over popularity |
| H3: BM25 length-normalization helps | Confirmed — +22% relative gain over co-occurrence |
| H4: ALS underfits on sparse data | Confirmed — 9.9% HitRate, below co-occurrence |
| H5: KNN competitive on sparse data | Partially confirmed — beats ALS but below co-occurrence at this scale |

---

## Repository Structure

```
CS506-project/
├── main.ipynb          # Full pipeline: data loading, models, evaluation, visualizations
├── data_processing.py  # Standalone data pipeline utilities
├── recommendation.py   # Standalone recommendation functions
├── download_data.py    # Downloads dataset via kagglehub
├── requirements.txt    # Python dependencies
└── tests/              # Unit tests for preprocessing
```

---

## Dependencies

```
numpy, pandas, matplotlib, scikit-learn, scipy, rank-bm25, kagglehub, jupyter
```

No compiled extensions required — works on Python 3.14+ without a C++ compiler.
