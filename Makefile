.PHONY: install data clean-data process explore models summarize viz all run test clean

PYTHON ?= python

# --- Setup -------------------------------------------------------------------

install:
	pip install -r requirements.txt

# --- Data pipeline -----------------------------------------------------------

# Raw download (~1.18 GB). File target avoids re-downloading.
data data/spotify_dataset.csv:
	$(PYTHON) download_data.py --dest data

data/spotify_dataset_clean.csv: data/spotify_dataset.csv data_cleaning.py
	$(PYTHON) data_cleaning.py

clean-data: data/spotify_dataset_clean.csv

data/spotify/playlists_train.csv: data/spotify_dataset_clean.csv data_processing.py
	$(PYTHON) data_processing.py

process: data/spotify/playlists_train.csv

explore: data/spotify_dataset_clean.csv
	$(PYTHON) explore_data.py

# --- Models ------------------------------------------------------------------

models: process
	$(PYTHON) rec_pop.py
	$(PYTHON) rec_cooc.py
	$(PYTHON) rec_bm25.py
	$(PYTHON) rec_als.py
	$(PYTHON) rec_knn.py
	$(PYTHON) rec_knn_advanced.py

# --- Reporting ---------------------------------------------------------------

summarize:
	$(PYTHON) summarize_results.py

viz: summarize
	$(PYTHON) make_interactive_viz.py

# --- End-to-end --------------------------------------------------------------

all: install process explore models summarize viz

# Backwards-compatible: legacy `make run` still executes the preprocessing notebook.
run: data/spotify_dataset.csv
	jupyter nbconvert --to notebook --execute preprocessing.ipynb --output preprocessing_output.ipynb

# --- Tests -------------------------------------------------------------------

test:
	$(PYTHON) -m pytest tests/ -v

# --- Cleanup -----------------------------------------------------------------
# Removes derived artifacts but preserves the raw 1.18 GB Kaggle download.

clean:
	rm -f preprocessing_output.ipynb
	rm -rf __pycache__ .pytest_cache tests/__pycache__
	rm -rf results
	rm -rf data/spotify
	rm -f data/spotify_dataset_clean.csv
