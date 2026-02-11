# Project Proposal

## Project Description
Music streaming platforms rely heavily on recommender systems to help users discover new songs. In this project, we aim to build a playlist-based music recommendation system that predicts which songs are likely to belong in a playlist, based on:
- Other songs already in the same playlist
- A user’s other playlists

We begin with collaborative filtering methods and, if time permits, extend the system with content-based features derived from song lyrics to build a hybrid recommender.

## Project Goals
Primary Goal
- Build a model that can predict missing songs in a playlist given partial playlist information.

Secondary Goal
- Leverage user's other playlists or lyrics of the song to boost accuracy of the model

Measurable Objectives
- Formulate playlist completion as a ranking problem
- Evaluate performance using :
	- Hit Rate @ K
	- Recall @ K
	- Mean Reciprocal Rank (MRR)

Baseline
- A very basic baseline is the model that recommends music based on popularity (e.g., recommending globally popular tracks)

## Data Collection
- Spotify Playlists Dataset from Kaggle (https://www.kaggle.com/datasets/andrewmvd/spotify-playlists)
	- Collection method : Download from kaggle
	- features in the dataset
		- user_id, artist_name, track_name, playlist_name
- Lyrics of the song
	- Collection method : scrape genius.com (https://genius.com/) or use a third-party scraping library (https://lyricsgenius.readthedocs.io/en/master/)
	- Could be potentially used to measure similarity between musics (for content-based filtering)

## Data Cleaning & Processing
- Remove duplicate tracks and playlists
- Normalize artist and track names
- Handle missing or inconsistent metadata
- Filter users and playlists with extremely small sizes (cold start problem)

## Modeling
- First try Collaborative Filtering methods
	- Neighborhood-based methods
	- Matrix Factorization
	- Autoencoder
- Then, if we have time, add Content-based methods to make the approach hybrid
	- Lyrics embeddings using large language models
	- cosine similarity between song embeddings

## Evaluation
- Hold out a portion of tracks from a portion of playlists for testing
- Train on remaining playlists and tracks
- Measure ranking metrics (Recall@K, MRR)
- Compare against baseline models

## Project timeline (8 week)
- Data Collection (1-2nd week)
    - Downloading Spotify Playlists Dataset from Kaggle
    - Scraping Lytrics in the dataset
    - Find ways to add more data
- Preliminary Modeling (3-4th week)
	- mostly focus on Collaborative Filtering methods (Neighborhood-based methods, Matrix Factorization, Autoendoer)
    - compare and analyze results
- Content-based Modeling & Develop a Hybrid model (Collaborative + Content-based) (5-6th week)
	- Based on the lyrics of the song, measure similarity between songs (from embeddings from Large language models)
	- Ensemble two methods to develop a Hybrid model
- Make report and presentation slides (7-8th week)

