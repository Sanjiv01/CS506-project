"""
Download the Spotify Playlists dataset from Kaggle.
Requires: pip install kagglehub
"""
import os
import shutil
import kagglehub

DATA_FILE = "spotify_dataset.csv"

def main():
    if os.path.exists(DATA_FILE):
        print(f"'{DATA_FILE}' already exists. Skipping download.")
        return

    print("Downloading Spotify Playlists dataset from Kaggle...")
    path = kagglehub.dataset_download("andrewmvd/spotify-playlists")
    print(f"Dataset downloaded to: {path}")

    # Find the CSV file in the downloaded directory
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.endswith(".csv"):
                src = os.path.join(root, f)
                shutil.copy2(src, DATA_FILE)
                print(f"Copied '{src}' -> '{DATA_FILE}'")
                return

    raise FileNotFoundError(f"No CSV file found in downloaded dataset at {path}")

if __name__ == "__main__":
    main()
