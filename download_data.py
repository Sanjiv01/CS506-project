import argparse
import shutil
from pathlib import Path

import kagglehub


DATASET = "andrewmvd/spotify-playlists"
FILENAME = "spotify_dataset.csv"


def download(dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    src = Path(kagglehub.dataset_download(DATASET)) / FILENAME
    target = dest / FILENAME
    shutil.copy2(src, target)
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Download {DATASET} from Kaggle.")
    parser.add_argument("--dest", type=Path, default=Path("data"))
    args = parser.parse_args()
    path = download(args.dest)
    print(f"Saved to: {path}")
