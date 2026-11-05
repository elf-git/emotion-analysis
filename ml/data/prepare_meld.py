"""
prepare_meld.py - MELD Dataset Preparation
Downloads and prepares the MELD dataset for training.
Designed to run on Kaggle with GPU runtime.
"""
import os
import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# MELD GitHub raw CSV URLs
MELD_CSV_URLS = {
    "train": "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/train_sent_emo.csv",
    "dev": "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/dev_sent_emo.csv",
    "test": "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/test_sent_emo.csv",
}

EMOTION_LABEL_MAP = {
    "neutral": 0, "surprise": 1, "fear": 2, "sadness": 3,
    "joy": 4, "disgust": 5, "anger": 6,
}
SENTIMENT_LABEL_MAP = {"neutral": 0, "positive": 1, "negative": 2}


def download_csv(url: str, dest_path: str) -> str:
    """Download a CSV file from URL if not already present."""
    if os.path.exists(dest_path):
        logger.info(f"Already exists: {dest_path}")
        return dest_path

    import urllib.request
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    logger.info(f"Downloading {url} -> {dest_path}")
    urllib.request.urlretrieve(url, dest_path)
    return dest_path


def load_meld_csv(csv_path: str) -> List[Dict]:
    """Load a MELD CSV and return list of utterance dicts."""
    utterances = []
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            utterances.append({
                "sr_no": row.get("Sr No.", ""),
                "utterance": row.get("Utterance", "").strip(),
                "speaker": row.get("Speaker", "").strip(),
                "emotion": row.get("Emotion", "").strip().lower(),
                "sentiment": row.get("Sentiment", "").strip().lower(),
                "dialogue_id": int(row.get("Dialogue_ID", 0)),
                "utterance_id": int(row.get("Utterance_ID", 0)),
                "season": row.get("Season", ""),
                "episode": row.get("Episode", ""),
                "start_time": row.get("StartTime", ""),
                "end_time": row.get("EndTime", ""),
            })
    return utterances


def find_local_csv(data_dir: str, fname: str) -> Optional[str]:
    """Search possible locations for an existing CSV file."""
    candidates = [
        os.path.join(data_dir, fname),
        os.path.join(data_dir, "MELD-master", "data", "MELD", fname),
        os.path.join(data_dir, "data", "MELD", fname),
        os.path.join(os.path.dirname(__file__), "raw", "MELD-master", "data", "MELD", fname),
        os.path.join("/kaggle/input/meld-dataset/data/MELD", fname),
        os.path.join("/kaggle/input/meld-dataset", fname),
        os.path.join("/kaggle/input/meld", fname),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def prepare_dataset(
    data_dir: str,
    github_repo_url: Optional[str] = None,
    use_local: bool = False,
) -> Dict[str, List[Dict]]:
    """
    Download (or load locally) the MELD CSVs and return split data.

    Args:
        data_dir: Directory to store/find CSV files.
        github_repo_url: Override GitHub repo URL for CSVs.
        use_local: If True, only search locally and fail if not found.

    Returns:
        Dict with keys 'train', 'dev', 'test', each a list of utterance dicts.
    """
    splits = {}

    for split_name, default_url in MELD_CSV_URLS.items():
        fname = f"{split_name}_sent_emo.csv"
        local_path = find_local_csv(data_dir, fname)

        if local_path is None:
            if use_local:
                raise FileNotFoundError(
                    f"Local CSV not found: {fname} in {data_dir}. "
                    f"Set use_local=False to download from URL."
                )
            dest_path = os.path.join(data_dir, fname)
            url = default_url
            if github_repo_url:
                url = f"{github_repo_url.rstrip('/')}/raw/master/data/MELD/{fname}"
            local_path = download_csv(url, dest_path)

        utterances = load_meld_csv(local_path)
        splits[split_name] = utterances
        logger.info(f"Loaded {split_name} from {local_path}: {len(utterances)} utterances")

    # Verify split dialogue counts (MELD dialogue IDs are 0-indexed within each split partition)
    train_dias = {u["dialogue_id"] for u in splits["train"]}
    dev_dias = {u["dialogue_id"] for u in splits["dev"]}
    test_dias = {u["dialogue_id"] for u in splits["test"]}
    logger.info(
        f"MELD split partitions verified: train={len(train_dias)} dialogues, "
        f"dev={len(dev_dias)} dialogues, test={len(test_dias)} dialogues."
    )

    return splits


def get_video_clip_path(
    split: str,
    dialogue_id: int,
    utterance_id: int,
    video_dir: str,
) -> str:
    """
    Construct the expected path to a video clip.
    MELD video clips are named: dia{dialogue_id}_utt{utterance_id}.mp4
    """
    fname = f"dia{dialogue_id}_utt{utterance_id}.mp4"
    return os.path.join(video_dir, split, fname)


def create_smoke_test_subset(
    splits: Dict[str, List[Dict]],
    n_samples: int = 200,
    seed: int = 42,
) -> Dict[str, List[Dict]]:
    """Create a small subset for smoke testing."""
    import random
    rng = random.Random(seed)

    smoke = {}
    for split_name, utterances in splits.items():
        n = min(n_samples, len(utterances))
        smoke[split_name] = rng.sample(utterances, n)
        logger.info(f"Smoke test {split_name}: {n} samples")

    return smoke


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare MELD dataset")
    parser.add_argument("--data-dir", default="ml/data/processed", help="Output directory")
    parser.add_argument("--github-url", default=None, help="Override GitHub repo URL")
    parser.add_argument("--local", action="store_true", help="Use local files only")
    parser.add_argument("--smoke-test", type=int, default=0, help="Create smoke test subset with N samples")
    args = parser.parse_args()

    splits = prepare_dataset(args.data_dir, args.github_url, args.local)

    for name, data in splits.items():
        out_path = os.path.join(args.data_dir, f"{name}_processed.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved {name} -> {out_path}")

    if args.smoke_test > 0:
        smoke = create_smoke_test_subset(splits, args.smoke_test)
        for name, data in smoke.items():
            out_path = os.path.join(args.data_dir, f"{name}_smoke.json")
            with open(out_path, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved smoke {name} -> {out_path}")
