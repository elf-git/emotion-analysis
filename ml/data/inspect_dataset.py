"""
inspect_dataset.py - Quick dataset inspection utility.
Run standalone to print dataset statistics, or import for programmatic access.
"""
import os
import csv
import json
from collections import Counter
from typing import Dict, List


def inspect_csv(csv_path: str) -> Dict:
    """Inspect a single MELD CSV file and return statistics."""
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return {"error": "Empty file", "path": csv_path}

    emotions = Counter(r.get("Emotion", "").strip() for r in rows)
    sentiments = Counter(r.get("Sentiment", "").strip() for r in rows)
    speakers = Counter(r.get("Speaker", "").strip() for r in rows)
    dialogues = set(r.get("Dialogue_ID", "") for r in rows)

    utt_lengths = [
        len(r.get("Utterance", "").split())
        for r in rows
        if r.get("Utterance", "").strip()
    ]

    return {
        "path": csv_path,
        "columns": list(rows[0].keys()),
        "num_columns": len(rows[0].keys()),
        "total_utterances": len(rows),
        "unique_dialogues": len(dialogues),
        "unique_speakers": len(speakers),
        "top_speakers": dict(speakers.most_common(10)),
        "emotion_distribution": dict(emotions.most_common()),
        "sentiment_distribution": dict(sentiments.most_common()),
        "avg_utterance_words": round(sum(utt_lengths) / len(utt_lengths), 2) if utt_lengths else 0,
        "max_utterance_words": max(utt_lengths) if utt_lengths else 0,
        "min_utterance_words": min(utt_lengths) if utt_lengths else 0,
        "missing_utterances": sum(1 for r in rows if not r.get("Utterance", "").strip()),
        "missing_emotions": sum(1 for r in rows if not r.get("Emotion", "").strip()),
    }


def print_report(stats: Dict) -> None:
    """Pretty-print inspection report."""
    print(f"\n{'='*60}")
    print(f"  File: {stats['path']}")
    print(f"{'='*60}")
    for key, val in stats.items():
        if key == "path":
            continue
        if isinstance(val, dict):
            print(f"  {key}:")
            for k, v in val.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {val}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inspect MELD CSV files")
    parser.add_argument("csv_files", nargs="+", help="Path(s) to CSV files")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    all_stats = {}
    for path in args.csv_files:
        stats = inspect_csv(path)
        if args.json:
            all_stats[os.path.basename(path)] = stats
        else:
            print_report(stats)

    if args.json:
        print(json.dumps(all_stats, indent=2))
