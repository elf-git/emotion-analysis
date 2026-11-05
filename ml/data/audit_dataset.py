"""
MELD Dataset Audit Script
Analyzes annotation CSVs for columns, splits, emotion distribution,
missing values, and data quality checks.
"""
import csv
import os
import json

BASE = os.path.join(os.path.dirname(__file__), "raw", "MELD-master", "data", "MELD")
FILES = {
    "train": "train_sent_emo.csv",
    "dev": "dev_sent_emo.csv",
    "test": "test_sent_emo.csv",
}

results = {}

for split_name, fname in FILES.items():
    path = os.path.join(BASE, fname)
    if not os.path.exists(path):
        print(f"[MISSING] {path}")
        continue

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    columns = list(rows[0].keys()) if rows else []
    total = len(rows)

    emotions = {}
    sentiments = {}
    missing_utt = 0
    missing_emo = 0
    empty_speaker = 0
    dialogues = set()
    speakers = set()
    utt_lengths = []

    for r in rows:
        e = r.get("Emotion", "").strip()
        s = r.get("Sentiment", "").strip()
        utt = r.get("Utterance", "").strip()
        spk = r.get("Speaker", "").strip()

        emotions[e] = emotions.get(e, 0) + 1
        sentiments[s] = sentiments.get(s, 0) + 1

        if not utt:
            missing_utt += 1
        else:
            utt_lengths.append(len(utt.split()))

        if not e:
            missing_emo += 1
        if not spk:
            empty_speaker += 1

        dialogues.add(r.get("Dialogue_ID", ""))
        speakers.add(spk)

    avg_len = sum(utt_lengths) / len(utt_lengths) if utt_lengths else 0
    max_len = max(utt_lengths) if utt_lengths else 0

    info = {
        "file": fname,
        "columns": columns,
        "total_utterances": total,
        "unique_dialogues": len(dialogues),
        "unique_speakers": len(speakers),
        "missing_utterances": missing_utt,
        "missing_emotions": missing_emo,
        "empty_speakers": empty_speaker,
        "avg_utterance_words": round(avg_len, 2),
        "max_utterance_words": max_len,
        "emotion_distribution": dict(sorted(emotions.items())),
        "sentiment_distribution": dict(sorted(sentiments.items())),
    }
    results[split_name] = info

    print(f"\n{'='*60}")
    print(f"  {split_name.upper()} SPLIT  -  {fname}")
    print(f"{'='*60}")
    print(f"  Columns ({len(columns)}): {columns}")
    print(f"  Total utterances : {total}")
    print(f"  Unique dialogues : {len(dialogues)}")
    print(f"  Unique speakers  : {len(speakers)}")
    print(f"  Missing utts     : {missing_utt}")
    print(f"  Missing emotions : {missing_emo}")
    print(f"  Empty speakers   : {empty_speaker}")
    print(f"  Avg words/utt    : {avg_len:.2f}")
    print(f"  Max words/utt    : {max_len}")
    print(f"  Emotions         : {dict(sorted(emotions.items()))}")
    print(f"  Sentiments       : {dict(sorted(sentiments.items()))}")

# Check for raw video/audio files
print(f"\n{'='*60}")
print("  RAW MEDIA & FEATURE FILES CHECK")
print(f"{'='*60}")

raw_base = os.path.join(os.path.dirname(__file__), "raw", "MELD-master")
for check_dir in ["data/pickles", "data/MELD"]:
    full = os.path.join(raw_base, check_dir)
    if os.path.exists(full):
        files = os.listdir(full)
        print(f"  {check_dir}: {files}")
    else:
        print(f"  {check_dir}: [NOT FOUND]")

# Check for .mp4, .wav, .tar.gz anywhere under raw
media_exts = {".mp4", ".wav", ".avi", ".tar", ".gz", ".pkl", ".pickle"}
media_found = []
for root, dirs, files in os.walk(raw_base):
    for f in files:
        if any(f.endswith(ext) for ext in media_exts):
            media_found.append(os.path.join(root, f))

if media_found:
    print(f"  Media/feature files found: {len(media_found)}")
    for m in media_found[:10]:
        print(f"    - {m}")
else:
    print("  No raw media files (.mp4/.wav/.avi) found locally.")
    print("  No pre-extracted features (.pkl/.pickle) found locally.")
    print("  >> Audio/video data must be downloaded on Kaggle at runtime.")

# Save audit results as JSON
out_path = os.path.join(os.path.dirname(__file__), "audit_results.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n  Audit results saved to: {out_path}")
