"""
test_data_integrity.py - Tests for dataset integrity and split validation.
Ensures no data leakage and validates MELD CSV structure.
"""
import os
import sys
import pytest
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "ml", "data", "raw", "MELD-master", "data", "MELD")

EXPECTED_COLUMNS = [
    "Sr No.", "Utterance", "Speaker", "Emotion", "Sentiment",
    "Dialogue_ID", "Utterance_ID", "Season", "Episode", "StartTime", "EndTime"
]

VALID_EMOTIONS = {"neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"}
VALID_SENTIMENTS = {"neutral", "positive", "negative"}


def load_csv(split: str):
    """Load a MELD split CSV file."""
    path = os.path.join(DATA_DIR, f"{split}_sent_emo.csv")
    if not os.path.exists(path):
        pytest.skip(f"CSV not found: {path}")
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


class TestDatasetStructure:
    """Tests for CSV structure and content validity."""

    @pytest.mark.parametrize("split", ["train", "dev", "test"])
    def test_csv_has_expected_columns(self, split):
        rows = load_csv(split)
        assert len(rows) > 0, f"{split} CSV is empty"
        columns = list(rows[0].keys())
        for col in EXPECTED_COLUMNS:
            assert col in columns, f"Missing column '{col}' in {split}"

    @pytest.mark.parametrize("split", ["train", "dev", "test"])
    def test_no_empty_utterances(self, split):
        rows = load_csv(split)
        empty = [r for r in rows if not r.get("Utterance", "").strip()]
        assert len(empty) == 0, f"Found {len(empty)} empty utterances in {split}"

    @pytest.mark.parametrize("split", ["train", "dev", "test"])
    def test_valid_emotion_labels(self, split):
        rows = load_csv(split)
        for r in rows:
            emotion = r.get("Emotion", "").strip().lower()
            assert emotion in VALID_EMOTIONS, (
                f"Invalid emotion '{emotion}' in {split}, row {r.get('Sr No.')}"
            )

    @pytest.mark.parametrize("split", ["train", "dev", "test"])
    def test_valid_sentiment_labels(self, split):
        rows = load_csv(split)
        for r in rows:
            sentiment = r.get("Sentiment", "").strip().lower()
            assert sentiment in VALID_SENTIMENTS, (
                f"Invalid sentiment '{sentiment}' in {split}"
            )

    def test_expected_split_sizes(self):
        """Verify split sizes match MELD paper (approximately)."""
        train = load_csv("train")
        dev = load_csv("dev")
        test = load_csv("test")
        assert 9900 <= len(train) <= 10100, f"Train size {len(train)} outside expected range"
        assert 1050 <= len(dev) <= 1200, f"Dev size {len(dev)} outside expected range"
        assert 2500 <= len(test) <= 2700, f"Test size {len(test)} outside expected range"


class TestNoDataLeakage:
    """Tests to ensure split separation and data integrity."""

    def test_split_dialogue_counts(self):
        """In MELD, dialogue IDs are 0-indexed per split partition."""
        train = load_csv("train")
        dev = load_csv("dev")
        test = load_csv("test")
        train_dias = {int(r["Dialogue_ID"]) for r in train}
        dev_dias = {int(r["Dialogue_ID"]) for r in dev}
        test_dias = {int(r["Dialogue_ID"]) for r in test}
        assert len(train_dias) == 1038
        assert len(dev_dias) == 114
        assert len(test_dias) == 280

    def test_no_utterance_text_overlap(self):
        """Check that identical utterance texts don't excessively overlap across splits."""
        train = load_csv("train")
        test = load_csv("test")
        train_texts = {r["Utterance"].strip().lower() for r in train}
        test_texts = {r["Utterance"].strip().lower() for r in test}
        # Common short phrases like 'Yeah' appear naturally across splits
        overlap = train_texts & test_texts
        overlap_pct = len(overlap) / len(test_texts) * 100 if test_texts else 0
        assert overlap_pct <= 20, f"Suspicious text overlap ({overlap_pct:.1f}%) between train and test"


class TestModelConfig:
    """Tests for configuration validity."""

    def test_config_loads(self):
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs", "config.json"
        )
        if not os.path.exists(config_path):
            pytest.skip("Config not found")
        import json
        with open(config_path) as f:
            config = json.load(f)

        assert "dataset" in config
        assert "model" in config
        assert "training" in config
        assert set(config["dataset"]["emotion_labels"]) == VALID_EMOTIONS
        assert config["training"]["seed"] == 42

    def test_emotion_label_map_complete(self):
        import json
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs", "config.json"
        )
        if not os.path.exists(config_path):
            pytest.skip("Config not found")
        with open(config_path) as f:
            config = json.load(f)
        label_map = config["dataset"]["emotion_label_map"]
        assert set(label_map.keys()) == VALID_EMOTIONS
        assert set(label_map.values()) == set(range(7))
