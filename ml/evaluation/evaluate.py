"""
evaluate.py - Evaluation Pipeline
Comprehensive evaluation with per-class metrics, confusion matrices,
and comparison across modalities.
"""
import os
import json
import logging
import torch
import numpy as np
from typing import Dict, List, Optional
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EMOTION_LABELS = ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"]


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: List[str] = None,
) -> Dict:
    """
    Comprehensive evaluation of emotion predictions.

    Returns dict with accuracy, F1 scores, precision, recall,
    per-class metrics, and confusion matrix.
    """
    if label_names is None:
        label_names = EMOTION_LABELS

    results = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "classification_report": classification_report(
            y_true, y_pred, target_names=label_names, zero_division=0, output_dict=True
        ),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(len(label_names)))
        ).tolist(),
    }

    # Per-class F1
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    results["f1_per_class"] = {
        label_names[i]: float(f1_per_class[i]) for i in range(len(label_names))
    }

    return results


def compare_models(results_dir: str) -> Dict:
    """
    Compare results across all trained models.
    Reads *_results.json files from the results directory.
    """
    comparison = {}
    for fname in os.listdir(results_dir):
        if fname.endswith("_results.json"):
            model_name = fname.replace("_results.json", "")
            with open(os.path.join(results_dir, fname)) as f:
                data = json.load(f)
            test_metrics = data.get("test_metrics", {})
            comparison[model_name] = {
                "accuracy": test_metrics.get("accuracy", 0),
                "f1_weighted": test_metrics.get("f1_weighted", 0),
                "f1_macro": test_metrics.get("f1_macro", 0),
            }

    if comparison:
        logger.info("\n=== Model Comparison ===")
        logger.info(f"{'Model':<20} {'Accuracy':<12} {'F1 (Weighted)':<15} {'F1 (Macro)':<12}")
        logger.info("-" * 60)
        for model, metrics in sorted(comparison.items(), key=lambda x: x[1]["f1_weighted"], reverse=True):
            logger.info(
                f"{model:<20} {metrics['accuracy']:<12.4f} {metrics['f1_weighted']:<15.4f} {metrics['f1_macro']:<12.4f}"
            )

    return comparison


def generate_evaluation_report(results: Dict, output_path: str) -> str:
    """Generate a formatted evaluation report and save to file."""
    lines = []
    lines.append("=" * 60)
    lines.append("  EMOTION RECOGNITION EVALUATION REPORT")
    lines.append("=" * 60)
    lines.append(f"\n  Overall Accuracy : {results['accuracy']:.4f}")
    lines.append(f"  F1 (Weighted)    : {results['f1_weighted']:.4f}")
    lines.append(f"  F1 (Macro)       : {results['f1_macro']:.4f}")
    lines.append(f"  Precision        : {results['precision_weighted']:.4f}")
    lines.append(f"  Recall           : {results['recall_weighted']:.4f}")
    lines.append(f"\n  Per-Class F1 Scores:")
    for label, f1 in results.get("f1_per_class", {}).items():
        bar = "█" * int(f1 * 40)
        lines.append(f"    {label:<12} {f1:.4f}  {bar}")

    report_text = "\n".join(lines)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report_text)

    logger.info(report_text)
    return report_text


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="checkpoints", help="Dir with *_results.json")
    parser.add_argument("--output", default="outputs/evaluation_report.txt")
    args = parser.parse_args()

    compare_models(args.results_dir)
