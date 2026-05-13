"""Evaluation helpers for BLIP-2 VQA inference results."""

from __future__ import annotations

import collections
import json
import os

import numpy as np

from src.config import CFG


def load_results(path: str) -> list:
    """Load inference result dictionaries from a JSON file."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def evaluate_by_question_type(results: list) -> dict:
    """Compute VQA soft accuracy by inferred answer category."""
    color_set = {
        "red",
        "green",
        "blue",
        "yellow",
        "black",
        "white",
        "brown",
        "orange",
        "pink",
        "purple",
        "gray",
        "grey",
    }
    grouped_scores = collections.defaultdict(list)

    for item in results:
        answer = str(item.get("ground_truth", "")).strip().lower()
        if answer in {"yes", "no"}:
            category = "yes/no"
        elif answer.isdigit():
            category = "number"
        elif answer in color_set:
            category = "color"
        else:
            category = "other"
        grouped_scores[category].append(float(item.get("vqa_score", 0.0)))

    summary = {}
    for category in ("yes/no", "number", "color", "other"):
        scores = grouped_scores.get(category, [])
        summary[category] = {
            "count": len(scores),
            "accuracy": float(np.mean(scores) * 100.0) if scores else 0.0,
        }
    return summary


def evaluate_by_answer_length(results: list) -> dict:
    """Compute VQA soft accuracy grouped by ground-truth answer length."""
    grouped_scores = {"1": [], "2": [], "3+": []}

    for item in results:
        answer = str(item.get("ground_truth", "")).strip()
        word_count = len(answer.split()) if answer else 0
        if word_count <= 1:
            key = "1"
        elif word_count == 2:
            key = "2"
        else:
            key = "3+"
        grouped_scores[key].append(float(item.get("vqa_score", 0.0)))

    return {
        key: {
            "count": len(values),
            "accuracy": float(np.mean(values) * 100.0) if values else 0.0,
        }
        for key, values in grouped_scores.items()
    }


def baseline_comparison() -> dict:
    """Combine published baselines with classical and BLIP-2 project results."""
    comparison = {
        "BOW": 48.1,
        "LSTM": 53.3,
        "IMG+BOW": 55.9,
        "2-VIS+BLSTM": 58.4,
    }

    cnn_history_path = os.path.join(CFG.train.output_dir, "cnn_lstm_history.json")
    san_history_path = os.path.join(CFG.train.output_dir, "san_history.json")
    metrics_zero_path = os.path.join(CFG.train.output_dir, "metrics_zeroshot.json")
    metrics_tuned_path = os.path.join(CFG.train.output_dir, "metrics_finetuned.json")

    if os.path.exists(cnn_history_path):
        with open(cnn_history_path, "r", encoding="utf-8") as handle:
            cnn_history = json.load(handle)
        comparison["CNN+LSTM (ours)"] = float(cnn_history.get("best_val_acc", 0.0))
    else:
        comparison["CNN+LSTM (ours)"] = 0.0

    if os.path.exists(san_history_path):
        with open(san_history_path, "r", encoding="utf-8") as handle:
            san_history = json.load(handle)
        comparison["SAN (ours)"] = float(san_history.get("best_val_acc", 0.0))
    else:
        comparison["SAN (ours)"] = 0.0

    llava_zero = 0.0
    if os.path.exists(metrics_zero_path):
        with open(metrics_zero_path, "r", encoding="utf-8") as handle:
            metrics_zero = json.load(handle)
        llava_zero = float(metrics_zero.get("vqa_soft_accuracy", 0.0))
    comparison["BLIP-2 ZeroShot (ours)"] = llava_zero

    llava_tuned = 0.0
    if os.path.exists(metrics_tuned_path):
        with open(metrics_tuned_path, "r", encoding="utf-8") as handle:
            metrics_tuned = json.load(handle)
        llava_tuned = float(metrics_tuned.get("vqa_soft_accuracy", 0.0))
    comparison["BLIP-2 FineTuned (ours)"] = llava_tuned
    return comparison


def generate_evaluation_report(results, metrics) -> str:
    """Generate and save a text report with metrics and comparisons."""
    type_summary = evaluate_by_question_type(results)
    length_summary = evaluate_by_answer_length(results)
    baseline = baseline_comparison()

    lines = []
    lines.append("Three-Model VQA Evaluation Report")
    lines.append("=" * 36)
    lines.append("")
    lines.append("Model Family Summary")
    lines.append("-" * 36)
    lines.append("Families included: CNN+LSTM, SAN, and BLIP-2.")
    lines.append("")
    lines.append("Overall Metrics")
    lines.append("-" * 36)
    lines.append(f"{'Metric':24s}{'Value':>12s}")
    lines.append(f"{'Exact Match Accuracy (%)':24s}{metrics.get('exact_match_accuracy', 0.0):>12.2f}")
    lines.append(f"{'VQA Soft Accuracy (%)':24s}{metrics.get('vqa_soft_accuracy', 0.0):>12.2f}")
    lines.append(f"{'Avg Latency (ms)':24s}{metrics.get('avg_latency_ms', 0.0):>12.2f}")
    lines.append(f"{'Median Latency (ms)':24s}{metrics.get('median_latency_ms', 0.0):>12.2f}")
    lines.append(f"{'Total Samples':24s}{int(metrics.get('total_samples', 0)):>12d}")
    lines.append("")
    lines.append("Question-Type Breakdown")
    lines.append("-" * 36)
    lines.append(f"{'Type':12s}{'Count':>10s}{'VQA Soft Acc (%)':>18s}")
    for key, values in type_summary.items():
        lines.append(f"{key:12s}{values['count']:>10d}{values['accuracy']:>18.2f}")
    lines.append("")
    lines.append("Answer-Length Breakdown")
    lines.append("-" * 36)
    lines.append(f"{'Length':12s}{'Count':>10s}{'VQA Soft Acc (%)':>18s}")
    for key, values in length_summary.items():
        lines.append(f"{key:12s}{values['count']:>10d}{values['accuracy']:>18.2f}")
    lines.append("")
    lines.append("Baseline Comparison")
    lines.append("-" * 36)
    lines.append(f"{'Model':20s}{'Accuracy (%)':>16s}")
    for model_name, score in baseline.items():
        lines.append(f"{model_name:20s}{score:>16.2f}")

    report = "\n".join(lines)

    os.makedirs(CFG.train.output_dir, exist_ok=True)
    report_path = os.path.join(CFG.train.output_dir, "evaluation_report.txt")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(report)
    return report
