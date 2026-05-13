"""Visualization helpers for BLIP-2 VQA evaluation artifacts."""

from __future__ import annotations

import json
import os
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.config import CFG
from src.evaluate import evaluate_by_question_type

DARK_BG = "#0f0f1a"
ACCENT = "#00d4aa"
RED = "#ff6b6b"
BLUE = "#4e9af1"


def plot_baseline_comparison(comparison_dict):
    """Create and save a horizontal baseline comparison bar chart."""
    os.makedirs(CFG.train.output_dir, exist_ok=True)
    labels = list(comparison_dict.keys())
    scores = [comparison_dict[name] for name in labels]
    colors = [ACCENT if "ours" in name.lower() else "#8b949e" for name in labels]

    fig_height = max(6, 0.7 * len(labels))
    fig, ax = plt.subplots(figsize=(10, fig_height), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)

    y = np.arange(len(labels))
    bars = ax.barh(y, scores, color=colors, edgecolor="#30363d")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, color="#e6edf3")
    ax.set_xlabel("Accuracy (%)", color="#e6edf3")
    ax.set_title("VQA Baseline Comparison", color="#f0f6fc", pad=12)
    ax.tick_params(axis="x", colors="#c9d1d9")
    ax.grid(axis="x", linestyle="--", alpha=0.25, color="#30363d")
    x_max = max(scores) if scores else 100.0
    ax.set_xlim(0, max(100.0, x_max + 10.0))

    for bar, score in zip(bars, scores):
        ax.text(score + 0.5, bar.get_y() + bar.get_height() / 2, f"{score:.1f}", va="center", color="#e6edf3")

    output_path = os.path.join(CFG.train.output_dir, "baseline_comparison.png")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


def plot_question_type_accuracy(type_results: dict):
    """Create and save grouped bars for accuracy and sample share by type."""
    os.makedirs(CFG.train.output_dir, exist_ok=True)
    categories = list(type_results.keys())
    counts = np.array([type_results[key]["count"] for key in categories], dtype=float)
    accuracies = np.array([type_results[key]["accuracy"] for key in categories], dtype=float)
    count_share = (counts / counts.sum() * 100.0) if counts.sum() > 0 else np.zeros_like(counts)

    x = np.arange(len(categories))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.bar(x - width / 2, accuracies, width=width, color=BLUE, label="VQA Soft Accuracy (%)", edgecolor="#30363d")
    ax.bar(x + width / 2, count_share, width=width, color=ACCENT, label="Sample Share (%)", edgecolor="#30363d")

    ax.set_xticks(x)
    ax.set_xticklabels(categories, color="#e6edf3")
    ax.set_ylabel("Percent (%)", color="#e6edf3")
    ax.set_title("Question-Type Performance Breakdown", color="#f0f6fc", pad=12)
    ax.tick_params(colors="#c9d1d9")
    ax.grid(axis="y", linestyle="--", alpha=0.25, color="#30363d")
    ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#f0f6fc")

    output_path = os.path.join(CFG.train.output_dir, "question_type_accuracy.png")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


def _to_pil_image(image_payload):
    """Convert supported image payloads to RGB PIL images."""
    if isinstance(image_payload, Image.Image):
        return image_payload.convert("RGB")
    if isinstance(image_payload, str) and os.path.exists(image_payload):
        return Image.open(image_payload).convert("RGB")
    if isinstance(image_payload, dict):
        if image_payload.get("bytes") is not None:
            return Image.open(BytesIO(image_payload["bytes"])).convert("RGB")
        if image_payload.get("path") and os.path.exists(image_payload["path"]):
            return Image.open(image_payload["path"]).convert("RGB")
    return None


def plot_sample_predictions(results: list, processor, n=8):
    """Create and save a 2x4 sample prediction grid with correctness borders."""
    _ = processor
    os.makedirs(CFG.train.output_dir, exist_ok=True)
    output_path = os.path.join(CFG.train.output_dir, "sample_predictions.png")

    selected = results[: max(0, min(n, 8))]
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), facecolor=DARK_BG)
    axes = axes.flatten()

    for idx, ax in enumerate(axes):
        ax.set_facecolor(DARK_BG)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#30363d")
            spine.set_linewidth(2.0)

        if idx >= len(selected):
            ax.text(0.5, 0.5, "No Sample", ha="center", va="center", color="#8b949e", fontsize=11)
            continue

        item = selected[idx]
        image = _to_pil_image(item.get("image"))
        if image is not None:
            ax.imshow(np.asarray(image))
        else:
            ax.text(0.5, 0.55, "Image unavailable", ha="center", va="center", color="#8b949e", fontsize=10)
            ax.text(0.5, 0.40, "in inference results", ha="center", va="center", color="#8b949e", fontsize=10)

        question = str(item.get("question", ""))[:80]
        predicted = str(item.get("predicted", ""))
        ground_truth = str(item.get("ground_truth", ""))
        is_correct = bool(item.get("is_exact_match", predicted == ground_truth))

        title = f"Q: {question}\nP: {predicted} | GT: {ground_truth}"
        ax.set_title(title, color="#e6edf3", fontsize=9, pad=8)

        border_color = ACCENT if is_correct else RED
        for spine in ax.spines.values():
            spine.set_color(border_color)
            spine.set_linewidth(3.0)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


def generate_all_plots(results, metrics, comparison):
    """Generate all requested plots and print saved file paths."""
    _ = metrics
    os.makedirs(CFG.train.output_dir, exist_ok=True)

    baseline_path = plot_baseline_comparison(comparison)
    type_results = evaluate_by_question_type(results)
    type_path = plot_question_type_accuracy(type_results)
    sample_path = plot_sample_predictions(results, processor=None, n=8)

    metadata_path = os.path.join(CFG.train.output_dir, "plot_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "baseline_comparison": baseline_path,
                "question_type_accuracy": type_path,
                "sample_predictions": sample_path,
            },
            handle,
            indent=2,
        )

    print(f"Saved: {baseline_path}")
    print(f"Saved: {type_path}")
    print(f"Saved: {sample_path}")
    print(f"Saved: {metadata_path}")
