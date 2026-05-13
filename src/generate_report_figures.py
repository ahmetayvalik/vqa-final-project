"""Generate final report figures for the VQA project."""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import CFG

DARK_BG = "#0f0f1a"
FG = "#f8fafc"
GRID = "#475569"
ACCENT = "#00d4aa"
BLUE = "#4e9af1"
ORANGE = "#f59e0b"
GRAY = "#9ca3af"
RED = "#ff6b6b"


def _setup_ax(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    """Apply shared dark-theme styling to an axis."""
    ax.set_facecolor(DARK_BG)
    ax.set_title(title, color=FG, fontsize=15, pad=14)
    ax.set_xlabel(xlabel, color=FG)
    ax.set_ylabel(ylabel, color=FG)
    ax.tick_params(colors=FG)
    ax.grid(alpha=0.25, color=GRID, linestyle="--")
    for spine in ax.spines.values():
        spine.set_color(GRID)


def figure_model_comparison() -> str:
    """Create horizontal model-comparison bar chart."""
    models = ["Question-Only", "SAN", "BLIP-2 Zero", "CNN+LSTM"]
    scores = [18.35, 35.80, 33.33, 41.20]
    colors = [GRAY, BLUE, ACCENT, ORANGE]

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=DARK_BG)
    bars = ax.barh(models, scores, color=colors)
    _setup_ax(ax, "Figure 1: Model Comparison", xlabel="Validation Accuracy (%)")
    ax.set_xlim(0, 50)

    for bar, score in zip(bars, scores):
        ax.text(score + 0.8, bar.get_y() + bar.get_height() / 2, f"{score:.2f}%", color=FG, va="center")

    fig.tight_layout()
    path = os.path.join(CFG.train.output_dir, "fig1_comparison.png")
    fig.savefig(path, dpi=220, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def figure_training_curve() -> str:
    """Create CNN+LSTM validation-accuracy training curve."""
    epochs = [1, 2, 3, 4, 5, 6]
    val_acc = [34.75, 37.75, 39.10, 38.90, 37.70, 37.70]
    best_epoch = 3
    best_acc = val_acc[best_epoch - 1]

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=DARK_BG)
    ax.plot(epochs, val_acc, color=BLUE, linewidth=2.5, marker="o", markersize=7)
    ax.scatter([best_epoch], [best_acc], marker="*", s=300, color=ORANGE, edgecolor=FG, zorder=5)
    ax.annotate(
        f"Best epoch {best_epoch}: {best_acc:.2f}%",
        xy=(best_epoch, best_acc),
        xytext=(best_epoch + 0.4, best_acc + 0.5),
        color=FG,
        arrowprops={"arrowstyle": "->", "color": ORANGE},
    )
    _setup_ax(ax, "Figure 2: CNN+LSTM Training Curve", xlabel="Epoch", ylabel="Validation Accuracy (%)")
    ax.set_xticks(epochs)
    ax.set_ylim(33, 41)

    fig.tight_layout()
    path = os.path.join(CFG.train.output_dir, "fig2_training_curve.png")
    fig.savefig(path, dpi=220, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def figure_data_vs_accuracy() -> str:
    """Create data-size versus accuracy figure."""
    sample_labels = ["5K", "20K", "50K"]
    x = [5, 20, 50]
    acc = [41.20, 39.10, 35.89]

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=DARK_BG)
    ax.plot(x, acc, color=RED, linewidth=2.5, marker="o", markersize=8)
    _setup_ax(
        ax,
        "Figure 3: Data Size vs Accuracy",
        xlabel="Training Samples",
        ylabel="Best Validation Accuracy (%)",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(sample_labels)
    ax.set_ylim(34, 43)
    ax.annotate(
        "More data did not improve accuracy\nwithout hyperparameter tuning",
        xy=(50, 35.89),
        xytext=(16, 34.8),
        color=FG,
        arrowprops={"arrowstyle": "->", "color": RED},
    )

    for xi, yi in zip(x, acc):
        ax.text(xi, yi + 0.25, f"{yi:.2f}%", color=FG, ha="center")

    fig.tight_layout()
    path = os.path.join(CFG.train.output_dir, "fig3_data_vs_acc.png")
    fig.savefig(path, dpi=220, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def figure_vision_contribution() -> str:
    """Create question-only versus vision-language contribution bar chart."""
    labels = ["Question-Only", "CNN+LSTM"]
    scores = [18.35, 41.20]
    gap = scores[1] - scores[0]

    fig, ax = plt.subplots(figsize=(8, 6), facecolor=DARK_BG)
    bars = ax.bar(labels, scores, color=[GRAY, ORANGE], width=0.55)
    _setup_ax(ax, "Figure 4: Vision Contribution", ylabel="Validation Accuracy (%)")
    ax.set_ylim(0, 50)

    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, score + 0.8, f"{score:.2f}%", color=FG, ha="center")

    ax.annotate(
        f"+{gap:.2f} points from visual features",
        xy=(1, scores[1]),
        xytext=(0.15, 46),
        color=ACCENT,
        arrowprops={"arrowstyle": "->", "color": ACCENT, "lw": 2},
    )

    fig.tight_layout()
    path = os.path.join(CFG.train.output_dir, "fig4_vision_contribution.png")
    fig.savefig(path, dpi=220, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def main() -> None:
    """Generate all report figures."""
    os.makedirs(CFG.train.output_dir, exist_ok=True)
    paths = [
        figure_model_comparison(),
        figure_training_curve(),
        figure_data_vs_accuracy(),
        figure_vision_contribution(),
    ]
    for path in paths:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
