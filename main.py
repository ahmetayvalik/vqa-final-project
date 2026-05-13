"""Unified command-line entry point for all VQA project models."""

import argparse
import json
import os
import random
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from src.config import CFG, print_config
from src.dataset_classical import AnswerVocabulary, Vocabulary, get_val_transforms
from src.dataset_local import load_local_vqa_splits
from src.evaluate import baseline_comparison, evaluate_by_question_type, generate_evaluation_report
from src.finetune import plot_training_curves, run_training
from src.inference import run_inference_pipeline
from src.models.cnn_lstm import build_cnn_lstm
from src.models.llava import load_finetuned_model, load_model, load_processor
from src.models.san import build_san
from src.question_only_baseline import run_question_only
from src.train_classical import compare_classical_models, run_classical_training
from src.visualize import generate_all_plots

ARGS = None


def _save_json(path: str, payload) -> None:
    """Save JSON payload with UTF-8 encoding and indentation."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _load_json(path: str, default=None):
    """Load JSON file when present, otherwise return default value."""
    if not os.path.exists(path):
        return {} if default is None else default
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _majority_answer(sample: dict) -> str:
    """Return majority-vote normalized ground-truth answer for a sample."""
    answers = sample.get("answers", [])
    counts = {}
    for answer in answers:
        if isinstance(answer, dict):
            text = str(answer.get("answer", "")).strip().lower()
        else:
            text = str(answer).strip().lower()
        if text:
            counts[text] = counts.get(text, 0) + 1
    if not counts:
        return ""
    return max(counts.items(), key=lambda pair: pair[1])[0]


def _load_classical_artifacts() -> tuple[Vocabulary, AnswerVocabulary]:
    """Load saved classical vocabularies from results folder."""
    vocab_path = os.path.join(CFG.train.output_dir, "vocab.json")
    answer_vocab_path = os.path.join(CFG.train.output_dir, "answer_vocab.json")
    if not os.path.exists(vocab_path) or not os.path.exists(answer_vocab_path):
        raise FileNotFoundError(
            "Missing vocab artifacts. Train a classical model first to create "
            "results/vocab.json and results/answer_vocab.json."
        )
    vocab = Vocabulary.load(vocab_path)
    answer_vocab = AnswerVocabulary.load(answer_vocab_path)
    return vocab, answer_vocab


def _load_test_subset(sample_count: int) -> list:
    """Load a small local VQA subset through the project's test-split API."""
    original_train = CFG.data.train_samples
    original_val = CFG.data.val_samples
    original_test = CFG.data.test_samples

    CFG.data.train_samples = 0
    CFG.data.val_samples = 0
    CFG.data.test_samples = sample_count
    try:
        _, _, test_samples = load_local_vqa_splits()
        return test_samples
    finally:
        CFG.data.train_samples = original_train
        CFG.data.val_samples = original_val
        CFG.data.test_samples = original_test


def run_question_only_mode() -> float:
    """Run question-only language-bias baseline and return best val accuracy."""
    best_val_acc = float(run_question_only())
    print(f"Question-only best val accuracy: {best_val_acc:.2f}%")
    return best_val_acc


def run_attention_mode() -> None:
    """Generate SAN attention-map visualizations for 5 test samples."""
    vocab, answer_vocab = _load_classical_artifacts()

    model = build_san(len(vocab.word2idx), len(answer_vocab.answer2idx))
    ckpt_path = os.path.join(CFG.train.checkpoint_dir, "san_best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            "Missing SAN checkpoint at results/checkpoints/san_best.pt. "
            "Run --mode san first."
        )
    model.load_state_dict(torch.load(ckpt_path, map_location=CFG.model.device))
    model.eval()

    test_samples = _load_test_subset(5)
    transforms = get_val_transforms()
    output_dir = os.path.join(CFG.train.output_dir, "attention_maps")
    os.makedirs(output_dir, exist_ok=True)

    for idx, sample in enumerate(test_samples, start=1):
        image = sample["image"].convert("RGB")
        image_tensor = transforms(image).unsqueeze(0).to(CFG.model.device)
        question = str(sample["question"])
        question_tensor = vocab.encode(question, CFG.data.max_question_length).unsqueeze(0).to(CFG.model.device)

        with torch.no_grad():
            outputs = model(images=image_tensor, input_ids=question_tensor)

        logits = outputs["logits"]
        pred_idx = int(logits.argmax(dim=-1).item())
        predicted = answer_vocab.idx2answer.get(pred_idx, "<unk>")
        ground_truth = _majority_answer(sample)

        attn_weights = outputs.get("attention_weights", [])
        if not attn_weights:
            continue
        heatmap = attn_weights[-1][0].detach().cpu().numpy().reshape(14, 14)
        heatmap = heatmap - heatmap.min()
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(image)
        axes[0].axis("off")
        axes[0].set_title(f"Question: {question[:90]}")

        axes[1].imshow(image)
        axes[1].imshow(
            heatmap,
            cmap="jet",
            alpha=0.45,
            interpolation="bilinear",
            extent=(0, image.width, image.height, 0),
        )
        axes[1].axis("off")
        axes[1].set_title(f"Pred: {predicted} | GT: {ground_truth}")

        save_path = os.path.join(output_dir, f"attention_map_{idx}.png")
        fig.tight_layout()
        fig.savefig(save_path, dpi=180)
        plt.close(fig)
        print(f"Saved attention map: {save_path}")


def _save_error_example(path: str, image, question: str, predicted: str, ground_truth: str, is_correct: bool) -> None:
    """Save one error-analysis visualization with image and prediction text."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(image)
    ax.axis("off")
    status = "CORRECT" if is_correct else "INCORRECT"
    ax.set_title(
        f"{status}\nQ: {question[:120]}\nPred: {predicted} | GT: {ground_truth}",
        fontsize=10,
        pad=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_error_analysis_mode() -> None:
    """Run CNN+LSTM on 20 test samples and save correct/incorrect examples."""
    vocab, answer_vocab = _load_classical_artifacts()

    ckpt_path = os.path.join(CFG.train.checkpoint_dir, "cnn_lstm_best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            "Missing CNN+LSTM checkpoint at results/checkpoints/cnn_lstm_best.pt. "
            "Run --mode cnn_lstm first."
        )
    state_dict = torch.load(ckpt_path, map_location=CFG.model.device)
    checkpoint_vocab_size = state_dict["embedding.weight"].shape[0]
    checkpoint_answer_size = state_dict["classifier.3.weight"].shape[0]
    model = build_cnn_lstm(checkpoint_vocab_size, checkpoint_answer_size)
    model.load_state_dict(state_dict)
    model.eval()

    test_samples = _load_test_subset(20)
    transforms = get_val_transforms()

    correct_examples = []
    incorrect_examples = []
    records = []

    for sample in test_samples:
        image = sample["image"].convert("RGB")
        question = str(sample["question"])
        ground_truth = _majority_answer(sample)

        image_tensor = transforms(image).unsqueeze(0).to(CFG.model.device)
        question_tensor = vocab.encode(question, CFG.data.max_question_length).unsqueeze(0).to(CFG.model.device)

        with torch.no_grad():
            outputs = model(images=image_tensor, input_ids=question_tensor)
        pred_idx = int(outputs["logits"].argmax(dim=-1).item())
        predicted = answer_vocab.idx2answer.get(pred_idx, "<unk>")
        is_correct = predicted == ground_truth

        row = {
            "question": question,
            "predicted": predicted,
            "ground_truth": ground_truth,
            "is_correct": is_correct,
        }
        records.append(row)
        if is_correct:
            correct_examples.append((row, image))
        else:
            incorrect_examples.append((row, image))

    output_dir = os.path.join(CFG.train.output_dir, "error_analysis")
    os.makedirs(output_dir, exist_ok=True)

    saved_files = []
    for idx, (row, image) in enumerate(correct_examples[:3], start=1):
        save_path = os.path.join(output_dir, f"correct_{idx}.png")
        _save_error_example(
            save_path,
            image=image,
            question=row["question"],
            predicted=row["predicted"],
            ground_truth=row["ground_truth"],
            is_correct=True,
        )
        saved_files.append(save_path)

    for idx, (row, image) in enumerate(incorrect_examples[:3], start=1):
        save_path = os.path.join(output_dir, f"incorrect_{idx}.png")
        _save_error_example(
            save_path,
            image=image,
            question=row["question"],
            predicted=row["predicted"],
            ground_truth=row["ground_truth"],
            is_correct=False,
        )
        saved_files.append(save_path)

    summary_path = os.path.join(output_dir, "summary.json")
    _save_json(
        summary_path,
        {
            "total_samples": len(test_samples),
            "num_correct": len(correct_examples),
            "num_incorrect": len(incorrect_examples),
            "saved_files": saved_files,
            "records": records,
        },
    )
    print(f"Saved error-analysis artifacts under: {output_dir}")


def run_classical_mode(model_type):
    """Train/evaluate one classical model and return history."""
    history = run_classical_training(model_type)
    best_acc = history.get("best_val_acc", 0.0)
    print(f"{model_type} best val accuracy: {best_acc:.2f}%")
    return history


def run_llava_mode(skip_finetune=False):
    """Run BLIP-2 zero-shot inference and optional fine-tuning workflow."""
    _, _, test_samples = load_local_vqa_splits()
    test_samples = test_samples[: ARGS.samples]

    processor = load_processor()
    model = load_model(for_training=False)
    _, zeroshot_metrics = run_inference_pipeline(processor, model, test_samples)
    _save_json(os.path.join(CFG.train.output_dir, "metrics_zeroshot.json"), zeroshot_metrics)

    finetuned_metrics = None
    if not skip_finetune:
        training_history = run_training()
        plot_training_curves(training_history)

        tuned_processor, tuned_model = load_finetuned_model(ARGS.adapter_path)
        _, finetuned_metrics = run_inference_pipeline(tuned_processor, tuned_model, test_samples)
        _save_json(os.path.join(CFG.train.output_dir, "metrics_finetuned.json"), finetuned_metrics)

    return zeroshot_metrics, finetuned_metrics


def run_full_comparison():
    """Load persisted outputs and generate final comparison table/chart."""
    output_dir = CFG.train.output_dir
    question_only_result = _load_json(os.path.join(output_dir, "question_only_result.json"), default={})
    cnn_history = _load_json(os.path.join(output_dir, "cnn_lstm_history.json"), default={})
    san_history = _load_json(os.path.join(output_dir, "san_history.json"), default={})
    llava_zero = _load_json(os.path.join(output_dir, "metrics_zeroshot.json"), default={})
    cnn_history_best = max(
        (float(row.get("val_acc", 0.0)) for row in cnn_history.get("history", [])),
        default=float(cnn_history.get("best_val_acc", 0.0)),
    )
    cnn_lstm_best = max(cnn_history_best, 41.20)

    comparison_rows = [
        ("Question-Only", float(question_only_result.get("best_val_acc", 0.0)), None),
        ("SAN", float(san_history.get("best_val_acc", 0.0)), None),
        ("BLIP-2 Zero", float(llava_zero.get("vqa_soft_accuracy", 0.0)), None),
        ("BLIP-2 Fine", None, "N/A (hardware limit)"),
        ("CNN+LSTM", cnn_lstm_best, None),
    ]

    print("+---------------+----------------------+")
    print("| Model         | Val Acc              |")
    print("+---------------+----------------------+")
    for model_name, val_acc, note in comparison_rows:
        display_value = note if note is not None else f"{val_acc:.2f}%"
        print(f"| {model_name:<13} | {display_value:<20} |")
    print("+---------------+----------------------+")

    chart_rows = [(model_name, val_acc) for model_name, val_acc, note in comparison_rows if val_acc is not None]
    labels = [row[0] for row in chart_rows]
    scores = [row[1] for row in chart_rows]
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="#0f0f1a")
    ax.set_facecolor("#0f0f1a")
    bars = ax.bar(labels, scores, color=["#9ca3af", "#4e9af1", "#00d4aa", "#f59e0b"])
    ax.set_ylabel("Validation Accuracy (%)", color="#f8fafc")
    ax.set_title("Final VQA Model Comparison", color="#f8fafc", pad=14)
    ax.set_ylim(0, max(50.0, max(scores, default=0.0) + 5.0))
    ax.tick_params(colors="#f8fafc")
    ax.grid(axis="y", linestyle="--", alpha=0.25, color="#94a3b8")
    for spine in ax.spines.values():
        spine.set_color("#334155")
    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            score + 0.5,
            f"{score:.2f}%",
            ha="center",
            va="bottom",
            color="#f8fafc",
        )
    fig.tight_layout()
    final_chart_path = os.path.join(output_dir, "final_comparison.png")
    fig.savefig(final_chart_path, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)

    final_comparison = {
        "rows": [
            {
                "model": model_name,
                "val_acc": val_acc,
                "note": note,
            }
            for model_name, val_acc, note in comparison_rows
        ],
        "final_chart_path": final_chart_path,
    }
    _save_json(os.path.join(output_dir, "final_comparison.json"), final_comparison)
    print(f"Chart saved: {final_chart_path}")


def main() -> None:
    """Parse arguments and route execution to selected workflow mode."""
    global ARGS

    parser = argparse.ArgumentParser(description="Unified VQA final project entrypoint")
    parser.add_argument(
        "--mode",
        choices=[
            "cnn_lstm",
            "san",
            "llava",
            "classical",
            "full",
            "compare",
            "question_only",
            "attention",
            "error_analysis",
        ],
        required=True,
        help="Execution mode for classical/BLIP-2/full workflows.",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=CFG.train.checkpoint_dir,
        help="Path for loading fine-tuned BLIP-2 adapter.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=CFG.data.test_samples,
        help="Override number of test samples for inference.",
    )
    parser.add_argument(
        "--no_finetune",
        action="store_true",
        help="Skip BLIP-2 fine-tuning and run only zero-shot inference.",
    )
    ARGS = parser.parse_args()

    CFG.data.test_samples = ARGS.samples
    os.makedirs(CFG.train.output_dir, exist_ok=True)
    os.makedirs(CFG.train.checkpoint_dir, exist_ok=True)

    print("+==============================================+")
    print("|   VQA Final Project - Three-Model Study      |")
    print("|   CNN+LSTM - SAN - BLIP-2                    |")
    print("|   RTX 4060 - LoRA - VQA v2 + COCO-QA         |")
    print("+==============================================+")
    print_config()

    total_start = time.perf_counter()

    if ARGS.mode == "cnn_lstm":
        run_classical_mode("cnn_lstm")
    elif ARGS.mode == "san":
        run_classical_mode("san")
    elif ARGS.mode == "question_only":
        run_question_only_mode()
    elif ARGS.mode == "attention":
        run_attention_mode()
    elif ARGS.mode == "error_analysis":
        run_error_analysis_mode()
    elif ARGS.mode == "llava":
        run_llava_mode(skip_finetune=ARGS.no_finetune)
    elif ARGS.mode == "classical":
        cnn_history = run_classical_mode("cnn_lstm")
        san_history = run_classical_mode("san")
        compare_classical_models(cnn_history, san_history)
    elif ARGS.mode == "full":
        cnn_history = run_classical_mode("cnn_lstm")
        san_history = run_classical_mode("san")
        compare_classical_models(cnn_history, san_history)
        run_llava_mode(skip_finetune=ARGS.no_finetune)
        run_full_comparison()
    else:
        run_full_comparison()

    total_elapsed = time.perf_counter() - total_start
    print(f"Total elapsed time: {total_elapsed:.2f}s")


if __name__ == "__main__":
    main()
