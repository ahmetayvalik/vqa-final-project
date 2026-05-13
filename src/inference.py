"""Inference utilities for BLIP-2 VQA evaluation."""

import json
import os
import sys
import time
from collections import Counter
from statistics import median
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import CFG
def predict_single(processor, model, image, question: str) -> str:
    text = f"Question: {question} Short answer:"
    inputs = processor(
        images=image.convert("RGB"),
        text=text,
        return_tensors="pt"
    )
    inputs = {k: v.to(CFG.model.device) for k, v in inputs.items()}
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
        )
    
    # Decode FULL output then remove the input prompt
    full_text = processor.decode(generated_ids[0], skip_special_tokens=True)
    # Remove the question prefix to get only the answer
    marker = "Short answer:"
    if marker in full_text:
        answer = full_text.split(marker)[-1].strip().lower()
    else:
        answer = full_text.strip().lower()
    
    # Take only first word/phrase
    answer = answer.split("\n")[0].strip()
    return answer


def predict_batch(processor, model, samples: list) -> list:
    """Run batched inference loop with latency and per-item VQA scores."""
    results = []
    for sample in tqdm(samples, desc="Running inference", unit="sample"):
        image = sample["image"].convert("RGB")
        question = sample["question"]
        all_answers = sample["answers"]

        start = time.perf_counter()
        predicted = predict_single(processor, model, image, question)
        latency_ms = (time.perf_counter() - start) * 1000.0

        count = sum(1 for a in all_answers if a["answer"].lower() == predicted)
        vqa_score = min(count / 3.0, 1.0)

        majority = Counter(a["answer"].lower() for a in all_answers).most_common(1)
        ground_truth = majority[0][0] if majority else ""
        is_exact_match = predicted == ground_truth

        results.append(
            {
                "question": question,
                "predicted": predicted,
                "ground_truth": ground_truth,
                "all_answers": all_answers,
                "vqa_score": vqa_score,
                "latency_ms": latency_ms,
                "is_exact_match": is_exact_match,
            }
        )
    return results


def compute_metrics(results: list) -> dict[str, Any]:
    """Compute aggregate inference metrics and print a summary table."""
    total_samples = len(results)
    if total_samples == 0:
        metrics: dict[str, Any] = {
            "exact_match_accuracy": 0.0,
            "vqa_soft_accuracy": 0.0,
            "avg_latency_ms": 0.0,
            "median_latency_ms": 0.0,
            "total_samples": 0,
        }
    else:
        exact_matches = sum(1 for item in results if item["is_exact_match"])
        soft_scores = sum(item["vqa_score"] for item in results)
        latencies = [item["latency_ms"] for item in results]
        metrics = {
            "exact_match_accuracy": (exact_matches / total_samples) * 100.0,
            "vqa_soft_accuracy": (soft_scores / total_samples) * 100.0,
            "avg_latency_ms": sum(latencies) / total_samples,
            "median_latency_ms": median(latencies),
            "total_samples": total_samples,
        }

    print("+----------------------+------------------+")
    print("| Metric               | Value            |")
    print("+----------------------+------------------+")
    print(f"| Exact Match Acc (%)  | {metrics['exact_match_accuracy']:<16.2f} |")
    print(f"| VQA Soft Acc (%)     | {metrics['vqa_soft_accuracy']:<16.2f} |")
    print(f"| Avg Latency (ms)     | {metrics['avg_latency_ms']:<16.2f} |")
    print(f"| Median Latency (ms)  | {metrics['median_latency_ms']:<16.2f} |")
    print(f"| Total Samples        | {int(metrics['total_samples']):<16d} |")
    print("+----------------------+------------------+")
    return metrics


def run_inference_pipeline(processor, model, test_samples) -> tuple[list, dict[str, Any]]:
    """Run inference end-to-end, compute metrics, and save JSON results."""
    results = predict_batch(processor, model, test_samples)
    metrics = compute_metrics(results)

    os.makedirs(CFG.train.output_dir, exist_ok=True)
    output_path = os.path.join(CFG.train.output_dir, "inference_results.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)

    return results, metrics


if __name__ == "__main__":
    from src.models.llava import load_processor, load_model
    from src.dataset_local import load_local_vqa_samples
    proc = load_processor()
    mdl = load_model(for_training=False)
    samples = load_local_vqa_samples()[:3]
    for s in samples:
        ans = predict_single(proc, mdl, s["image"], s["question"])
        gt = s["answers"][0]["answer"]
        print(f"Q: {s['question']} | GT: {gt} | Pred: {ans}")
