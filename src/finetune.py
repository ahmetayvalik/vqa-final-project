"""Training utilities for BLIP-2 LoRA fine-tuning."""

import json
import math
import os
import time
from typing import Any

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import torch
from torch.optim import AdamW
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from src.config import CFG
from src.dataset import get_dataloaders
from src.dataset_local import load_local_vqa_train_val_splits
from src.models.llava import count_parameters, load_model, load_processor


def train_one_epoch(model, loader, optimizer, scheduler, epoch) -> float:
    """Train one epoch with gradient accumulation and return average loss."""
    model.train()
    optimizer.zero_grad()

    grad_accum_steps = max(1, CFG.train.grad_accum_steps)
    running_loss = 0.0
    seen_batches = 0
    micro_step = 0

    progress = tqdm(loader, desc=f"Train Epoch {epoch}", unit="batch")
    for batch in progress:
        if batch is None:
            continue

        batch = {key: value.to(CFG.model.device) for key, value in batch.items()}
        batch["pixel_values"] = batch["pixel_values"].to(torch.float16)

        outputs = model(**batch)
        raw_loss = outputs.loss
        if raw_loss is None or not raw_loss.requires_grad:
            continue
        loss = raw_loss / grad_accum_steps
        if not torch.isfinite(loss):
            optimizer.zero_grad()
            continue
        optimizer.zero_grad()
        loss.backward()

        micro_step += 1
        seen_batches += 1
        running_loss += raw_loss.item()

        if micro_step % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        progress.set_postfix(loss=f"{raw_loss.item():.4f}")

    if micro_step > 0 and micro_step % grad_accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return running_loss / max(1, seen_batches)


def evaluate_epoch(model, loader) -> float:
    """Evaluate one epoch and return average validation loss."""
    model.eval()
    running_loss = 0.0
    seen_batches = 0

    with torch.no_grad():
        progress = tqdm(loader, desc="Validation", unit="batch")
        for batch in progress:
            if batch is None:
                continue

            batch = {key: value.to(CFG.model.device) for key, value in batch.items()}
            batch["pixel_values"] = batch["pixel_values"].to(torch.float16)

            outputs = model(**batch)
            loss = outputs.loss
            running_loss += loss.item()
            seen_batches += 1
            progress.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / max(1, seen_batches)


def run_training() -> dict[str, Any]:
    """Run full fine-tuning pipeline and return training history."""
    os.makedirs(CFG.train.output_dir, exist_ok=True)
    os.makedirs(CFG.train.checkpoint_dir, exist_ok=True)

    train_data, val_data = load_local_vqa_train_val_splits()
    processor = load_processor()
    model = load_model(for_training=True)
    param_stats = count_parameters(model)

    train_loader, val_loader = get_dataloaders(train_data, val_data, processor)

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = AdamW(
        trainable_params,
        lr=CFG.train.learning_rate,
        weight_decay=CFG.train.weight_decay,
    )

    steps_per_epoch = math.ceil(len(train_loader) / max(1, CFG.train.grad_accum_steps))
    total_train_steps = max(1, steps_per_epoch * CFG.train.epochs)
    warmup_steps = int(total_train_steps * CFG.train.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_train_steps,
    )

    best_val_loss = float("inf")
    history_rows = []

    for epoch in range(1, CFG.train.epochs + 1):
        start_time = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, epoch)
        val_loss = evaluate_epoch(model, val_loader)
        elapsed = time.perf_counter() - start_time

        print(
            f"Epoch {epoch}/{CFG.train.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | time={elapsed:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(CFG.train.checkpoint_dir)
            processor.save_pretrained(CFG.train.checkpoint_dir)
            print(f"Saved best checkpoint to: {CFG.train.checkpoint_dir}")

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "time": elapsed,
            }
        )

    history = {
        "history": history_rows,
        "best_val_loss": best_val_loss,
        "parameter_summary": param_stats,
    }

    history_path = os.path.join(CFG.train.output_dir, "training_history.json")
    with open(history_path, "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    return history


def plot_training_curves(history: dict[str, Any]) -> None:
    """Plot and save train/validation loss curves with professional styling."""
    import matplotlib.pyplot as plt

    rows = history.get("history", history)
    epochs = [row["epoch"] for row in rows]
    train_losses = [row["train_loss"] for row in rows]
    val_losses = [row["val_loss"] for row in rows]

    os.makedirs(CFG.train.output_dir, exist_ok=True)
    output_path = os.path.join(CFG.train.output_dir, "training_curves.png")

    fig, ax = plt.subplots(figsize=(10, 6), facecolor="#0f0f1a")
    ax.set_facecolor("#0f0f1a")

    ax.plot(epochs, train_losses, marker="o", linewidth=2.0, color="#58a6ff", label="Train Loss")
    ax.plot(epochs, val_losses, marker="s", linewidth=2.0, color="#ff7b72", label="Val Loss")

    ax.set_title("BLIP-2 LoRA Fine-tuning Curves", color="#f0f6fc", fontsize=14, pad=12)
    ax.set_xlabel("Epoch", color="#c9d1d9")
    ax.set_ylabel("Loss", color="#c9d1d9")
    ax.tick_params(colors="#c9d1d9")
    ax.grid(alpha=0.25, color="#30363d", linestyle="--")
    ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#f0f6fc")

    for spine in ax.spines.values():
        spine.set_color("#30363d")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
