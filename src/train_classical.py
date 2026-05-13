"""Training pipeline for classical VQA baselines (CNN+LSTM and SAN)."""

import json
import os
import time

import torch
import tqdm
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.config import CFG
from src.dataset_classical import build_vocabularies, get_classical_dataloaders
from src.dataset_local import load_local_vqa_train_val_splits
from src.models.cnn_lstm import build_cnn_lstm
from src.models.san import build_san

scaler = GradScaler()


def train_one_epoch_classical(model, loader, optimizer, device) -> float:
    """Train one epoch for a classical VQA model and return average loss."""
    model.train()
    optimizer.zero_grad()

    running_loss = 0.0
    seen_batches = 0
    progress = tqdm.tqdm(loader, desc="Train", unit="batch")

    for batch in progress:
        if batch is None:
            continue

        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        with autocast():
            outputs = model(images=images, input_ids=input_ids, labels=labels)
        loss = outputs["loss"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        running_loss += loss.item()
        seen_batches += 1
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / max(1, seen_batches)


def evaluate_classical(model, loader, device) -> tuple:
    """Evaluate classical model and return average loss and accuracy (%)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    seen_batches = 0

    with torch.no_grad():
        progress = tqdm.tqdm(loader, desc="Eval", unit="batch")
        for batch in progress:
            if batch is None:
                continue

            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(images=images, input_ids=input_ids, labels=labels)
            logits = outputs["logits"]
            loss = outputs["loss"]

            preds = outputs["logits"].argmax(dim=-1)
            batch_correct = (preds == labels).sum().item()
            correct += batch_correct
            total += labels.size(0)

            running_loss += loss.item()
            seen_batches += 1
            running_acc = (correct / total * 100.0) if total else 0.0
            progress.set_postfix(loss=f"{loss.item():.4f}", acc=f"{running_acc:.2f}")

    if seen_batches == 0 or total == 0:
        raise RuntimeError(
            "No valid validation batches were produced. Check image cache/network access for validation images."
        )

    avg_loss = running_loss / max(1, seen_batches)
    accuracy = (correct / total * 100.0) if total else 0.0
    return avg_loss, accuracy


def run_classical_training(model_type: str = "cnn_lstm") -> dict:
    """Run full training pipeline for selected classical model baseline."""
    if model_type not in {"cnn_lstm", "san"}:
        raise ValueError("model_type must be either 'cnn_lstm' or 'san'.")

    os.makedirs(CFG.train.output_dir, exist_ok=True)
    os.makedirs(CFG.train.checkpoint_dir, exist_ok=True)

    train_samples, val_samples = load_local_vqa_train_val_splits()
    vocab, answer_vocab = build_vocabularies(train_samples)
    train_loader, val_loader = get_classical_dataloaders(train_samples, val_samples, vocab, answer_vocab)

    vocab_size = len(vocab.word2idx)
    answer_vocab_size = len(answer_vocab.answer2idx)
    if model_type == "cnn_lstm":
        model = build_cnn_lstm(vocab_size, answer_vocab_size)
    else:
        model = build_san(vocab_size, answer_vocab_size)

    device = CFG.model.device
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CFG.train.learning_rate,
        weight_decay=1e-4,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=CFG.train.epochs)

    best_val_loss = float("inf")
    patience = 3
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    history_rows = []

    for epoch in range(1, CFG.train.epochs + 1):
        start = time.perf_counter()
        train_loss = train_one_epoch_classical(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate_classical(model, val_loader, device)
        scheduler.step()
        elapsed = time.perf_counter() - start

        print(
            f"Epoch {epoch} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Time: {elapsed:.2f}s"
        )

        best_val_loss = min(best_val_loss, val_loss)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            ckpt_path = os.path.join(CFG.train.checkpoint_dir, f"{model_type}_best.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Best model saved at epoch {epoch} with val_acc={val_acc:.2f}%")
        else:
            patience_counter += 1

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "time": elapsed,
            }
        )

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    history = {
        "model_type": model_type,
        "history": history_rows,
        "best_val_loss": best_val_loss if history_rows else 0.0,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "total_time": sum(row["time"] for row in history_rows),
    }

    history_path = os.path.join(CFG.train.output_dir, f"{model_type}_history.json")
    with open(history_path, "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    return history


def load_classical_model(model_type, vocab, answer_vocab):
    """Load a trained classical model checkpoint and return eval-ready model."""
    if model_type == "cnn_lstm":
        model = build_cnn_lstm(len(vocab.word2idx), len(answer_vocab.answer2idx))
    elif model_type == "san":
        model = build_san(len(vocab.word2idx), len(answer_vocab.answer2idx))
    else:
        raise ValueError("model_type must be either 'cnn_lstm' or 'san'.")

    ckpt_path = os.path.join(CFG.train.checkpoint_dir, f"{model_type}_best.pt")
    state_dict = torch.load(ckpt_path, map_location=CFG.model.device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_classical(model, image_tensor, question_tensor, answer_vocab) -> str:
    """Run single-sample inference for classical baseline and return answer text."""
    model.eval()
    image_batch = image_tensor.unsqueeze(0).to(CFG.model.device)
    question_batch = question_tensor.unsqueeze(0).to(CFG.model.device)

    with torch.no_grad():
        outputs = model(images=image_batch, input_ids=question_batch)
        pred_idx = outputs["logits"].argmax(dim=-1).item()

    return answer_vocab.idx2answer.get(pred_idx, "<unk>")


def compare_classical_models(cnn_lstm_history, san_history):
    """Print and save side-by-side comparison between classical baselines."""
    def _summarize(history):
        rows = history.get("history", [])
        best_val_loss = min((row["val_loss"] for row in rows), default=0.0)
        best_val_acc = max((row["val_acc"] for row in rows), default=0.0)
        total_time = sum(row["time"] for row in rows)
        return best_val_loss, best_val_acc, total_time

    cnn_best_loss, cnn_best_acc, cnn_time = _summarize(cnn_lstm_history)
    san_best_loss, san_best_acc, san_time = _summarize(san_history)

    print("+-----------+---------------+--------------+---------------+")
    print("| Model     | Best Val Loss | Best Val Acc | Training Time |")
    print("+-----------+---------------+--------------+---------------+")
    print(f"| CNN+LSTM  | {cnn_best_loss:<13.4f} | {cnn_best_acc:<12.2f} | {cnn_time:<13.2f} |")
    print(f"| SAN       | {san_best_loss:<13.4f} | {san_best_acc:<12.2f} | {san_time:<13.2f} |")
    print("+-----------+---------------+--------------+---------------+")

    comparison = {
        "cnn_lstm": {
            "best_val_loss": cnn_best_loss,
            "best_val_acc": cnn_best_acc,
            "training_time": cnn_time,
        },
        "san": {
            "best_val_loss": san_best_loss,
            "best_val_acc": san_best_acc,
            "training_time": san_time,
        },
    }

    os.makedirs(CFG.train.output_dir, exist_ok=True)
    output_path = os.path.join(CFG.train.output_dir, "classical_comparison.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(comparison, handle, indent=2)
