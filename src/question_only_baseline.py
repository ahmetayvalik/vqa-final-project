"""Question-only VQA baseline to measure language bias."""

import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import tqdm

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import CFG
from src.dataset_classical import build_vocabularies, get_classical_dataloaders
from src.dataset_local import load_local_vqa_train_val_splits


class QuestionOnlyModel(nn.Module):
    """LSTM question encoder with answer classification head."""

    def __init__(self, vocab_size: int, answer_vocab_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 512)
        self.lstm = nn.LSTM(512, 1024, num_layers=2, batch_first=True)
        self.classifier = nn.Linear(1024, answer_vocab_size)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, labels=None):
        embedded = self.embedding(input_ids)
        _, (hidden_states, _) = self.lstm(embedded)
        last_hidden = hidden_states[-1]
        logits = self.classifier(last_hidden)

        output = {"logits": logits}
        if labels is not None:
            output["loss"] = self.loss_fn(logits, labels)
        return output


def _majority_vote(answers) -> str:
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
    return max(counts.items(), key=lambda item: item[1])[0]


def _encode_samples(samples, vocab, answer_vocab, split_name: str):
    encoded = []
    progress = tqdm.tqdm(samples, desc=f"Encoding {split_name}", unit="sample")
    for sample in progress:
        input_ids = vocab.encode(sample["question"], CFG.data.max_question_length)
        label = answer_vocab.encode(_majority_vote(sample.get("answers", [])))
        encoded.append((input_ids, torch.tensor(label, dtype=torch.long)))
    return encoded


def _collate_question_only(batch):
    input_ids = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    return {"input_ids": input_ids, "labels": labels}


def _make_text_loader(encoded_samples, shuffle: bool):
    return torch.utils.data.DataLoader(
        encoded_samples,
        batch_size=CFG.data.batch_size,
        shuffle=shuffle,
        num_workers=CFG.data.num_workers,
        pin_memory=CFG.model.device == "cuda",
        collate_fn=_collate_question_only,
    )


def _evaluate(model, loader, device: str) -> float:
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm.tqdm(loader, desc="Val", unit="batch"):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, labels=labels)
            preds = outputs["logits"].argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    if total == 0:
        raise RuntimeError("Validation set is empty; cannot compute accuracy.")
    return 100.0 * correct / total


def run_question_only():
    _ = get_classical_dataloaders
    original_train_samples = CFG.data.train_samples
    original_epochs = CFG.train.epochs

    CFG.data.train_samples = 5000
    CFG.train.epochs = 5

    try:
        train_samples, val_samples = load_local_vqa_train_val_splits()
        vocab, answer_vocab = build_vocabularies(train_samples)

        train_encoded = _encode_samples(train_samples, vocab, answer_vocab, "train")
        val_encoded = _encode_samples(val_samples, vocab, answer_vocab, "validation")

        train_loader = _make_text_loader(train_encoded, shuffle=True)
        val_loader = _make_text_loader(val_encoded, shuffle=False)

        model = QuestionOnlyModel(len(vocab.word2idx), len(answer_vocab.answer2idx)).to(CFG.model.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.train.learning_rate, weight_decay=1e-4)

        best_val_acc = 0.0
        best_epoch = 0

        for epoch in range(1, CFG.train.epochs + 1):
            model.train()
            running_loss = 0.0
            seen_batches = 0

            for batch in tqdm.tqdm(train_loader, desc=f"Train Epoch {epoch}", unit="batch"):
                input_ids = batch["input_ids"].to(CFG.model.device)
                labels = batch["labels"].to(CFG.model.device)

                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs["loss"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                running_loss += loss.item()
                seen_batches += 1

            train_loss = running_loss / max(1, seen_batches)
            val_acc = _evaluate(model, val_loader, CFG.model.device)
            print(f"Epoch {epoch}/5 | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.2f}%")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch

        os.makedirs(CFG.train.output_dir, exist_ok=True)
        result_path = os.path.join(CFG.train.output_dir, "question_only_result.json")
        with open(result_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "train_samples": 5000,
                    "val_samples": len(val_samples),
                    "epochs": 5,
                    "best_val_acc": best_val_acc,
                    "best_epoch": best_epoch,
                },
                handle,
                indent=2,
            )

        print(f"Final question-only val accuracy: {best_val_acc:.2f}%")
        return best_val_acc
    finally:
        CFG.data.train_samples = original_train_samples
        CFG.train.epochs = original_epochs


if __name__ == "__main__":
    run_question_only()
