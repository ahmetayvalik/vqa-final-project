"""Dataset utilities for preparing VQA data in BLIP-2 format."""

from collections import Counter

import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.config import CFG

BLIP2_PROMPT = "Question: {question} Short answer:"


def _collect_split(stream, limit: int, split_name: str) -> list:
    """Collect a fixed number of samples from a streamed split."""
    samples = []
    for sample in tqdm(stream, total=limit, desc=f"Loading {split_name}", unit="sample"):
        samples.append(sample)
        if len(samples) >= limit:
            break
    return samples


def load_vqa_splits() -> tuple:
    """Load train/val/test data from HuggingFace VQA with streaming."""
    if CFG.data.dataset_name == "lmms-lab/VQAv2":
        train_split = "validation"
        val_split = "testdev"
        test_split = "test"
    else:
        train_split = "train"
        val_split = "validation"
        test_split = "test"

    train_stream = load_dataset(CFG.data.dataset_name, split=train_split, streaming=True)
    val_stream = load_dataset(CFG.data.dataset_name, split=val_split, streaming=True)
    test_stream = load_dataset(CFG.data.dataset_name, split=test_split, streaming=True)

    train_samples = _collect_split(train_stream, CFG.data.train_samples, train_split)
    val_samples = _collect_split(val_stream, CFG.data.val_samples, val_split)
    test_samples = _collect_split(test_stream, CFG.data.test_samples, test_split)

    print(f"Loaded train samples: {len(train_samples)}")
    print(f"Loaded validation samples: {len(val_samples)}")
    print(f"Loaded test samples: {len(test_samples)}")

    return train_samples, val_samples, test_samples


def load_vqa_train_val_splits() -> tuple:
    """Load only train/val streams for classical training to reduce Hub calls."""
    if CFG.data.dataset_name == "lmms-lab/VQAv2":
        train_split = "validation"
        val_split = "testdev"
    else:
        train_split = "train"
        val_split = "validation"

    train_stream = load_dataset(CFG.data.dataset_name, split=train_split, streaming=True)
    val_stream = load_dataset(CFG.data.dataset_name, split=val_split, streaming=True)

    train_samples = _collect_split(train_stream, CFG.data.train_samples, train_split)
    val_samples = _collect_split(val_stream, CFG.data.val_samples, val_split)

    print(f"Loaded train samples: {len(train_samples)}")
    print(f"Loaded validation samples: {len(val_samples)}")

    return train_samples, val_samples


class VQADataset(Dataset):
    """PyTorch dataset for BLIP-2 VQA examples."""

    def __init__(self, samples, processor, split: str = "train") -> None:
        """Store raw samples and shared processor."""
        self.samples = samples
        self.processor = processor
        self.split = split

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.samples)

    @staticmethod
    def _majority_answer(answers) -> str:
        """Compute majority-vote answer from the sample answer list."""
        if not answers:
            return ""

        normalized = []
        for item in answers:
            if isinstance(item, dict):
                normalized.append(str(item.get("answer", "")))
            else:
                normalized.append(str(item))
        return Counter(normalized).most_common(1)[0][0]

    def __getitem__(self, index):
        """Encode a single sample; return None if sample processing fails."""
        try:
            sample = self.samples[index]
            image = sample["image"].convert("RGB")
            question = sample["question"]
            answer = self._majority_answer(sample["answers"])
            prompt = BLIP2_PROMPT.format(question=question)

            encoding = self.processor(
                images=image,
                text=prompt,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=CFG.data.max_question_length,
            )

            label_ids = self.processor.tokenizer(
                answer,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=CFG.data.max_answer_length,
            ).input_ids.squeeze()
            label_ids[label_ids == self.processor.tokenizer.pad_token_id] = -100

            return {
                "input_ids": encoding.input_ids.squeeze(),
                "attention_mask": encoding.attention_mask.squeeze(),
                "pixel_values": encoding.pixel_values.squeeze(),
                "labels": label_ids,
            }
        except Exception:
            return None


def collate_fn(batch):
    """Filter invalid items and stack tensor fields into a clean batch dict."""
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
    }


def get_dataloaders(train_data, val_data, processor):
    """Build train/validation dataloaders using project config settings."""
    train_dataset = VQADataset(train_data, processor, split="train")
    val_dataset = VQADataset(val_data, processor, split="validation")
    pin_memory = CFG.model.device == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.data.batch_size,
        shuffle=True,
        num_workers=CFG.data.num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG.data.batch_size,
        shuffle=False,
        num_workers=CFG.data.num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader
