"""Dataset pipeline for classical VQA baselines (CNN+LSTM and SAN)."""

import collections
import json
import os
import re

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from src.config import CFG

IMAGE_SIZE = 224
COCO_QA_URL = "https://www.cs.toronto.edu/~mren/research/imageqa/data/cocoqa/"


def _tokenize(text: str) -> list[str]:
    """Tokenize text with lowercase normalization and punctuation splitting."""
    return re.findall(r"\b\w+\b", text.lower())


class Vocabulary:
    """Question-token vocabulary for classical text encoders."""

    def __init__(self, min_freq=1):
        self.min_freq = min_freq
        self.word2idx = {"<pad>": 0, "<unk>": 1}
        self.idx2word = {0: "<pad>", 1: "<unk>"}
        self.word_freq = collections.Counter()

    def build_from_questions(self, questions: list):
        """Build vocabulary from question strings with minimum frequency filtering."""
        for question in tqdm(questions, desc="Building question vocab", unit="question"):
            self.word_freq.update(_tokenize(str(question)))

        for token, freq in self.word_freq.items():
            if freq >= self.min_freq and token not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[token] = idx
                self.idx2word[idx] = token

        print(f"Question vocab size: {len(self.word2idx)}")

    def encode(self, question: str, max_len: int) -> torch.LongTensor:
        """Encode question into a fixed-length index tensor."""
        tokens = _tokenize(str(question))
        ids = [self.word2idx.get(token, self.word2idx["<unk>"]) for token in tokens]
        if len(ids) < max_len:
            ids.extend([self.word2idx["<pad>"]] * (max_len - len(ids)))
        else:
            ids = ids[:max_len]
        return torch.LongTensor(ids)

    def save(self, path: str):
        """Save question vocabulary mapping to disk."""
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.word2idx, handle, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str):
        """Load question vocabulary mapping from disk."""
        with open(path, "r", encoding="utf-8") as handle:
            word2idx = json.load(handle)
        vocab = cls()
        vocab.word2idx = {str(key): int(value) for key, value in word2idx.items()}
        vocab.idx2word = {idx: token for token, idx in vocab.word2idx.items()}
        return vocab


class AnswerVocabulary:
    """Answer-label vocabulary for classification-based VQA baselines."""

    def __init__(self, top_k=1000):
        self.answer2idx = {}
        self.idx2answer = {}
        self.top_k = top_k

    def build_from_answers(self, answers: list):
        """Build answer mapping from top-k most frequent answers."""
        counter = collections.Counter(str(answer).strip().lower() for answer in answers)
        total = sum(counter.values())
        most_common = counter.most_common(self.top_k)

        self.answer2idx = {"<unk>": 0}
        self.idx2answer = {0: "<unk>"}

        for idx, (answer, _) in enumerate(most_common, start=1):
            self.answer2idx[answer] = idx
            self.idx2answer[idx] = answer

        kept = sum(freq for _, freq in most_common)
        coverage = (kept / total * 100.0) if total else 0.0
        print(f"Answer vocab size: {len(self.answer2idx)}")
        print(f"Answer coverage: {coverage:.2f}%")

    def encode(self, answer: str) -> int:
        """Encode answer string to index, defaulting to unknown token."""
        return self.answer2idx.get(str(answer).strip().lower(), 0)

    def save(self, path: str):
        """Save answer vocabulary mapping to disk."""
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.answer2idx, handle, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str):
        """Load answer vocabulary mapping from disk."""
        with open(path, "r", encoding="utf-8") as handle:
            answer2idx = json.load(handle)
        vocab = cls()
        vocab.answer2idx = {str(key): int(value) for key, value in answer2idx.items()}
        vocab.idx2answer = {idx: answer for answer, idx in vocab.answer2idx.items()}
        return vocab


def get_train_transforms() -> transforms.Compose:
    """Return train-time image transforms for classical VQA baselines."""
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_val_transforms() -> transforms.Compose:
    """Return validation-time image transforms for classical VQA baselines."""
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class ClassicalVQADataset(Dataset):
    """PyTorch dataset wrapper for classical VQA model training/evaluation."""

    def __init__(self, samples, vocab, answer_vocab, split="train"):
        self.samples = samples
        self.vocab = vocab
        self.answer_vocab = answer_vocab
        self.split = split
        self.transforms = get_train_transforms() if split == "train" else get_val_transforms()

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _majority_vote(answers) -> str:
        """Get majority-vote answer from a sample answer list."""
        normalized = []
        for answer in answers:
            if isinstance(answer, dict):
                normalized.append(str(answer.get("answer", "")).strip().lower())
            else:
                normalized.append(str(answer).strip().lower())
        if not normalized:
            return ""
        return collections.Counter(normalized).most_common(1)[0][0]

    def __getitem__(self, index):
        """Load one sample and return transformed image, token ids, and label."""
        try:
            sample = self.samples[index]
            image = sample["image"]
            if isinstance(image, Image.Image):
                image = image.convert("RGB")
            else:
                image = Image.open(image).convert("RGB")
            image_tensor = self.transforms(image)

            question_tensor = self.vocab.encode(sample["question"], CFG.data.max_question_length)
            majority_answer = self._majority_vote(sample.get("answers", []))
            answer_idx = self.answer_vocab.encode(majority_answer)
            label_tensor = torch.tensor(answer_idx, dtype=torch.long)

            return {
                "image": image_tensor,
                "input_ids": question_tensor,
                "labels": label_tensor,
            }
        except Exception:
            return None


def collate_fn_classical(batch):
    """Collate valid classical VQA samples into batched tensors."""
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
    }


def build_vocabularies(train_samples) -> tuple:
    """Build and persist question/answer vocabularies from train data."""
    os.makedirs(CFG.train.output_dir, exist_ok=True)

    questions = [sample.get("question", "") for sample in train_samples]
    answers = []
    for sample in train_samples:
        for answer in sample.get("answers", []):
            if isinstance(answer, dict):
                answers.append(str(answer.get("answer", "")))
            else:
                answers.append(str(answer))

    vocab = Vocabulary(min_freq=1)
    vocab.build_from_questions(questions)

    answer_vocab = AnswerVocabulary(top_k=1000)
    answer_vocab.build_from_answers(answers)

    vocab_path = os.path.join(CFG.train.output_dir, "vocab.json")
    answer_vocab_path = os.path.join(CFG.train.output_dir, "answer_vocab.json")
    vocab.save(vocab_path)
    answer_vocab.save(answer_vocab_path)

    return vocab, answer_vocab


def get_classical_dataloaders(train_data, val_data, vocab, answer_vocab):
    """Create train/validation dataloaders for classical VQA baselines."""
    train_dataset = ClassicalVQADataset(train_data, vocab, answer_vocab, split="train")
    val_dataset = ClassicalVQADataset(val_data, vocab, answer_vocab, split="validation")

    pin_memory = CFG.model.device == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.data.batch_size,
        shuffle=True,
        num_workers=CFG.data.num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn_classical,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG.data.batch_size,
        shuffle=False,
        num_workers=CFG.data.num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn_classical,
    )
    return train_loader, val_loader
