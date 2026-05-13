"""Local VQA v2 dataset loader using annotation/question JSON files."""

import json
import os
from io import BytesIO

import requests
from PIL import Image
from tqdm import tqdm

from src.config import CFG

QUESTIONS_PATH = os.path.join("data", "vqa", "v2_OpenEnded_mscoco_train2014_questions.json")
ANNOTATIONS_PATH = os.path.join("data", "vqa", "v2_mscoco_train2014_annotations.json")
IMAGE_CACHE_DIR = os.path.join("data", "images")
COCO_TRAIN_URL = "http://images.cocodataset.org/train2014/COCO_train2014_{image_id}.jpg"


class LocalVQASample:
    """Lazy local VQA sample that downloads and caches its image on first access."""

    def __init__(self, question: str, answers: list, image_id: int) -> None:
        self.question = question
        self.answers = answers
        self.image_id = image_id

    def __getitem__(self, key):
        if key == "image":
            return self._load_image()
        if key == "question":
            return self.question
        if key == "answers":
            return self.answers
        if key == "image_id":
            return self.image_id
        raise KeyError(key)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def _load_image(self) -> Image.Image:
        os.makedirs(IMAGE_CACHE_DIR, exist_ok=True)
        image_name = f"{self.image_id}.jpg"
        image_path = os.path.join(IMAGE_CACHE_DIR, image_name)

        if not os.path.exists(image_path):
            padded_id = str(self.image_id).zfill(12)
            url = COCO_TRAIN_URL.format(image_id=padded_id)
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            with open(image_path, "wb") as handle:
                handle.write(response.content)

        return Image.open(image_path).convert("RGB")


def _load_json(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required VQA file not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_local_vqa_samples(limit: int | None = None) -> list:
    """Load and merge local VQA questions/annotations by question_id."""
    questions_data = _load_json(QUESTIONS_PATH)
    annotations_data = _load_json(ANNOTATIONS_PATH)

    annotations_by_qid = {
        item["question_id"]: item for item in annotations_data.get("annotations", [])
    }

    samples = []
    questions = questions_data.get("questions", [])
    if limit is not None:
        questions = questions[:limit]

    for question_item in tqdm(questions, desc="Loading local VQA metadata", unit="sample"):
        question_id = question_item["question_id"]
        annotation = annotations_by_qid.get(question_id)
        if annotation is None:
            continue
        samples.append(
            LocalVQASample(
                question=question_item["question"],
                answers=annotation.get("answers", []),
                image_id=question_item["image_id"],
            )
        )

    print(f"Loaded local VQA samples: {len(samples)}")
    return samples


def load_local_vqa_train_val_splits() -> tuple:
    """Load local VQA data and split it into train/validation subsets."""
    total_needed = CFG.data.train_samples + CFG.data.val_samples
    samples = load_local_vqa_samples(limit=total_needed)
    train_samples = samples[: CFG.data.train_samples]
    val_samples = samples[CFG.data.train_samples : total_needed]

    print(f"Loaded train samples: {len(train_samples)}")
    print(f"Loaded validation samples: {len(val_samples)}")
    return train_samples, val_samples


def load_local_vqa_splits() -> tuple:
    """Load local VQA data and split it into train/validation/test subsets."""
    total_needed = CFG.data.train_samples + CFG.data.val_samples + CFG.data.test_samples
    samples = load_local_vqa_samples(limit=total_needed)

    train_end = CFG.data.train_samples
    val_end = train_end + CFG.data.val_samples

    train_samples = samples[:train_end]
    val_samples = samples[train_end:val_end]
    test_samples = samples[val_end:total_needed]

    if not test_samples and samples:
        test_samples = samples[-min(CFG.data.test_samples, len(samples)) :]

    print(f"Loaded train samples: {len(train_samples)}")
    print(f"Loaded validation samples: {len(val_samples)}")
    print(f"Loaded test samples: {len(test_samples)}")
    return train_samples, val_samples, test_samples
