"""Generate SAN attention-map visualizations for local VQA samples."""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import CFG
from src.dataset_classical import AnswerVocabulary, Vocabulary
from src.dataset_local import load_local_vqa_splits
from src.models.san import build_san


def _load_vocabularies() -> tuple[Vocabulary, AnswerVocabulary]:
    """Load saved classical question and answer vocabularies."""
    vocab_path = os.path.join(CFG.train.output_dir, "vocab.json")
    answer_vocab_path = os.path.join(CFG.train.output_dir, "answer_vocab.json")
    if not os.path.exists(vocab_path) or not os.path.exists(answer_vocab_path):
        raise FileNotFoundError("Missing results/vocab.json or results/answer_vocab.json.")
    return Vocabulary.load(vocab_path), AnswerVocabulary.load(answer_vocab_path)


def _load_test_samples(sample_count: int) -> list:
    """Load a small local VQA test split without changing config permanently."""
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


def _safe_encode_question(vocab: Vocabulary, question: str, checkpoint_vocab_size: int) -> torch.Tensor:
    """Encode question and clamp ids not available in the checkpoint vocabulary to <unk>."""
    input_ids = vocab.encode(question, CFG.data.max_question_length)
    input_ids[input_ids >= checkpoint_vocab_size] = vocab.word2idx.get("<unk>", 1)
    return input_ids


def _get_attention_transform() -> transforms.Compose:
    """Use 448x448 images so VGG produces 14x14 spatial attention regions."""
    return transforms.Compose(
        [
            transforms.Resize(448),
            transforms.CenterCrop(448),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _save_attention_plot(
    image,
    question: str,
    predicted: str,
    hop1: torch.Tensor,
    hop2: torch.Tensor,
    output_path: str,
) -> None:
    """Save original image plus two SAN attention hops."""
    hop1_map = hop1.detach().cpu().numpy().reshape(14, 14)
    hop2_map = hop2.detach().cpu().numpy().reshape(14, 14)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="#0f0f1a")
    fig.suptitle(
        f"Q: {question}\nPredicted: {predicted}",
        color="#f8fafc",
        fontsize=12,
    )

    panels = [
        ("Original", None),
        ("Hop 1 Attention", hop1_map),
        ("Hop 2 Attention", hop2_map),
    ]

    for ax, (title, heatmap) in zip(axes, panels):
        ax.set_facecolor("#0f0f1a")
        ax.imshow(image)
        if heatmap is not None:
            ax.imshow(
                heatmap,
                cmap="hot",
                alpha=0.55,
                interpolation="bilinear",
                extent=(0, image.width, image.height, 0),
            )
        ax.set_title(title, color="#f8fafc")
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    """Load SAN checkpoint and save attention maps for five samples."""
    checkpoint_path = os.path.join(CFG.train.checkpoint_dir, "san_best.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Missing SAN checkpoint: {checkpoint_path}")

    vocab, answer_vocab = _load_vocabularies()
    state_dict = torch.load(checkpoint_path, map_location=CFG.model.device)
    checkpoint_vocab_size = state_dict["embedding.weight"].shape[0]
    checkpoint_answer_size = state_dict["classifier.3.weight"].shape[0]

    model = build_san(checkpoint_vocab_size, checkpoint_answer_size, num_hops=2)
    model.load_state_dict(state_dict)
    model.eval()

    samples = _load_test_samples(5)
    transform = _get_attention_transform()
    output_dir = os.path.join(CFG.train.output_dir, "attention_maps")
    os.makedirs(output_dir, exist_ok=True)

    for index, sample in enumerate(samples, start=1):
        image = sample["image"].convert("RGB")
        question = str(sample["question"])
        image_tensor = transform(image).unsqueeze(0).to(CFG.model.device)
        input_ids = _safe_encode_question(vocab, question, checkpoint_vocab_size)
        input_ids = input_ids.unsqueeze(0).to(CFG.model.device)

        with torch.no_grad():
            outputs = model(images=image_tensor, input_ids=input_ids)

        pred_idx = int(outputs["logits"].argmax(dim=-1).item())
        predicted = answer_vocab.idx2answer.get(pred_idx, "<unk>")
        attention_weights = outputs["attention_weights"]

        output_path = os.path.join(output_dir, f"sample_{index}.png")
        _save_attention_plot(
            image=image,
            question=question,
            predicted=predicted,
            hop1=attention_weights[0][0],
            hop2=attention_weights[1][0],
            output_path=output_path,
        )
        print(f"Saved attention map: {output_path}")


if __name__ == "__main__":
    main()
