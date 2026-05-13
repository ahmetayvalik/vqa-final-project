"""Command-line VQA demo for BLIP-2 and CNN+LSTM models."""

import argparse
import os
import sys

import torch
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import CFG
from src.dataset_classical import AnswerVocabulary, Vocabulary, get_val_transforms
from src.inference import predict_single
from src.models.cnn_lstm import build_cnn_lstm
from src.models.llava import load_model, load_processor


def save_visualization(image: Image.Image, question: str, answer: str, model_name: str) -> str:
    """Save image with question and answer annotation."""
    os.makedirs(CFG.train.output_dir, exist_ok=True)
    output_path = os.path.join(CFG.train.output_dir, "demo_output.png")

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(image)
    ax.axis("off")
    ax.set_title(f"Model: {model_name}\nQ: {question}\nA: {answer}", fontsize=13, pad=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return output_path


def run_blip2(image: Image.Image, question: str) -> str:
    """Run BLIP-2 inference and return answer."""
    processor = load_processor()
    model = load_model(for_training=False)
    return predict_single(processor, model, image, question)


def run_cnn_lstm(image: Image.Image, question: str) -> tuple[str, list[tuple[str, float]]]:
    """Run trained CNN+LSTM inference and return top-3 predictions."""
    vocab_path = os.path.join(CFG.train.output_dir, "vocab.json")
    answer_vocab_path = os.path.join(CFG.train.output_dir, "answer_vocab.json")
    ckpt_path = os.path.join(CFG.train.checkpoint_dir, "cnn_lstm_best.pt")

    if not os.path.exists(vocab_path) or not os.path.exists(answer_vocab_path):
        raise FileNotFoundError("Missing results/vocab.json or results/answer_vocab.json.")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError("Missing results/checkpoints/cnn_lstm_best.pt.")

    vocab = Vocabulary.load(vocab_path)
    answer_vocab = AnswerVocabulary.load(answer_vocab_path)
    state_dict = torch.load(ckpt_path, map_location=CFG.model.device)
    vocab_size = state_dict["embedding.weight"].shape[0]
    answer_vocab_size = state_dict["classifier.3.weight"].shape[0]

    model = build_cnn_lstm(vocab_size, answer_vocab_size)
    model.load_state_dict(state_dict)
    model.eval()

    image_tensor = get_val_transforms()(image).unsqueeze(0).to(CFG.model.device)
    question_tensor = vocab.encode(question, CFG.data.max_question_length)
    question_tensor[question_tensor >= vocab_size] = vocab.word2idx.get("<unk>", 1)
    question_tensor = question_tensor.unsqueeze(0).to(CFG.model.device)

    with torch.no_grad():
        outputs = model(images=image_tensor, input_ids=question_tensor)
        probs = torch.softmax(outputs["logits"], dim=-1)
        top_probs, top_indices = torch.topk(probs, k=3, dim=-1)

    top_predictions = []
    for prob, idx in zip(top_probs[0].detach().cpu(), top_indices[0].detach().cpu()):
        answer = answer_vocab.idx2answer.get(int(idx.item()), "<unk>")
        top_predictions.append((answer, float(prob.item())))

    return top_predictions[0][0], top_predictions


def main() -> None:
    """Run VQA on a single image/question pair."""
    parser = argparse.ArgumentParser(description="Run a VQA demo.")
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument("--question", required=True, help="Question to ask about the image.")
    parser.add_argument(
        "--model",
        choices=["blip2", "cnn_lstm"],
        default="blip2",
        help="Model backend to use.",
    )
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")
    if args.model == "blip2":
        answer = run_blip2(image, args.question)
        top_predictions = None
    else:
        answer, top_predictions = run_cnn_lstm(image, args.question)

    save_visualization(image, args.question, answer, args.model)

    print(f"Question: {args.question}")
    print(f"Answer: {answer}")
    if top_predictions is not None:
        print("Top-3 predictions:")
        for rank, (pred_answer, confidence) in enumerate(top_predictions, start=1):
            print(f"{rank}. {pred_answer} ({confidence * 100:.2f}%)")


if __name__ == "__main__":
    main()
