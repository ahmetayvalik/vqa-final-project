# Visual Question Answering — Model Comparison

A course project comparing classical visual question answering models with a zero-shot vision-language baseline on VQA v2. The repository contains training and inference scripts, model implementations, experiment outputs, and the project report.

## Scope

The study evaluates a question-only baseline, CNN–LSTM, Stacked Attention Network (SAN), and BLIP-2 zero-shot inference. The models were compared within the project's experimental setup on an NVIDIA RTX 4060 laptop GPU (8 GB VRAM).

## Reported results

| Model | Reported accuracy |
| --- | ---: |
| Question-only | 18.35% |
| BLIP-2, zero-shot | 33.33% |
| SAN | 35.80% |
| CNN–LSTM | 41.20% |

These are results reported by this project, not general VQA v2 leaderboard figures. Differences in model setup, training data, and evaluation conditions can affect comparability.

## Repository contents

- `src/models/`: model implementations
- `src/train_classical.py`: classical-model training
- `src/evaluate.py`: evaluation utilities
- `src/vqa_demo.py`: inference entry point
- `results/`: experiment outputs and analyses
- `VQA_Final_Rapor.docx`: full academic report

## Setup

```bash
pip install -r requirements.txt
```

Dataset files, compatible model weights, and local paths may need to be configured separately. Review the scripts and `src/config.py` before running a full experiment.

## Example commands

```bash
python main.py --mode cnn_lstm
python main.py --mode san
python main.py --mode full
python src/vqa_demo.py --image photo.jpg --question "What is in the image?"
```

The demo and training entry points may require local datasets or model checkpoints.

## Notes

The project examines the trade-offs between classical architectures and pretrained vision-language models. Reported metrics should be interpreted within their respective experimental conditions. For detailed methodology, refer to `VQA_Final_Rapor.docx`.

## Author

Ahmet Ayvalık — Artificial Intelligence Engineering, OSTİM Technical University
