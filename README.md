# Visual Question Answering — Final Project

A comparative Visual Question Answering project evaluating classical deep learning baselines and a modern vision-language model on VQA v2.

## Project Overview

- 3 models: CNN+LSTM, SAN, BLIP-2
- Dataset: VQA v2
- Hardware: NVIDIA RTX 4060 (8GB VRAM)

## Results

| Model | Accuracy |
|-------|----------|
| Question-Only | 18.35% |
| BLIP-2 Zero-shot | 33.33% |
| SAN | 35.80% |
| CNN+LSTM | 41.20% |

## Project Structure

```text
vqa_project/
├── data/
│   ├── images/
│   └── vqa/
├── notebooks/
│   └── demo.ipynb
├── report_assets/
├── results/
│   ├── attention_maps/
│   ├── checkpoints/
│   └── error_analysis/
├── src/
│   ├── models/
│   │   ├── cnn_lstm.py
│   │   ├── llava.py
│   │   └── san.py
│   ├── attention_viz.py
│   ├── config.py
│   ├── dataset.py
│   ├── dataset_classical.py
│   ├── dataset_local.py
│   ├── evaluate.py
│   ├── finetune.py
│   ├── generate_report_figures.py
│   ├── inference.py
│   ├── question_only_baseline.py
│   ├── train_classical.py
│   ├── visualize.py
│   └── vqa_demo.py
├── main.py
├── README.md
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Demo with BLIP-2
python src/vqa_demo.py --image photo.jpg --question "What is in the image?"

# Demo with CNN+LSTM (no download needed)
python src/vqa_demo.py --image photo.jpg --question "What color is the car?" --model cnn_lstm

# Train CNN+LSTM
python main.py --mode cnn_lstm

# Train SAN
python main.py --mode san

# Run all models
python main.py --mode full
```

## Key Findings

- More data ≠ higher accuracy without hyperparameter tuning
- SAN requires more data to outperform CNN+LSTM
- BLIP-2 achieves 33.33% accuracy without any training (zero-shot)
- Visual features contribute +22.85 accuracy points over text-only

## References

1. Antol, S., Agrawal, A., Lu, J., Mitchell, M., Batra, D., Zitnick, C. L., & Parikh, D. (2015). VQA: Visual Question Answering.
2. Ren, M., Kiros, R., & Zemel, R. (2015). Exploring Models and Data for Image Question Answering.
3. Yang, Z., He, X., Gao, J., Deng, L., & Smola, A. (2016). Stacked Attention Networks for Image Question Answering.
4. Simonyan, K., & Zisserman, A. (2015). Very Deep Convolutional Networks for Large-Scale Image Recognition.
5. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition.
6. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory.
7. Chung, J., Gulcehre, C., Cho, K., & Bengio, Y. (2014). Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling.
8. Li, J., Li, D., Savarese, S., & Hoi, S. (2023). BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models.
9. Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). LoRA: Low-Rank Adaptation of Large Language Models.

## Author

Ahmet Ayvalık — 230212041  
Derin Öğrenme Final Projesi, 2026
