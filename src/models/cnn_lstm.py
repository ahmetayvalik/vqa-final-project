"""Classic CNN+LSTM baseline model for Visual Question Answering."""

import torch
import torch.nn as nn
import torchvision.models as models

from src.config import CFG


class CNNLSTMModel(nn.Module):
    """CNN+LSTM baseline architecture for VQA classification."""

    def __init__(
        self,
        vocab_size,
        answer_vocab_size,
        embed_dim=512,
        hidden_dim=1024,
        dropout=0.3,
    ):
        """Initialize vision encoder, question encoder, and classifier layers."""
        super().__init__()

        backbone = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
        self.vision_encoder = nn.Sequential(*list(backbone.children())[:-1])
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

        image_feat_dim = 2048
        self.image_projector = nn.Sequential(
            nn.Linear(image_feat_dim, embed_dim),
            nn.ReLU(),
        )

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )

        self.fusion = nn.Sequential(
            nn.Linear(embed_dim + hidden_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, answer_vocab_size),
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, images, input_ids, attention_mask=None, labels=None):
        """Compute logits (and loss when labels are provided) for VQA answers."""
        _ = attention_mask

        with torch.no_grad():
            img_feat = self.vision_encoder(images)
        img_feat = img_feat.flatten(start_dim=1)
        img_feat = self.image_projector(img_feat)

        embeddings = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embeddings)
        q_feat = lstm_out[:, -1, :]

        fused = torch.cat([img_feat, q_feat], dim=-1)
        fused = self.fusion(fused)
        logits = self.classifier(fused)
        loss = None

        if labels is not None:
            loss = self.loss_fn(logits, labels)
        return {"loss": loss, "logits": logits}


def build_cnn_lstm(vocab_size, answer_vocab_size) -> CNNLSTMModel:
    """Instantiate CNN+LSTM baseline, move to target device, and print stats."""
    model = CNNLSTMModel(vocab_size=vocab_size, answer_vocab_size=answer_vocab_size)
    model = model.to(CFG.model.device)

    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"CNN+LSTM total parameters: {total_params:,}")
    print(f"CNN+LSTM trainable parameters: {trainable_params:,}")
    return model
