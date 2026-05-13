"""Stacked Attention Network baseline for Visual Question Answering."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from src.config import CFG


class AttentionLayer(nn.Module):
    """Single attention hop over spatial image regions."""

    def __init__(self, img_feat_dim, question_feat_dim, attention_dim=512):
        """Build linear projections for SAN attention."""
        super().__init__()
        self.W_img = nn.Linear(img_feat_dim, attention_dim, bias=False)
        self.W_ques = nn.Linear(question_feat_dim, attention_dim, bias=False)
        self.W_attn = nn.Linear(attention_dim, 1, bias=False)
        self.dropout = nn.Dropout(0.3)

    def forward(self, img_features, question_feat):
        """Apply attention over regions conditioned on question features."""
        img_proj = self.W_img(img_features)
        ques_proj = self.W_ques(question_feat).unsqueeze(1)
        h = torch.tanh(img_proj + ques_proj)
        attn_weights = F.softmax(self.W_attn(self.dropout(h)), dim=1)
        attended = (attn_weights * img_features).sum(dim=1)
        return attended, attn_weights.squeeze(-1)


class SANModel(nn.Module):
    """Stacked Attention Network (Yang et al., 2016) baseline model."""

    def __init__(
        self,
        vocab_size,
        answer_vocab_size,
        num_attention_hops=2,
        embed_dim=512,
        hidden_dim=1024,
        attention_dim=512,
        dropout=0.3,
    ):
        """Initialize vision, question, stacked attention, and classifier blocks."""
        super().__init__()

        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        self.vision_encoder = vgg.features
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

        self.img_feat_dim = 512
        self.num_regions = 196
        self.question_feat_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(
            embed_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )

        self.attention_layers = nn.ModuleList(
            [
                AttentionLayer(self.img_feat_dim, self.question_feat_dim, attention_dim)
                for _ in range(num_attention_hops)
            ]
        )
        self.img_to_question = nn.Linear(self.img_feat_dim, self.question_feat_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, answer_vocab_size),
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, images, input_ids, attention_mask=None, labels=None):
        """Run SAN forward pass and optionally compute CE loss."""
        _ = attention_mask

        with torch.no_grad():
            vgg_features = self.vision_encoder(images)

        batch_size = vgg_features.size(0)
        img_features = vgg_features.view(batch_size, self.img_feat_dim, -1).permute(0, 2, 1)

        embeddings = self.embedding(input_ids)
        gru_out, _ = self.gru(embeddings)
        q_current = gru_out[:, -1, :]

        attn_list = []
        for layer in self.attention_layers:
            img_attended, attn_weights = layer(img_features, q_current)
            attn_list.append(attn_weights)
            q_current = torch.tanh(self.img_to_question(img_attended) + q_current)

        logits = self.classifier(q_current)
        loss = None

        if labels is not None:
            loss = self.loss_fn(logits, labels)
        return {"loss": loss, "logits": logits, "attention_weights": attn_list}


def build_san(vocab_size, answer_vocab_size, num_hops=2) -> SANModel:
    """Instantiate SAN baseline, move to target device, and print stats."""
    model = SANModel(
        vocab_size=vocab_size,
        answer_vocab_size=answer_vocab_size,
        num_attention_hops=num_hops,
    )
    model = model.to(CFG.model.device)

    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"SAN total parameters: {total_params:,}")
    print(f"SAN trainable parameters: {trainable_params:,}")
    return model
