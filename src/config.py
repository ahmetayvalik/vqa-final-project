"""Centralized configuration module for the VQA project."""

import os
from dataclasses import asdict, dataclass, field
from pprint import pformat
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _default_device() -> str:
    """Return the default compute device based on CUDA availability."""
    return "cuda" if torch is not None and torch.cuda.is_available() else "cpu"


def _default_torch_dtype() -> Any:
    """Return float16 on CUDA and float32 otherwise."""
    if torch is None:
        return "float32"
    return torch.float16 if torch.cuda.is_available() else torch.float32


@dataclass
class ModelConfig:
    """Model and adaptation settings for VQA training and inference."""

    model_name: str = "Salesforce/blip2-opt-2.7b"
    device: str = field(default_factory=_default_device)
    torch_dtype: Any = field(default_factory=_default_torch_dtype)
    quantization: str = "none"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])


@dataclass
class QuantizationConfig:
    """Disabled quantization settings retained for config compatibility."""

    load_in_4bit: bool = False
    bnb_4bit_compute_dtype: Any = field(
        default_factory=lambda: torch.float16 if torch is not None else "float16"
    )
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True


@dataclass
class DataConfig:
    """Dataset and preprocessing settings for VQA experiments."""

    dataset_name: str = "lmms-lab/VQAv2"
    train_samples: int = 50000
    val_samples: int = 2000
    test_samples: int = 100
    max_question_length: int = 64
    max_answer_length: int = 10
    batch_size: int = 4
    num_workers: int = 4


@dataclass
class TrainConfig:
    """Optimization and checkpoint settings for training."""

    epochs: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    grad_accum_steps: int = 4
    warmup_ratio: float = 0.1
    output_dir: str = "results"
    checkpoint_dir: str = "results/checkpoints"

    def __post_init__(self) -> None:
        """Ensure training output directories exist."""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)


@dataclass
class ProjectConfig:
    """Top-level config that groups model, data, and training settings."""

    model: ModelConfig = field(default_factory=ModelConfig)
    quant_cfg: QuantizationConfig = field(default_factory=QuantizationConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


QUANT_CFG = QuantizationConfig()
CFG = ProjectConfig()


def print_config() -> None:
    """Pretty-print the active project configuration."""
    print(pformat(asdict(CFG), sort_dicts=False))
