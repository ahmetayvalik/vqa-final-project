"""Model loading utilities for BLIP-2 with LoRA fine-tuning."""

import os

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
import peft.tuners.lora.model as peft_lora_model
from transformers import AutoTokenizer, Blip2ForConditionalGeneration, Blip2Processor, BlipImageProcessor

from src.config import CFG


def load_processor():
    """Load the BLIP-2 processor."""
    image_processor = BlipImageProcessor.from_pretrained(CFG.model.model_name)
    tokenizer = AutoTokenizer.from_pretrained(CFG.model.model_name, use_fast=False)
    processor = Blip2Processor(image_processor=image_processor, tokenizer=tokenizer)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    return processor


def load_model(for_training: bool = False):
    """Load BLIP-2 and optionally attach LoRA adapters to its language model."""
    model = Blip2ForConditionalGeneration.from_pretrained(
        CFG.model.model_name,
        torch_dtype=torch.float16,
    )
    model = model.to(CFG.model.device)

    if for_training:
        for param in model.parameters():
            param.requires_grad = False

        # Native Windows bitsandbytes installs can be import-visible but unusable.
        # Force PEFT to use regular torch LoRA layers because quantization is disabled.
        peft_lora_model.is_bnb_available = lambda: False
        peft_lora_model.is_bnb_4bit_available = lambda: False

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=CFG.model.lora_r,
            lora_alpha=CFG.model.lora_alpha,
            lora_dropout=CFG.model.lora_dropout,
            target_modules=CFG.model.target_modules,
            bias="none",
        )
        model.language_model = get_peft_model(model.language_model, lora_config)
        for name, param in model.named_parameters():
            if "lora" in name.lower():
                param.data = param.data.float()
                param.requires_grad = True
        model.language_model.print_trainable_parameters()
    else:
        model.eval()

    return model


def load_finetuned_model(adapter_path: str):
    """Load BLIP-2 and attach a saved LoRA adapter to its language model."""
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"Adapter path does not exist: {adapter_path}")

    processor = load_processor()
    model = load_model(for_training=False)
    model.language_model = PeftModel.from_pretrained(model.language_model, adapter_path)
    model.eval()
    return processor, model


def count_parameters(model) -> dict[str, float]:
    """Return and print parameter statistics for a model."""
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    trainable_ratio = (trainable_params / total_params * 100.0) if total_params else 0.0

    print("+-------------------+----------------------+")
    print("| Metric            | Value                |")
    print("+-------------------+----------------------+")
    print(f"| Total params      | {total_params:<20,d} |")
    print(f"| Trainable params  | {trainable_params:<20,d} |")
    print(f"| Trainable ratio   | {trainable_ratio:<19.4f}% |")
    print("+-------------------+----------------------+")

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_ratio": trainable_ratio,
    }
