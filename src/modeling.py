from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_tokenizer_and_model(model_name: str, num_labels: int = 2, cache_dir: str | None = None):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, cache_dir=cache_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
        cache_dir=cache_dir,
    )
    return tokenizer, model


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total_params': int(total), 'trainable_params': int(trainable)}


def freeze_base_encoder(model: torch.nn.Module) -> None:
    base = getattr(model, 'base_model', None)
    if base is None:
        raise ValueError('Không tìm thấy base_model để freeze. Hãy kiểm tra kiến trúc model.')
    for p in base.parameters():
        p.requires_grad = False


def _get_encoder_layers(model: torch.nn.Module):
    candidates = [
        ('distilbert.transformer.layer', lambda m: getattr(getattr(getattr(m, 'distilbert', None), 'transformer', None), 'layer', None)),
        ('bert.encoder.layer', lambda m: getattr(getattr(getattr(m, 'bert', None), 'encoder', None), 'layer', None)),
        ('roberta.encoder.layer', lambda m: getattr(getattr(getattr(m, 'roberta', None), 'encoder', None), 'layer', None)),
        ('albert.encoder.albert_layer_groups', lambda m: getattr(getattr(getattr(m, 'albert', None), 'encoder', None), 'albert_layer_groups', None)),
    ]
    for _, getter in candidates:
        layers = getter(model)
        if layers is not None:
            return layers
    return None


def unfreeze_last_n_layers(model: torch.nn.Module, n: int) -> None:
    if n <= 0:
        return
    layers = _get_encoder_layers(model)
    if layers is None:
        print('[WARN] Không tìm thấy encoder layers để unfreeze_last_n_layers; bỏ qua.')
        return
    for layer in list(layers)[-n:]:
        for p in layer.parameters():
            p.requires_grad = True


def apply_lora_if_requested(
    model: torch.nn.Module,
    use_lora: bool,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    lora_target_modules: str = 'q_lin,v_lin',
):
    if not use_lora:
        return model, {'lora_enabled': False}
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise ImportError('Muốn dùng --use_lora cần cài peft: pip install peft') from exc

    targets = [x.strip() for x in lora_target_modules.split(',') if x.strip()]
    config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=targets or None,
    )
    model = get_peft_model(model, config)
    try:
        model.print_trainable_parameters()
    except Exception:
        pass
    return model, {
        'lora_enabled': True,
        'lora_r': lora_r,
        'lora_alpha': lora_alpha,
        'lora_dropout': lora_dropout,
        'lora_target_modules': targets,
    }
