from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from .utils import ensure_dir, load_json, save_json


def compute_classification_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'precision_macro': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'recall_macro': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
    }


def save_epoch_history(history: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    pd.DataFrame(history).to_csv(path, index=False)


def plot_training_curves(history: list[dict[str, Any]], fig_dir: str | Path) -> None:
    fig_dir = ensure_dir(fig_dir)
    if not history:
        return
    df = pd.DataFrame(history)

    plt.figure(figsize=(7, 4))
    plt.plot(df['epoch'], df['train_loss'], marker='o', label='train_loss')
    plt.plot(df['epoch'], df['val_loss'], marker='o', label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training / Validation Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / 'loss_curve.png', dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(df['epoch'], df['val_accuracy'], marker='o', label='val_accuracy')
    plt.plot(df['epoch'], df['val_macro_f1'], marker='o', label='val_macro_f1')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.title('Validation Metrics')
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / 'metric_curve.png', dpi=160)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, path: str | Path, label_names: list[str] | None = None) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    label_names = label_names or ['negative', 'positive']
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(label_names))))
    fig, ax = plt.subplots(figsize=(max(5, len(label_names)), max(4, len(label_names)*0.85)))
    im = ax.imshow(cm)
    ax.set_xticks(range(len(label_names)))
    ax.set_yticks(range(len(label_names)))
    ax.set_xticklabels(label_names, rotation=45, ha='right')
    ax.set_yticklabels(label_names)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_metrics_summary(metrics: dict[str, Any], output_dir: str | Path) -> None:
    output_dir = ensure_dir(output_dir)
    save_json(metrics, output_dir / 'metrics_summary.json')
    lines = [
        '# Metrics summary',
        '',
        f"- Dataset: `{metrics.get('dataset')}`",
        f"- Model: `{metrics.get('model_name')}`",
        f"- Fine-tuning mode: `{metrics.get('fine_tuning_mode')}`",
        f"- Seed: `{metrics.get('seed')}`",
        f"- Device: `{metrics.get('device')}`",
        f"- Max length: `{metrics.get('max_length')}`",
        f"- Trainable params: `{metrics.get('trainable_params')}` / `{metrics.get('total_params')}`",
        '',
        '## Validation',
        '',
        f"- Loss: `{metrics['val']['loss']:.4f}`",
        f"- Accuracy: `{metrics['val']['accuracy']:.4f}`",
        f"- Macro-F1: `{metrics['val']['macro_f1']:.4f}`",
        '',
        '## Test',
        '',
        f"- Loss: `{metrics['test']['loss']:.4f}`",
        f"- Accuracy: `{metrics['test']['accuracy']:.4f}`",
        f"- Macro-F1: `{metrics['test']['macro_f1']:.4f}`",
        '',
        '## Notes',
        '',
        metrics.get('notes', ''),
    ]
    (output_dir / 'metrics_summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def create_model_comparison(
    transformer_metrics: dict[str, Any],
    output_path: str | Path,
    lab4_metrics_path: str | None = None,
) -> None:
    rows = []
    if lab4_metrics_path:
        try:
            prev = load_json(lab4_metrics_path)
            rows.append({
                'source_lab': 'Lab 4',
                'model': prev.get('model', prev.get('model_type', 'lstm/gru')),
                'accuracy': prev.get('test', {}).get('accuracy'),
                'macro_f1': prev.get('test', {}).get('macro_f1'),
                'trainable_params': prev.get('trainable_params'),
                'notes': 'Imported from Lab 4 metrics_summary.json',
            })
        except Exception as exc:
            rows.append({'source_lab': 'Lab 4', 'model': 'N/A', 'notes': f'Could not read Lab 4 metrics: {exc}'})
    rows.append({
        'source_lab': 'Lab 5',
        'model': transformer_metrics.get('model_name'),
        'fine_tuning_mode': transformer_metrics.get('fine_tuning_mode'),
        'accuracy': transformer_metrics.get('test', {}).get('accuracy'),
        'macro_f1': transformer_metrics.get('test', {}).get('macro_f1'),
        'trainable_params': transformer_metrics.get('trainable_params'),
        'notes': 'Transformer fine-tuning result',
    })
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def create_transfer_comparison(lab2_metrics_path: str, transformer_metrics: dict, output_path: str | Path) -> None:
    lab2 = load_json(lab2_metrics_path)
    if lab2.get('dataset') != 'local_csv' or transformer_metrics['dataset'] != 'local_csv':
        raise ValueError('Both experiments must use local_csv')
    rows = [
        {'lab':'Lab 2', 'model': f"{lab2.get('vectorizer')} + {lab2.get('model')}", 'accuracy':lab2['test']['accuracy'], 'macro_f1':lab2['test']['macro_f1']},
        {'lab':'Lab 5', 'model': transformer_metrics['model_name'], 'accuracy':transformer_metrics['test']['accuracy'], 'macro_f1':transformer_metrics['test']['macro_f1']},
    ]
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    pd.DataFrame(rows).to_csv(output_path, index=False)
