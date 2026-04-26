from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from .evaluate import compute_classification_metrics


class TextClassificationDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def create_dataloaders(splits: dict[str, pd.DataFrame], tokenizer, max_length: int, batch_size: int, seed: int = 42):
    datasets = {
        name: TextClassificationDataset(
            df['text'].tolist(), df['label'].astype(int).tolist(), tokenizer=tokenizer, max_length=max_length
        )
        for name, df in splits.items()
    }
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(datasets['train'], batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(datasets['val'], batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(datasets['test'], batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) for k, v in batch.items()}


def train_one_epoch(
    model,
    loader: DataLoader,
    optimizer,
    scheduler,
    device: torch.device,
    grad_accum_steps: int = 1,
    max_grad_norm: float = 1.0,
    use_amp: bool = False,
) -> float:
    model.train()
    losses = []
    optimizer.zero_grad(set_to_none=True)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type == 'cuda')

    for step, batch in enumerate(tqdm(loader, desc='train', leave=False), start=1):
        batch = _move_batch(batch, device)
        with torch.cuda.amp.autocast(enabled=use_amp and device.type == 'cuda'):
            outputs = model(**batch)
            loss = outputs.loss / grad_accum_steps
        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if step % grad_accum_steps == 0 or step == len(loader):
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu().item() * grad_accum_steps))
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def evaluate_model(model, loader: DataLoader, device: torch.device) -> tuple[float, dict, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    losses = []
    all_labels, all_preds, all_probs = [], [], []
    for batch in tqdm(loader, desc='eval', leave=False):
        batch = _move_batch(batch, device)
        outputs = model(**batch)
        loss = outputs.loss
        logits = outputs.logits.detach().cpu()
        probs = torch.softmax(logits, dim=-1).numpy()
        preds = probs.argmax(axis=1)
        labels = batch['labels'].detach().cpu().numpy()
        losses.append(float(loss.detach().cpu().item()))
        all_labels.extend(labels.tolist())
        all_preds.extend(preds.tolist())
        all_probs.extend(probs.tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)
    metrics = compute_classification_metrics(y_true, y_pred)
    return float(np.mean(losses)) if losses else 0.0, metrics, y_true, y_pred, y_prob


def train_model(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    warmup_ratio: float,
    grad_accum_steps: int,
    patience: int,
    min_delta: float,
    use_amp: bool,
    epoch_logger: Callable[[dict], None] | None = None,
):
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
    total_steps = max(1, (len(train_loader) // max(1, grad_accum_steps)) * max(1, epochs))
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    history: list[dict] = []
    best_state = None
    best_val_f1 = -1.0
    wait = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            grad_accum_steps=grad_accum_steps,
            use_amp=use_amp,
        )
        val_loss, val_metrics, *_ = evaluate_model(model, val_loader, device)
        row = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_metrics['accuracy'],
            'val_macro_f1': val_metrics['macro_f1'],
            'val_precision_macro': val_metrics['precision_macro'],
            'val_recall_macro': val_metrics['recall_macro'],
            'lr': optimizer.param_groups[0]['lr'],
        }
        history.append(row)
        if epoch_logger is not None:
            epoch_logger(row)
        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={row['val_accuracy']:.4f} | val_f1={row['val_macro_f1']:.4f}")

        if row['val_macro_f1'] > best_val_f1 + min_delta:
            best_val_f1 = row['val_macro_f1']
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f'Early stopping at epoch {epoch}; best val macro-F1={best_val_f1:.4f}')
                break
    return history, best_state


@torch.no_grad()
def predict_dataframe(model, loader: DataLoader, device: torch.device) -> pd.DataFrame:
    _, _, y_true, y_pred, y_prob = evaluate_model(model, loader, device)
    return pd.DataFrame({
        'true_label': y_true,
        'pred_label': y_pred,
        'prob_negative': y_prob[:, 0] if y_prob.ndim == 2 and y_prob.shape[1] > 0 else np.nan,
        'prob_positive': y_prob[:, 1] if y_prob.ndim == 2 and y_prob.shape[1] > 1 else np.nan,
    })
