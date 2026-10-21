from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch

from src.data import prepare_splits
from src.error_analysis import build_error_analysis, save_error_analysis
from src.evaluate import (
    create_model_comparison,
    plot_confusion_matrix,
    plot_training_curves,
    save_epoch_history,
    save_metrics_summary,
)
from src.modeling import (
    apply_lora_if_requested,
    count_parameters,
    freeze_base_encoder,
    load_tokenizer_and_model,
    unfreeze_last_n_layers,
)
from src.tokenization_audit import build_tokenization_audit, render_tokenization_audit_md
from src.train_transformer import create_dataloaders, evaluate_model, predict_dataframe, train_model
from src.utils import ensure_dir, get_device, save_json, set_seed
from src.vi_preprocess import segment_splits
from src.wandb_utils import init_wandb, log_epoch, log_final, safe_finish


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='CSC4007 Lab 5: Transformer fine-tuning for sentiment classification')
    ap.add_argument('--dataset', default='imdb', choices=['imdb', 'local_csv'])
    ap.add_argument('--data_path', default=None)
    ap.add_argument('--split_dir', default=None)
    ap.add_argument('--output_dir', default='outputs')
    ap.add_argument('--language', choices=['en','vi'], default='en')
    ap.add_argument('--vi_segment', choices=['none','presegmented','underthesea','vncorenlp'], default='none')
    ap.add_argument('--vncorenlp_dir', default=None)
    ap.add_argument('--lab2_metrics_path', default=None)
    ap.add_argument('--text_col', default='text')
    ap.add_argument('--label_col', default='label')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max_rows', type=int, default=None)

    ap.add_argument('--model_name', default='distilbert-base-uncased')
    ap.add_argument('--cache_dir', default=None)
    ap.add_argument('--max_length', type=int, default=256)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--weight_decay', type=float, default=0.01)
    ap.add_argument('--warmup_ratio', type=float, default=0.06)
    ap.add_argument('--grad_accum_steps', type=int, default=1)
    ap.add_argument('--patience', type=int, default=2)
    ap.add_argument('--min_delta', type=float, default=1e-4)
    ap.add_argument('--use_amp', action='store_true', help='Mixed precision on CUDA only')
    ap.add_argument('--prefer_mps', action='store_true', help='Use Apple Silicon MPS when available')

    # Development / advanced variants
    ap.add_argument('--freeze_encoder', action='store_true', help='Freeze pretrained encoder; train classification head only')
    ap.add_argument('--unfreeze_last_n_layers', type=int, default=0, help='When encoder is frozen, unfreeze last N layers for partial fine-tuning')
    ap.add_argument('--use_lora', action='store_true', help='Advanced: use PEFT/LoRA for parameter-efficient fine-tuning')
    ap.add_argument('--lora_r', type=int, default=8)
    ap.add_argument('--lora_alpha', type=int, default=16)
    ap.add_argument('--lora_dropout', type=float, default=0.1)
    ap.add_argument('--lora_target_modules', default='q_lin,v_lin', help='DistilBERT default: q_lin,v_lin; BERT often: query,value')

    ap.add_argument('--use_wandb', action='store_true')
    ap.add_argument('--wandb_project', default='csc4007-lab5-transformer')
    ap.add_argument('--wandb_entity', default=None)
    ap.add_argument('--wandb_mode', default='online', choices=['online', 'offline', 'disabled'])
    ap.add_argument('--run_name', default=None)
    ap.add_argument('--lab4_metrics_path', default=None, help='Optional: path to Lab 4 outputs/metrics/metrics_summary.json')
    return ap.parse_args()


def infer_fine_tuning_mode(args: argparse.Namespace) -> str:
    if args.use_lora:
        return 'peft_lora'
    if args.freeze_encoder and args.unfreeze_last_n_layers > 0:
        return f'partial_finetune_last_{args.unfreeze_last_n_layers}_layers'
    if args.freeze_encoder:
        return 'frozen_encoder_classifier_head_only'
    return 'full_finetune'


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(prefer_mps=args.prefer_mps)

    out_dir = Path(args.output_dir)
    if 'phobert' in args.model_name.lower():
        if args.language != 'vi' or args.vi_segment == 'none':
            raise ValueError('PhoBERT requires --language vi and segmented input (--vi_segment).')
    if args.lab2_metrics_path:
        if args.dataset != 'local_csv' or args.language != 'vi':
            raise ValueError('--lab2_metrics_path is intended for Vietnamese local_csv runs only.')
        if not args.split_dir:
            raise ValueError('Lab 2 comparison requires --split_dir.')
        if Path(args.split_dir).resolve() != (Path(args.lab2_metrics_path).resolve().parent.parent / 'splits').resolve():
            raise ValueError('Lab 2 metrics and splits must originate from the SAME run.')
    if args.lab4_metrics_path and args.dataset != 'imdb':
        raise ValueError('Lab 4 comparison is valid for IMDB only.')
    for sub in ['logs', 'splits', 'metrics', 'figures', 'models', 'predictions', 'error_analysis']:
        ensure_dir(out_dir / sub)

    splits, label_names = prepare_splits(
        name=args.dataset,
        data_path=args.data_path,
        text_col=args.text_col,
        label_col=args.label_col,
        max_rows=args.max_rows,
        seed=args.seed,
        split_dir=args.split_dir,
    )
    if args.lab2_metrics_path:
        from src.utils import load_json
        source = load_json(args.lab2_metrics_path)
        if any(source.get('splits', {}).get(k) != len(splits[k]) for k in ('train', 'val', 'test')):
            raise ValueError('Lab 2 baseline split sizes differ from selected --split_dir.')
    if args.language == 'vi':
        splits = segment_splits(splits, args.vi_segment, args.vncorenlp_dir)
    save_json({str(i): name for i,name in enumerate(label_names)}, out_dir / 'logs' / 'id2label.json')
    for split_name, split_df in splits.items():
        split_df.to_csv(out_dir / 'splits' / f'{split_name}.csv', index=False)

    tokenizer, model = load_tokenizer_and_model(args.model_name, num_labels=len(label_names), cache_dir=args.cache_dir, label_names=label_names)

    if args.freeze_encoder:
        freeze_base_encoder(model)
        unfreeze_last_n_layers(model, args.unfreeze_last_n_layers)

    model, lora_info = apply_lora_if_requested(
        model,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
    )

    model.to(device)
    param_counts = count_parameters(model)

    audit = build_tokenization_audit(splits, tokenizer=tokenizer, max_length=args.max_length)
    render_tokenization_audit_md(out_dir / 'logs' / 'tokenization_audit.md', audit)

    train_loader, val_loader, test_loader = create_dataloaders(
        splits, tokenizer=tokenizer, max_length=args.max_length, batch_size=args.batch_size, seed=args.seed
    )

    run = init_wandb(args, audit=audit, param_counts=param_counts)

    history, best_state = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        grad_accum_steps=args.grad_accum_steps,
        patience=args.patience,
        min_delta=args.min_delta,
        use_amp=args.use_amp,
        epoch_logger=lambda row: log_epoch(run, row),
    )

    if best_state is not None:
        model.load_state_dict(best_state)

    # Save Hugging Face format for reuse.
    model_save_dir = out_dir / 'models' / 'best_model'
    ensure_dir(model_save_dir)
    try:
        model.save_pretrained(model_save_dir)
        tokenizer.save_pretrained(model_save_dir)
    except Exception as exc:
        print(f'[WARN] Could not save pretrained model format: {exc}')
        torch.save(model.state_dict(), out_dir / 'models' / 'best_model_state_dict.pt')

    val_loss, val_metrics, *_ = evaluate_model(model, val_loader, device)
    test_loss, test_metrics, y_true, y_pred, y_prob = evaluate_model(model, test_loader, device)

    pred_df = predict_dataframe(model, test_loader, device)
    pred_export = splits['test'].reset_index(drop=True).copy()
    pred_export['pred_label'] = pred_df['pred_label']
    pred_export['pred_label_name'] = pred_export['pred_label'].map(dict(enumerate(label_names)))
    for col in pred_df.columns:
        if col.startswith('prob_label_') or col == 'confidence':
            pred_export[col] = pred_df[col]
    if args.dataset == 'imdb':
        pred_export['prob_negative'] = pred_df['prob_label_0']
        pred_export['prob_positive'] = pred_df['prob_label_1']
    pred_export.to_csv(out_dir / 'predictions' / 'test_predictions.csv', index=False)

    save_epoch_history(history, out_dir / 'metrics' / 'epoch_history.csv')
    plot_training_curves(history, out_dir / 'figures')
    plot_confusion_matrix(y_true, y_pred, out_dir / 'figures' / 'confusion_matrix.png', label_names=label_names)

    fine_tuning_mode = infer_fine_tuning_mode(args)
    metrics_summary = {
        'dataset': args.dataset,
        'dataset_path': args.data_path if args.dataset == 'local_csv' else None,
        'language': args.language,
        'label_names': label_names,
        'num_labels': len(label_names),
        'split_dir': args.split_dir,
        'vi_segment': args.vi_segment,
        'seed': args.seed,
        'device': str(device),
        'model_name': args.model_name,
        'fine_tuning_mode': fine_tuning_mode,
        'max_length': args.max_length,
        'batch_size': args.batch_size,
        'epochs_requested': args.epochs,
        'epochs_trained': len(history),
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'warmup_ratio': args.warmup_ratio,
        'grad_accum_steps': args.grad_accum_steps,
        'freeze_encoder': args.freeze_encoder,
        'unfreeze_last_n_layers': args.unfreeze_last_n_layers,
        **param_counts,
        **lora_info,
        'splits': {k: int(len(v)) for k, v in splits.items()},
        'tokenization_audit': audit,
        'val': {'loss': val_loss, **val_metrics},
        'test': {'loss': test_loss, **test_metrics},
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'notes': 'Raw text -> pretrained tokenizer -> pretrained Transformer -> classification head. Best checkpoint selected by validation macro-F1.',
    }
    save_metrics_summary(metrics_summary, out_dir / 'metrics')

    errors = build_error_analysis(pred_export)
    save_error_analysis(errors, out_dir / 'error_analysis', min_expected=5 if args.language == 'vi' else 10, language=args.language)

    create_model_comparison(
        transformer_metrics=metrics_summary,
        output_path=out_dir / 'metrics' / 'model_comparison.csv',
        lab4_metrics_path=args.lab4_metrics_path,
    )

    if args.lab2_metrics_path:
        from src.evaluate import create_transfer_comparison
        create_transfer_comparison(args.lab2_metrics_path, metrics_summary, out_dir / 'metrics' / 'vietnews_comparison.csv')

    run_summary = {
        'dataset': args.dataset,
        'seed': args.seed,
        'model_name': args.model_name,
        'fine_tuning_mode': fine_tuning_mode,
        'test_macro_f1': test_metrics['macro_f1'],
        'test_accuracy': test_metrics['accuracy'],
        'best_val_macro_f1': max(row['val_macro_f1'] for row in history) if history else None,
        'wandb_enabled': bool(run),
        'wandb_mode': args.wandb_mode if args.use_wandb else 'disabled',
    }
    save_json(run_summary, out_dir / 'logs' / 'run_summary.json')

    log_final(run, {
        'best_val_macro_f1': run_summary['best_val_macro_f1'],
        'test_macro_f1': test_metrics['macro_f1'],
        'test_accuracy': test_metrics['accuracy'],
        'test_loss': test_loss,
        'trainable_params': param_counts['trainable_params'],
        'total_params': param_counts['total_params'],
    })
    safe_finish(run)
    print('DONE.')


if __name__ == '__main__':
    main()
