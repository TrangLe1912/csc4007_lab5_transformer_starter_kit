from __future__ import annotations

from typing import Any


def init_wandb(args, audit: dict[str, Any] | None = None, param_counts: dict[str, int] | None = None):
    if not getattr(args, 'use_wandb', False) or getattr(args, 'wandb_mode', 'disabled') == 'disabled':
        return None
    try:
        import wandb
    except ImportError:
        print('[WARN] wandb chưa được cài. Bỏ qua W&B logging.')
        return None

    config = vars(args).copy()
    if audit:
        config['tokenization_audit'] = audit
    if param_counts:
        config.update(param_counts)
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        name=args.run_name,
        config=config,
    )
    return run


def log_epoch(run, row: dict[str, Any]) -> None:
    if run is not None:
        run.log(row, step=row.get('epoch'))


def log_final(run, metrics: dict[str, Any]) -> None:
    if run is not None:
        run.summary.update(metrics)


def safe_finish(run) -> None:
    if run is not None:
        run.finish()
