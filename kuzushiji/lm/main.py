import argparse
from collections import deque
import json
from pathlib import Path

from ignite.engine import (
    Events, create_supervised_evaluator, create_supervised_trainer)
from ignite.metrics import Loss
import json_log_plots
import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import tqdm

from ..data_utils import load_train_valid_df, get_encoded_classes
from ..utils import run_with_pbar, format_value, print_metrics
from .dataset import Dataset, TRAIN_TEXTS_PATH
from .models import build_model


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg('--seq-length', type=int, default=12)
    arg('--device', default='cuda', help='device')
    arg('--batch-size', default=64, type=int)
    arg('--lr', default=1e-3, type=float, help='initial learning rate')
    arg('--epochs', default=30, type=int, help='number of total epochs to run')
    arg('--output-dir', help='path where to save')
    arg('--resume', help='resume from checkpoint')
    arg('--test-only', help='Only test the model', action='store_true')
    arg('--workers', default=0, type=int,
        help='number of data loading workers')
    arg('--fold', type=int, default=0)
    arg('--n-folds', type=int, default=5)
    arg('--train-limit', type=int)
    arg('--test-limit', type=int)
    args = parser.parse_args()
    print(args)

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        if not args.resume:
            (output_dir / 'params.json').write_text(
                json.dumps(vars(args), indent=4))

    print('Loading data')
    df_text = pd.read_csv(TRAIN_TEXTS_PATH)
    df_train, df_valid = [
        df_text[df_text['image_id'].isin(df['image_id'])]
        for df in load_train_valid_df(args.fold, args.n_folds)]
    if args.train_limit:
        df_train = df_train.sample(n=args.train_limit, random_state=42)
    if args.test_limit:
        df_valid = df_valid.sample(n=args.test_limit, random_state=42)
    print(f'{len(df_train):,} in train, {len(df_valid):,} in valid')
    classes = get_encoded_classes()

    dataset_kwargs = dict(classes=classes, seq_length=args.seq_length)
    dataset = Dataset(df=df_train, **dataset_kwargs)
    dataset_test = Dataset(df=df_valid, **dataset_kwargs)
    loader_kwargs = dict(batch_size=args.batch_size, num_workers=args.workers)
    data_loader = DataLoader(dataset, shuffle=True, **loader_kwargs)
    data_loader_test = DataLoader(dataset_test, **loader_kwargs)

    print('Creating model')
    model: nn.Module = build_model(n_classes=len(classes))
    print(model)
    device = torch.device(args.device)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss = nn.CrossEntropyLoss()
    step = epoch = 0
    best_loss = float('inf')

    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location='cpu'))

    trainer = create_supervised_trainer(
        model, optimizer, loss, device=device)
    evaluator = create_supervised_evaluator(
        model,
        device=device,
        metrics={
            'loss': Loss(loss),
        })

    epochs_pbar = tqdm.trange(args.epochs)
    epoch_pbar = tqdm.trange(len(data_loader))
    train_losses = deque(maxlen=20)

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(_):
        nonlocal step
        train_losses.append(trainer.state.output)
        smoothed_loss = np.mean(train_losses)
        epoch_pbar.set_postfix(loss=f'{smoothed_loss:.4f}')
        epoch_pbar.update(1)
        step += 1
        if step % 20 == 0 and output_dir:
            json_log_plots.write_event(
                output_dir, step=step * args.batch_size,
                loss=smoothed_loss)

    def evaluate():
        run_with_pbar(evaluator, data_loader_test, desc='evaluate')
        metrics = {
            'valid_loss': evaluator.state.metrics['loss'],
        }
        return metrics

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(_):
        nonlocal best_loss
        metrics = evaluate()
        if output_dir:
            json_log_plots.write_event(
                output_dir, step=step * args.batch_size, **metrics)
        if metrics['valid_loss'] < best_loss:
            best_loss = metrics['valid_loss']
            if output_dir:
                torch.save(model.state_dict(), output_dir / 'model_best.pth')
        epochs_pbar.set_postfix({
            k: format_value(v) for k, v in metrics.items()})

    @trainer.on(Events.EPOCH_COMPLETED)
    def update_pbars_on_epoch_completion(_):
        nonlocal epoch
        epochs_pbar.update(1)
        epoch_pbar.reset()
        epoch += 1

    if args.test_only:
        if not args.resume:
            parser.error('please pass --resume when running with --test-only')
        print_metrics(evaluate())
        return

    trainer.run(data_loader, max_epochs=args.epochs)


if __name__ == '__main__':
    main()
