import argparse
from collections import deque
from pathlib import Path

from ignite.engine import (
    Events, create_supervised_evaluator, create_supervised_trainer)
from ignite.metrics import Accuracy, Loss
import json_log_plots
import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
import tqdm

from ..data_utils import TRAIN_ROOT, load_train_valid_df
from .dataset import Dataset, get_transform, get_encoded_classes, collate_fn
from .models import build_model


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg('--action', default='train')
    arg('--device', default='cuda', help='device')
    arg('--batch-size', default=16, type=int)
    arg('--workers', default=12, type=int,
        help='number of data loading workers (default: 16)')
    arg('--lr', default=1e-3, type=float, help='initial learning rate')
    arg('--epochs', default=50, type=int,
        help='number of total epochs to run')
    arg('--output-dir', help='path where to save', type=Path)
    arg('--test-only', help='Only test the model', action='store_true')
    arg('--fold', type=int, default=0)
    arg('--n-folds', type=int, default=5)
    args = parser.parse_args()
    print(args)

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    print('Loading data')
    df_train, df_valid = load_train_valid_df(args.fold, args.n_folds)
    classes = get_encoded_classes()
    dataset = Dataset(
        df=df_train,
        transform=get_transform(train=True),
        root=TRAIN_ROOT,
        skip_empty=True,
        classes=classes)
    dataset_test = Dataset(
        df=df_valid,
        transform=get_transform(train=False),
        root=TRAIN_ROOT,
        skip_empty=False,
        classes=classes)
    data_loader = DataLoader(
        dataset,
        num_workers=args.workers,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.batch_size)
    data_loader_test = DataLoader(
        dataset_test,
        batch_size=1,
        collate_fn=collate_fn,
        num_workers=args.workers)

    print('Creating model')
    model = build_model(n_classes=len(classes))
    device = torch.device(args.device)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss = nn.CrossEntropyLoss()

    trainer = create_supervised_trainer(
        model, optimizer, loss, device=device)
    evaluator = create_supervised_evaluator(
        model,
        device=device,
        metrics={
            'accuracy': Accuracy(),
            'loss': Loss(loss),
        })

    epochs_pbar = tqdm.trange(args.epochs)
    epoch_pbar = tqdm.trange(len(data_loader))
    train_losses = deque(maxlen=20)
    step = 0

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(trainer):
        nonlocal step
        train_losses.append(trainer.state.output)
        smoothed_loss = np.mean(train_losses)
        epoch_pbar.set_postfix(loss=f'{smoothed_loss:.4f}')
        epoch_pbar.update(1)
        step += 1
        if step % 20 == 0 and args.action == 'train' and args.output_dir:
            json_log_plots.write_event(
                args.output_dir, step=step * args.batch_size,
                loss=smoothed_loss)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        evaluator.run(data_loader_test)
        metrics = evaluator.state.metrics
        s = trainer.state.epoch
        print(f'Validation Results - Epoch: {s.epoch}  '
              f'Avg accuracy: {metrics["accuracy"]:.4f} '
              f'Avg loss: {metrics["loss"]:.4f}')

    @trainer.on(Events.EPOCH_COMPLETED)
    def checkpoint(_):
        if args.output_dir:
            torch.save(model.state_dict(), args.output_dir / 'model_last.pth')

    @trainer.on(Events.EPOCH_COMPLETED)
    def update_pbars_on_epoch_completion(_):
        epochs_pbar.update(1)
        epoch_pbar.reset()

    print('Starting training')
    trainer.run(data_loader, max_epochs=100)


if __name__ == '__main__':
    main()
