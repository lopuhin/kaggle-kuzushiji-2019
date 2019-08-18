import argparse
from collections import deque
from pathlib import Path
import pandas as pd
from typing import Dict

from ignite.engine import (
    Events, create_supervised_evaluator, create_supervised_trainer)
from ignite.metrics import Accuracy, Loss, Metric
from ignite.utils import convert_tensor
import json_log_plots
import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
import tqdm

from ..data_utils import (
    TRAIN_ROOT, load_train_valid_df, load_train_df, SEG_FP, from_coco,
    get_target_boxes_labels, print_metrics)
from ..metric import score_boxes, get_metrics
from .dataset import (
    Dataset, get_transform, get_encoded_classes, collate_fn, get_labels)
from .models import build_model, get_output


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg('clf_gt', help='segmentation predictions')
    arg('--base', default='resnet50')
    arg('--device', default='cuda', help='device')
    arg('--batch-size', default=12, type=int)
    arg('--workers', default=12, type=int,
        help='number of data loading workers (default: 16)')
    arg('--lr', default=2.5e-5, type=float, help='initial learning rate')
    arg('--epochs', default=30, type=int,
        help='number of total epochs to run')
    arg('--output-dir', help='path where to save', type=Path)
    arg('--resume', help='resume from checkpoint')
    arg('--test-only', help='Only test the model', action='store_true')
    arg('--fold', type=int, default=0)
    arg('--n-folds', type=int, default=5)
    arg('--repeat-train', type=int, default=4)
    args = parser.parse_args()
    print(args)

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    print('Loading data')
    df_train_gt, df_valid_gt = load_train_valid_df(args.fold, args.n_folds)
    df_clf_gt = load_train_df(args.clf_gt)
    df_train, df_valid = [
        df_clf_gt[df_clf_gt['image_id'].isin(set(df['image_id']))]
        for df in [df_train_gt, df_valid_gt]]
    df_valid = df_valid[df_valid['labels'] != '']
    gt_by_image_id = {item.image_id: item for item in df_valid_gt.itertuples()}
    print(f'{len(df_train):,} in train, {len(df_valid):,} in valid')
    classes = get_encoded_classes()
    dataset = Dataset(
        df=pd.concat([df_train] * args.repeat_train),
        transform=get_transform(train=True),
        root=TRAIN_ROOT,
        resample_empty=True,
        classes=classes)
    dataset_test = Dataset(
        df=df_valid,
        transform=get_transform(train=False),
        root=TRAIN_ROOT,
        resample_empty=False,
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
    model = build_model(base=args.base, n_classes=len(classes))
    print(model)
    device = torch.device(args.device)
    model.to(device)

    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location='cpu'))

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss = nn.CrossEntropyLoss()

    trainer = create_supervised_trainer(
        model, optimizer,
        loss_fn=lambda y_pred, y: loss(get_output(y_pred), get_labels(y)),
        device=device,
        prepare_batch=_prepare_batch,
    )

    def get_y_pred_y(output):
        y_pred, y = output
        return get_output(y_pred), get_labels(y)

    evaluator = create_supervised_evaluator(
        model,
        device=device,
        prepare_batch=_prepare_batch,
        metrics={
            'accuracy': Accuracy(output_transform=get_y_pred_y),
            'loss': Loss(loss, output_transform=get_y_pred_y),
            'predictions': GetPredictions(classes),
        })

    epochs_pbar = tqdm.trange(args.epochs)
    epoch_pbar = tqdm.trange(len(data_loader))
    train_losses = deque(maxlen=20)
    step = 0

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(_):
        nonlocal step
        train_losses.append(trainer.state.output)
        smoothed_loss = np.mean(train_losses)
        epoch_pbar.set_postfix(loss=f'{smoothed_loss:.4f}')
        epoch_pbar.update(1)
        step += 1
        if step % 20 == 0 and args.output_dir:
            json_log_plots.write_event(
                args.output_dir, step=step * args.batch_size,
                loss=smoothed_loss)

    @trainer.on(Events.EPOCH_COMPLETED)
    def checkpoint(_):
        if args.output_dir:
            torch.save(model.state_dict(), args.output_dir / 'model_last.pth')

    def evaluate():
        _run_with_pbar(evaluator, data_loader_test, desc='evaluate')
        metrics = {
            'valid_loss': evaluator.state.metrics['loss'],
            'accuracy': evaluator.state.metrics['accuracy'],
        }
        scores = []
        for prediction, meta in tqdm.tqdm(
                evaluator.state.metrics['predictions'], desc='metrics'):
            item = gt_by_image_id[meta['image_id']]
            target_boxes, target_labels = get_target_boxes_labels(item)
            target_boxes = torch.from_numpy(target_boxes)
            pred_centers = np.array([
                [x * meta['scale_w'], y * meta['scale_h']]
                for (x, y), _ in prediction])
            pred_labels = [l for _, l in prediction]
            scores.append(
                dict(score_boxes(
                    truth_boxes=from_coco(target_boxes).numpy(),
                    truth_label=target_labels,
                    preds_center=pred_centers,
                    preds_label=np.array(pred_labels),
                ), image_id=item.image_id))
        metrics.update(get_metrics(scores))
        return metrics

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(_):
        metrics = evaluate()
        if args.output_dir:
            json_log_plots.write_event(
                args.output_dir, step=step * args.batch_size, **metrics)
        epochs_pbar.set_postfix({k: f'{v:.4f}' for k, v in metrics.items()})

    @trainer.on(Events.EPOCH_COMPLETED)
    def update_pbars_on_epoch_completion(_):
        epochs_pbar.update(1)
        epoch_pbar.reset()

    if args.test_only:
        if not args.resume:
            parser.error('please pass --resume when running with --test-only')
        metrics = evaluate()
        print_metrics(metrics)

    trainer.run(data_loader, max_epochs=args.epochs)


def _run_with_pbar(engine, loader, desc=None):
    pbar = tqdm.trange(len(loader), desc=desc)
    engine.on(Events.ITERATION_COMPLETED)(lambda _: pbar.update(1))
    engine.run(loader)
    pbar.close()


def _prepare_batch(batch, device=None, non_blocking=False):
    x, (y, meta) = batch
    return (
        convert_tensor(x, device=device, non_blocking=non_blocking),
        (convert_tensor(y, device=device, non_blocking=non_blocking), meta))


class GetPredictions(Metric):
    def __init__(self, classes: Dict[str, int], *args, **kwargs):
        self._predictions = []
        self._classes = {idx: cls for cls, idx in classes.items()}
        super().__init__(*args, **kwargs)

    def reset(self):
        self._predictions.clear()

    def update(self, output):
        (y_pred, (boxes,)), (_, (meta,)) = output
        classes = [self._classes[int(idx)] for idx in y_pred.argmax(dim=1)]
        centers_x = 0.5 * (boxes[:, 0] + boxes[:, 2])
        centers_y = 0.5 * (boxes[:, 1] + boxes[:, 3])
        prediction = [
            ((float(x), float(y)), cls)
            for x, y, cls in zip(centers_x, centers_y, classes)
            if cls != SEG_FP]
        self._predictions.append((prediction, meta))

    def compute(self):
        return self._predictions


if __name__ == '__main__':
    main()
