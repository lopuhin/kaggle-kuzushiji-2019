import argparse
from collections import deque
import json
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
    TRAIN_ROOT, TEST_ROOT, load_train_valid_df, load_train_df, to_coco,
    SEG_FP, from_coco, get_target_boxes_labels, scaled_boxes,
    get_encoded_classes)
from ..utils import run_with_pbar, print_metrics, format_value
from ..metric import score_boxes, get_metrics
from .dataset import Dataset, get_transform, collate_fn, get_labels
from .models import build_model, get_output


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg('clf_gt', help='segmentation predictions')
    arg('--base', default='resnet50')
    arg('--device', default='cuda', help='device')
    arg('--batch-size', default=12, type=int)
    arg('--workers', default=12, type=int,
        help='number of data loading workers')
    arg('--lr', default=2.5e-5, type=float, help='initial learning rate')
    arg('--epochs', default=30, type=int, help='number of total epochs to run')
    arg('--output-dir', help='path where to save')
    arg('--resume', help='resume from checkpoint')
    arg('--test-only', help='Only test the model', action='store_true')
    arg('--submission', help='Create submission', action='store_true')
    arg('--fold', type=int, default=0)
    arg('--n-folds', type=int, default=5)
    arg('--repeat-train', type=int, default=4)
    arg('--train-limit', type=int)
    arg('--test-limit', type=int)
    args = parser.parse_args()
    if args.test_only and args.submission:
        parser.error('pass one of --test-only and --submission')
    print(args)

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        if not args.resume:
            (output_dir / 'params.json').write_text(
                json.dumps(vars(args), indent=4))

    print('Loading data')
    df_train_gt, df_valid_gt = load_train_valid_df(args.fold, args.n_folds)
    df_clf_gt = load_train_df(args.clf_gt)
    if args.submission:
        df_valid = df_train = df_clf_gt
        empty_index = df_valid['labels'] == ''
        empty_pages = df_valid[empty_index]['image_id'].values
        df_valid = df_valid[~empty_index]
        root = TEST_ROOT
    else:
        df_train, df_valid = [
            df_clf_gt[df_clf_gt['image_id'].isin(set(df['image_id']))]
            for df in [df_train_gt, df_valid_gt]]
        df_valid = df_valid[df_valid['labels'] != '']
        root = TRAIN_ROOT
    if args.train_limit:
        df_train = df_train.sample(n=args.train_limit, random_state=42)
    if args.test_limit:
        df_valid = df_valid.sample(n=args.test_limit, random_state=42)
    gt_by_image_id = {item.image_id: item for item in df_valid_gt.itertuples()}
    print(f'{len(df_train):,} in train, {len(df_valid):,} in valid')
    classes = get_encoded_classes()

    dataset = Dataset(
        df=pd.concat([df_train] * args.repeat_train),
        transform=get_transform(train=True),
        root=root,
        resample_empty=True,
        classes=classes)
    dataset_test = Dataset(
        df=df_valid,
        transform=get_transform(train=False),
        root=root,
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
    model: nn.Module = build_model(base=args.base, n_classes=len(classes))
    print(model)
    device = torch.device(args.device)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss = nn.CrossEntropyLoss()
    step = epoch = 0
    best_f1 = 0

    if args.resume:
        state = torch.load(args.resume, map_location='cpu')
        if 'optimizer' in state:
            optimizer.load_state_dict(state['optimizer'])
            model.load_state_dict(state['model'])
            step = state['step']
            epoch = state['epoch']
            best_f1 = state['best_f1']
        else:
            model.load_state_dict(state)
        del state

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
            'detailed': GetDetailedPrediction(classes),
        })

    epochs_left = args.epochs - epoch
    epochs_pbar = tqdm.trange(epochs_left)
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

    @trainer.on(Events.EPOCH_COMPLETED)
    def checkpoint(_):
        if output_dir:
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': step,
                'epoch': epoch,
                'best_f1': best_f1,
            }, output_dir / 'checkpoint.pth')

    def evaluate():
        run_with_pbar(evaluator, data_loader_test, desc='evaluate')
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
            pred_centers = np.array([p['center'] for p in prediction])
            pred_labels = [p['cls'] for p in prediction]
            scores.append(
                dict(score_boxes(
                    truth_boxes=from_coco(target_boxes).numpy(),
                    truth_label=target_labels,
                    preds_center=pred_centers,
                    preds_label=np.array(pred_labels),
                ), image_id=item.image_id))
        metrics.update(get_metrics(scores))
        if output_dir:
            pd.DataFrame(evaluator.state.metrics['detailed']).to_csv(
                output_dir / 'detailed.csv.gz', index=None)
        return metrics

    def make_submission():
        run_with_pbar(evaluator, data_loader_test, desc='evaluate')
        submission = []
        for prediction, meta in tqdm.tqdm(
                evaluator.state.metrics['predictions']):
            submission.append({
                'image_id': meta['image_id'],
                'labels': ' '.join(
                    ' '.join([p['cls']] +
                             [str(int(round(v))) for v in p['center']])
                    for p in prediction),
            })
        submission.extend(
            {'image_id': image_id, 'labels': ''} for image_id in empty_pages)
        pd.DataFrame(submission).to_csv(
            output_dir / f'submission_{output_dir.name}.csv.gz',
            index=None)
        pd.DataFrame(evaluator.state.metrics['detailed']).to_csv(
            output_dir / 'test_detailed.csv.gz', index=None)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(_):
        nonlocal best_f1
        metrics = evaluate()
        if output_dir:
            json_log_plots.write_event(
                output_dir, step=step * args.batch_size, **metrics)
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
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

    if args.test_only or args.submission:
        if not args.resume:
            parser.error('please pass --resume when running with --test-only '
                         'or --submission')
        if args.test_only:
            print_metrics(evaluate())
        elif args.submission:
            if not output_dir:
                parser.error('--output-dir required with --submission')
            make_submission()
        return

    trainer.run(data_loader, max_epochs=epochs_left)


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
            {'center': (float(x) * meta['scale_w'],
                        float(y) * meta['scale_h']),
             'cls': cls}
            for x, y, cls in zip(centers_x, centers_y, classes)
            if cls != SEG_FP]
        self._predictions.append((prediction, meta))

    def compute(self):
        return self._predictions


class GetDetailedPrediction(Metric):
    def __init__(self, classes: Dict[str, int], top_k=50, **kwargs):
        self._classes = {idx: cls for cls, idx in classes.items()}
        self._detailed = []
        self._top_k = top_k
        super().__init__(**kwargs)

    def reset(self):
        self._detailed.clear()

    def update(self, output):
        (y_pred_full, (boxes,)), (y, (meta,)) = output
        y_pred = y_pred_full.argmax(dim=1)
        boxes = to_coco(scaled_boxes(boxes, meta['scale_w'], meta['scale_h']))
        assert y_pred.shape == y.shape == (boxes.shape[0],)
        top_k_classes, top_k_logits = _get_top_k(y_pred_full, self._top_k)
        for i, (box, y_pred_i, y_i) in enumerate(zip(boxes, y_pred, y)):
            x, y, w, h = map(float, box)
            self._detailed.append(dict(
                image_id=meta['image_id'],
                x=x,
                y=y,
                w=w,
                h=h,
                pred=self._classes[int(y_pred_i)],
                true=self._classes[int(y_i)],
                **_top_k_entry(top_k_classes, top_k_logits, i),
            ))

    def compute(self):
        return self._detailed


def _get_top_k(y_pred, top_k: int):
    top_k_logits, top_k_classes = (
        v.cpu().numpy() for v in torch.topk(y_pred, top_k, dim=1))
    return top_k_classes, top_k_logits


def _top_k_entry(top_k_classes, top_k_logits, i):
    return {
        'top_k_logits': ' '.join(f'{v:.4f}' for v in top_k_logits[i]),
        'top_k_classes': ' '.join(map(str, top_k_classes[i])),
    }


if __name__ == '__main__':
    main()
