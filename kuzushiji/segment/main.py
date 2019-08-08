r"""PyTorch Detection Training.

To run in a multi-gpu environment, use the distributed launcher::

    python -m torch.distributed.launch --nproc_per_node=$NGPU --use_env \
        main.py ... --world-size $NGPU

"""
import datetime
from pathlib import Path
import time

import json_log_plots
import pandas as pd
import torch
import torch.utils.data
from torch import nn
import detection
from detection.rpn import AnchorGenerator
from detection.transform import GeneralizedRCNNTransform

from .engine import train_one_epoch, evaluate

from .import utils
from .dataset import Dataset, get_transform
from ..data_utils import TRAIN_ROOT, load_train_valid_df


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    arg = parser.add_argument

    arg('--model', default='fasterrcnn_resnet50_fpn', help='model')
    arg('--device', default='cuda', help='device')
    arg('--batch-size', default=16, type=int)
    arg('--workers', default=4, type=int, metavar='N',
        help='number of data loading workers (default: 16)')
    arg('--lr', default=0.01, type=float, help='initial learning rate')
    arg('--momentum', default=0.9, type=float, metavar='M',
        help='momentum')
    arg('--wd', '--weight-decay', default=1e-4, type=float,
        metavar='W', help='weight decay (default: 1e-4)',
        dest='weight_decay')
    arg('--epochs', default=30, type=int, metavar='N',
        help='number of total epochs to run')
    arg('--lr-steps', default=[24, 28], nargs='+', type=int,
        help='decrease lr every step-size epochs')
    arg('--lr-gamma', default=0.1, type=float,
        help='decrease lr by a factor of lr-gamma')
    arg('--print-freq', default=100, type=int, help='print frequency')
    arg('--output-dir', help='path where to save')
    arg('--resume', default='', help='resume from checkpoint')
    arg('--test-only',
        help='Only test the model',
        action='store_true')
    arg('--pretrained',  # TODO true by default
        help='Use pre-trained models from the modelzoo', action='store_true')
    arg('--score-threshold', type=float, default=0.5)
    arg('--nms-threshold', type=float, default=0.5)

    # fold parameters
    arg('--fold', type=int, default=0)
    arg('--n-folds', type=int, default=5)

    # distributed training parameters
    arg('--world-size', default=1, type=int,
        help='number of distributed processes')
    arg('--dist-url', default='env://',
        help='url used to set up distributed training')

    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    # Data loading code
    print('Loading data')

    df_train, df_valid = load_train_valid_df(args.fold, args.n_folds)
    dataset = Dataset(
        df_train, get_transform(train=True), TRAIN_ROOT, skip_empty=False)
    dataset_test = Dataset(
        df_valid, get_transform(train=False), TRAIN_ROOT, skip_empty=False)

    print('Creating data loaders')
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = \
            torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    train_batch_sampler = torch.utils.data.BatchSampler(
        train_sampler, args.batch_size, drop_last=True)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=train_batch_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn)

    print('Creating model')
    model = build_model(args.model, args.pretrained, args.nms_threshold)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu])
        model_without_ddp = model.module

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum,
        weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        print(f'Loaded from checkpoint {args.resume}')

    def save_eval_results(er):
        if output_dir:
            pd.DataFrame(er).to_csv(output_dir / 'eval.csv', index=None)

    if args.test_only:
        _, eval_results = evaluate(
            model, data_loader_test, device=device, output_dir=output_dir,
            threshold=args.score_threshold)
        save_eval_results(eval_results)
        return

    print('Start training')
    start_time = time.time()
    for epoch in range(args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_metrics = train_one_epoch(
            model, optimizer, data_loader, device, epoch, args.print_freq)
        lr_scheduler.step()
        if output_dir:
            json_log_plots.write_event(output_dir, step=epoch, **train_metrics)
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'args': args},
                output_dir / f'model_{epoch}.pth')

        # evaluate after every epoch
        eval_metrics, eval_results = evaluate(
            model, data_loader_test, device=device, output_dir=None,
            threshold=args.score_threshold)
        save_eval_results(eval_results)
        if output_dir:
            json_log_plots.write_event(output_dir, step=epoch, **eval_metrics)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def build_model(name: str, pretrained: bool, nms_threshold: float):
    anchor_sizes = [12, 24, 32, 64, 96]
    model = detection.__dict__[name](
        num_classes=2,
        pretrained=pretrained,
        rpn_anchor_generator=AnchorGenerator(
            sizes=tuple((s,) for s in anchor_sizes),
            aspect_ratios=tuple((0.5, 1.0, 2.0) for _ in anchor_sizes),
        ),
        box_detections_per_img=500,
        box_nms_thresh=nms_threshold,
    )
    model.transform = ModelTransform(
        image_mean=model.transform.image_mean,
        image_std=model.transform.image_std,
    )
    return model


class ModelTransform(GeneralizedRCNNTransform):
    def __init__(self, image_mean, image_std):
        nn.Module.__init__(self)
        self.image_mean = image_mean
        self.image_std = image_std

    def resize(self, image, target):
        return image, target


if __name__ == '__main__':
    main()
