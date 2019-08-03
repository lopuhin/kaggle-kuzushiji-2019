r"""PyTorch Detection Training.

To run in a multi-gpu environment, use the distributed launcher::

    python -m torch.distributed.launch --nproc_per_node=$NGPU --use_env \
        main.py ... --world-size $NGPU

"""
import datetime
from pathlib import Path
import time

import torch
import torch.utils.data
from torch import nn
import torchvision.models.detection
from torchvision.models.detection.rpn import AnchorGenerator

from .group_by_aspect_ratio import \
    GroupedBatchSampler, create_aspect_ratio_groups
from .engine import train_one_epoch, evaluate

from .import utils
from .dataset import Dataset, get_transform
from ..data_utils import TRAIN_ROOT, load_train_valid_df


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    arg = parser.add_argument

    arg('--model', default='faserrcnn_resnet50_fpn', help='model')
    arg('--device', default='cuda', help='device')
    arg('--batch-size', default=2, type=int)
    arg('--epochs', default=13, type=int, metavar='N',
        help='number of total epochs to run')
    arg('--workers', default=4, type=int, metavar='N',
        help='number of data loading workers (default: 16)')
    arg('--lr', default=0.02, type=float, help='initial learning rate')
    arg('--momentum', default=0.9, type=float, metavar='M',
        help='momentum')
    arg('--wd', '--weight-decay', default=1e-4, type=float,
        metavar='W', help='weight decay (default: 1e-4)',
        dest='weight_decay')
    arg('--lr-steps', default=[8, 11], nargs='+', type=int,
        help='decrease lr every step-size epochs')
    arg('--lr-gamma', default=0.1, type=float,
        help='decrease lr by a factor of lr-gamma')
    arg('--print-freq', default=20, type=int, help='print frequency')
    arg('--output-dir', default='.', help='path where to save')
    arg('--resume', default='', help='resume from checkpoint')
    arg('--aspect-ratio-group-factor', default=0, type=int)
    arg('--test-only',
        dest='test_only',
        help='Only test the model',
        action='store_true')
    arg('--pretrained',  # TODO true by default
        dest='pretrained',
        help='Use pre-trained models from the modelzoo',
        action='store_true')

    # fold parameters
    arg('--fold', type=int, default=0)
    arg('--n-folds', type=int, default=5)

    # distributed training parameters
    arg('--world-size', default=1, type=int,
        help='number of distributed processes')
    arg('--dist-url', default='env://',
        help='url used to set up distributed training')

    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    # Data loading code
    print('Loading data')

    df_train, df_valid = load_train_valid_df(args.fold, args.n_folds)
    dataset = Dataset(df_train, get_transform(train=True), TRAIN_ROOT)
    dataset_test = Dataset(df_valid, get_transform(train=False), TRAIN_ROOT)

    print('Creating data loaders')
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = \
            torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    if args.aspect_ratio_group_factor >= 0:
        group_ids = create_aspect_ratio_groups(
            dataset, k=args.aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(
            train_sampler, group_ids, args.batch_size)
    else:
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
    model = torchvision.models.detection.__dict__[args.model](
        num_classes=2,
        pretrained=args.pretrained,
        rpn_anchor_generator=AnchorGenerator(
            sizes=((16, 32, 64, 128),),
            aspect_ratios=((0.5, 1.0, 2.0),),
        ),
        box_detections_per_img=500,
    )
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

    if args.test_only:
        evaluate(model, data_loader_test, device=device)
        return

    print('Start training')
    start_time = time.time()
    for epoch in range(args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, optimizer, data_loader, device, epoch,
                        args.print_freq)
        lr_scheduler.step()
        if args.output_dir:
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'args': args},
                Path(args.output_dir) / f'model_{epoch}.pth')

        # evaluate after every epoch
        evaluate(model, data_loader_test, device=device)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    main()
