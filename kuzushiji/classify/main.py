import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..data_utils import TRAIN_ROOT, load_train_valid_df
from .dataset import Dataset, get_transform, get_encoded_classes


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg('--model', default='fasterrcnn_resnet50_fpn', help='model')
    arg('--device', default='cuda', help='device')
    arg('--batch-size', default=16, type=int)
    arg('--workers', default=4, type=int,
        help='number of data loading workers (default: 16)')
    arg('--lr', default=1e-4, type=float, help='initial learning rate')
    arg('--epochs', default=50, type=int,
        help='number of total epochs to run')
    arg('--output-dir', help='path where to save')
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
        batch_size=args.batch_size)
    data_loader_test = DataLoader(
        dataset_test,
        batch_size=1,
        num_workers=args.workers)

    print('Creating model')
    model = build_model(args.model)
    device = torch.device(args.device)
    model.to(device)


def build_model(name: str) -> nn.Module:
    pass


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


if __name__ == '__main__':
    main()
