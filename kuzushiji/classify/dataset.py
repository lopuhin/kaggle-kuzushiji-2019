import random
from pathlib import Path
from typing import Callable, Dict

import albumentations as A
from albumentations.pytorch import ToTensor
import numpy as np
import pandas as pd
import torch.utils.data

from ..data_utils import get_image_path, read_image, get_sequences


def get_transform(
        *,
        train: bool,
        test_height: int,
        crop_width: int,
        crop_height: int,
        scale_aug: float,
        color_hue_aug: int,
        color_sat_aug: int,
        color_value_aug: int,
        normalize: bool = True,
        ) -> Callable:
    train_initial_size = 3072
    crop_ratio = crop_height / test_height
    crop_min_max_height = tuple(int(crop_ratio * (1 + sign * scale_aug))
                                for sign in [-1, 1])
    if train:
        transforms = [
            A.LongestMaxSize(max_size=train_initial_size),
            A.RandomSizedCrop(
                min_max_height=crop_min_max_height,
                width=crop_width,
                height=crop_height,
                w2h_ratio=crop_width / crop_height,
            ),
            A.HueSaturationValue(
                hue_shift_limit=7,
                sat_shift_limit=30,
                val_shift_limit=30,
            ),
            A.RandomBrightnessContrast(),
            A.RandomGamma(),
        ]
    else:
        transforms = [
            A.LongestMaxSize(max_size=test_height),
        ]
    if normalize:
        transforms.append(A.Normalize())
    transforms.extend([
        ToTensor(),
    ])
    return A.Compose(
        transforms,
        bbox_params={
            'format': 'coco',
            'min_area': 0,
            'min_visibility': 0.99,
            'label_fields': ['labels'],
        },
    )


def collate_fn(batch):
    images = torch.stack([img for (img, _, _), _ in batch])
    boxes = [b for (_, b, _), _ in batch]
    sequences = [s for (_, _, s), _ in batch]
    labels = torch.cat([l for _, (l, _) in batch])
    meta = [m for _, (_, m) in batch]
    return (images, boxes, sequences), (labels, meta)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, *, df: pd.DataFrame, transform: Callable, root: Path,
                 resample_empty: bool, classes: Dict[str, int]):
        self.df = df
        self.root = root
        self.transform = transform
        self.resample_empty = resample_empty
        self.classes = classes

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        image = read_image(get_image_path(item, self.root))
        original_h, original_w, _ = image.shape
        if item.labels:
            labels = np.array(item.labels.split(' ')).reshape(-1, 5)
        else:
            labels = np.zeros((0, 5))
        bboxes = labels[:, 1:].astype(np.float)
        # clip bboxes
        height, width, _ = image.shape
        bboxes[:, 2] = (np.minimum(bboxes[:, 0] + bboxes[:, 2], width)
                        - bboxes[:, 0])
        bboxes[:, 3] = (np.minimum(bboxes[:, 1] + bboxes[:, 3], height)
                        - bboxes[:, 1])
        xy = {
            'image': image,
            'bboxes': bboxes,
            'labels': [self.classes[c] for c in labels[:, 0]],
        }
        xy = self.transform(**xy)
        if len(xy['bboxes']) == 0:
            if self.resample_empty:
                return self[random.randint(0, len(self) - 1)]
            else:
                raise ValueError('empty bboxes not expected')
        image = xy['image']
        boxes = torch.tensor(xy['bboxes']).reshape((len(xy['bboxes']), 4))
        sequences = [torch.tensor(seq) for seq in get_sequences(xy['bboxes'])]
        # convert to pytorch detection format
        boxes[:, 2] += boxes[:, 0]
        boxes[:, 3] += boxes[:, 1]
        labels = torch.tensor(xy['labels'], dtype=torch.long)
        _, h, w = image.shape
        meta = {
            'image_id': item.image_id,
            'scale_h': original_h / h,
            'scale_w': original_w / w,
        }
        return (image, boxes, sequences), (labels, meta)


def get_labels(y):
    labels, meta = y
    return labels
