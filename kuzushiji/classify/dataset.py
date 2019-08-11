import random
from pathlib import Path
from typing import Callable, Dict

import albumentations as A
from albumentations.pytorch import ToTensor
import cv2
import numpy as np
import pandas as pd
import torch.utils.data

from ..data_utils import load_train_df


def get_transform(train: bool) -> Callable:
    train_initial_size = 2048  # TODO higher?
    crop_min_max_height = (400, 533)
    # TODO smaller crop, swap height/width
    crop_width = 512
    crop_height = 384
    if train:
        transforms = [
            A.LongestMaxSize(max_size=train_initial_size),
            A.RandomSizedCrop(
                min_max_height=crop_min_max_height,
                width=crop_width,
                height=crop_height,
                w2h_ratio=crop_width / crop_height,
            ),
            #A.HueSaturationValue(
            #    hue_shift_limit=7,
            #    sat_shift_limit=30,
            #    val_shift_limit=30,
            #),
            #A.RandomBrightnessContrast(),
            #A.RandomGamma(),
        ]
    else:
        test_size = int(train_initial_size *
                        crop_height / np.mean(crop_min_max_height))
        print(f'Test image max size {test_size} px')
        transforms = [
            A.LongestMaxSize(max_size=test_size),
        ]
    transforms.extend([
        A.Normalize(),
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
    images = torch.stack([img for (img, _), _ in batch])
    boxes = [b for (_, b), _ in batch]
    labels = torch.cat([l for (_, _), l in batch])
    return (images, boxes), labels


def get_encoded_classes() -> Dict[str, int]:
    classes = set()
    df_train = load_train_df()
    for s in df_train['labels'].values:
         x = s.split()
         classes.update(x[i] for i in range(0, len(x), 5))
    return {cls: i for i, cls in enumerate(sorted(classes))}


class Dataset(torch.utils.data.Dataset):
    def __init__(self, *, df: pd.DataFrame, transform: Callable, root: Path,
                 skip_empty: bool, classes: Dict[str, int]):
        self.df = df
        self.root = root
        self.transform = transform
        self.skip_empty = skip_empty
        self.classes = classes

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        image = cv2.imread(str(self.root / f'{item.image_id}.jpg'))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
        if not xy['bboxes'] and self.skip_empty:
            return self[random.randint(0, len(self.df) - 1)]
        image = xy['image']
        boxes = torch.tensor(xy['bboxes']).reshape((len(xy['bboxes']), 4))
        # convert to pytorch detection format
        boxes[:, 2] += boxes[:, 0]
        boxes[:, 3] += boxes[:, 1]
        labels = torch.tensor(xy['labels'], dtype=torch.long)
        return (image, boxes), labels
