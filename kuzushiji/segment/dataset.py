import random
from pathlib import Path
from typing import Callable

import albumentations as A
from albumentations.pytorch import ToTensor
import numpy as np
from PIL import Image
import pandas as pd
import torch.utils.data


def get_transform(train: bool) -> Callable:
    train_initial_size = 2048
    crop_min_max_height = (400, 533)
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
        ToTensor(),
    ])
    return A.Compose(
        transforms,
        bbox_params={
            'format': 'coco',
            'min_area': 0,
            'min_visibility': 0.5,
            'label_fields': ['labels'],
        },
    )


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, transform: Callable, root: Path,
                 skip_empty: bool):
        self.df = df
        self.root = root
        self.transform = transform
        self.skip_empty = skip_empty

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        image_path = self.root / f'{item.image_id}.jpg'
        image = Image.open(image_path).convert('RGB')
        if item.labels:
            labels = np.array(item.labels.split(' ')).reshape(-1, 5)
        else:
            labels = np.zeros((0, 5))
        bboxes = labels[:, 1:].astype(np.float)
        # clip bboxes
        bboxes[:, 2] = (np.minimum(bboxes[:, 0] + bboxes[:, 2], image.width)
                        - bboxes[:, 0])
        bboxes[:, 3] = (np.minimum(bboxes[:, 1] + bboxes[:, 3], image.height)
                        - bboxes[:, 1])
        xy = {
            'image': np.array(image),
            'bboxes': bboxes,
            'labels': np.ones(labels.shape[0], dtype=np.long),
        }
        xy = self.transform(**xy)
        if not xy['bboxes'] and self.skip_empty:
            return self[random.randint(0, len(self.df) - 1)]
        image = xy['image']
        boxes = torch.tensor(xy['bboxes']).reshape((len(xy['bboxes']), 4))
        # convert to pytorch detection format
        boxes[:, 2] += boxes[:, 0]
        boxes[:, 3] += boxes[:, 1]
        target = {
            'boxes': boxes,
            'labels': torch.tensor(xy['labels'], dtype=torch.long),
            'idx': torch.tensor(idx),
        }
        return image, target
