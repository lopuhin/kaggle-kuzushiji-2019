from pathlib import Path
from typing import Callable

import albumentations as A
from albumentations.pytorch import ToTensor
import numpy as np
from PIL import Image
import pandas as pd
import torch.utils.data


def get_transform(train: bool) -> Callable:
    transforms = [
        A.LongestMaxSize(max_size=2048),  # all pages to have the same size
        A.RandomSizedCrop(  # TODO a different one for train=False
            min_max_height=(500, 750),
            width=384,
            height=256,
            w2h_ratio=384 / 256,
        ),
        ToTensor(),
    ]
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
    def __init__(self, df: pd.DataFrame, transform: Callable, root: Path):
        self.df = df
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        image_path = self.root / f'{item.image_id}.jpg'
        image = Image.open(image_path).convert('RGB')
        labels = np.array(item.labels.split(' ')).reshape(-1, 5)
        xy = {
            'image': np.array(image),
            'bboxes': labels[:, 1:].astype(np.float),
            'labels': np.ones(labels.shape[0], dtype=np.int64),
        }
        xy = self.transform(**xy)
        image = xy['image']
        target = {
            'boxes': xy['bboxes'],
            'labels': xy['labels'],
        }
        return image, target
