import re
from pathlib import Path
from typing import Dict

import cv2
import pandas as pd
import numpy as np
import torch


DATA_ROOT = Path(__file__).parent.parent / 'data'
TRAIN_ROOT = DATA_ROOT / 'train_images'

UNICODE_MAP = {codepoint: char for codepoint, char in
               pd.read_csv(DATA_ROOT / 'unicode_translation.csv').values}


SEG_FP = 'seg_fp'  # false positive from segmentation


def load_train_df(path=DATA_ROOT / 'train.csv'):
    df = pd.read_csv(path)
    df['labels'].fillna(value='', inplace=True)
    return df


def load_train_valid_df(fold: int, n_folds: int):
    df = load_train_df()
    df['book_id'] = df['image_id'].apply(_get_book_id)
    book_ids = np.array(sorted(set(df['book_id'].values)))
    with_counts = list(zip(
        book_ids,
        df.groupby('book_id')['image_id'].agg('count').loc[book_ids].values))
    with_counts.sort(key=lambda x: x[1])
    valid_book_ids = [book_id for i, (book_id, _) in enumerate(with_counts)
                      if i % n_folds == fold]
    train_book_ids = [book_id for book_id in book_ids
                      if book_id not in valid_book_ids]
    return tuple(df[df['book_id'].isin(ids)].copy()
                 for ids in [train_book_ids, valid_book_ids])


def get_image_path(item, root: Path) -> Path:
    return root / f'{item.image_id}.jpg'


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def get_target_boxes_labels(item):
    if item.labels:
        labels = np.array(item.labels.split(' ')).reshape(-1, 5)
    else:
        labels = np.zeros((0, 5))
    boxes = labels[:, 1:].astype(np.float)
    labels = labels[:, 0]
    return boxes, labels


def print_metrics(metrics: Dict):
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f'{k}: {v:.4f}')
        else:
            print(f'{k}: {v}')


def _get_book_id(image_id):
    book_id = re.split(r'[_-]', image_id)[0]
    m = re.search(r'^[a-z]+', book_id)
    if m:
        return m.group()
    else:
        return book_id


def to_coco(boxes: torch.Tensor) -> torch.Tensor:
    """ Convert from pytorch detection format to COCO format.
    """
    boxes = boxes.clone()
    boxes[:, 2] -= boxes[:, 0]
    boxes[:, 3] -= boxes[:, 1]
    return boxes


def from_coco(boxes: torch.Tensor) -> torch.Tensor:
    """ Convert from CODO to pytorch detection format.
    """
    boxes = boxes.clone()
    boxes[:, 2] += boxes[:, 0]
    boxes[:, 3] += boxes[:, 1]
    return boxes
