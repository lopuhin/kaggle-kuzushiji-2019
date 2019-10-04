import random
from typing import Callable, Dict, List

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
        color_val_aug: int,
        normalize: bool = True,
        ) -> Callable:
    train_initial_size = 3072  # this value should not matter any more?
    crop_ratio = crop_height / test_height
    crop_min_max_height = tuple(
        int(train_initial_size * crop_ratio * (1 + sign * scale_aug))
        for sign in [-1, 1])
    if train:
        transforms = [
            LongestMaxSizeRandomSizedCrop(
                max_size=train_initial_size,
                min_max_height=crop_min_max_height,
                width=crop_width,
                height=crop_height,
                w2h_ratio=crop_width / crop_height,
            ),
            A.HueSaturationValue(
                hue_shift_limit=color_hue_aug,
                sat_shift_limit=color_sat_aug,
                val_shift_limit=color_val_aug,
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


class LongestMaxSizeRandomSizedCrop(A.RandomSizedCrop):
    """ Combines LongestMaxSize and RandomSizedCrop into one transform.
    """
    def __init__(self, *, max_size, **kwargs):
        super().__init__(**kwargs)
        self.max_size = max_size

    @property
    def target_dependence(self):
        return {'bboxes': ['image']}

    def apply(self, img, crop_height=0, crop_width=0, **params):
        crop_height, crop_width = self._crop_h_w(img, crop_height, crop_width)
        return super().apply(
            img=img, crop_height=crop_height, crop_width=crop_width, **params)

    def apply_to_bbox(
            self, bbox, crop_height=0, crop_width=0, image=None, **params):
        crop_height, crop_width = self._crop_h_w(image, crop_height, crop_width)
        return super().apply_to_bbox(
            bbox, crop_height=crop_height, crop_width=crop_width, **params)

    def _crop_h_w(self, img, crop_height, crop_width):
        max_size = max(img.shape[:2])
        crop_height = int(crop_height * max_size / self.max_size)
        crop_width = int(crop_width * max_size / self.max_size)
        return crop_height, crop_width

    def get_transform_init_args_names(self):
        return super().get_transform_init_args_names() + ('max_size',)


def collate_fn(batch, max_targets=None, target_multiple=32):
    images = torch.stack([img for (img, _, _), _ in batch])
    boxes = [b for (_, b, _), _ in batch]
    sequences = [s for (_, _, s), _ in batch]
    meta = [m for _, (_, m) in batch]
    labels = [l for _, (l, _) in batch]
    if max_targets is not None:
        # Limit and quantize number of targets to have better performance
        # FIXME sequences not supported (they'd need to be re-mapped)
        n_targets = sum(l.shape[0] for l in labels)
        if n_targets > max_targets:
            n_targets_out = max_targets
        else:
            n_targets_out = n_targets // target_multiple * target_multiple
        if n_targets != n_targets_out:
            assert n_targets > n_targets_out
            n_out = 0
            for i in range(len(batch)):
                n_image = labels[i].shape[0]
                if i == len(batch) - 1:
                    n_out_image = n_targets_out - n_out
                else:
                    n_out_image = max(
                        1, int(n_image * n_targets_out / n_targets))
                if n_out_image != n_image:
                    image_indices = torch.randperm(n_image)[:n_out_image]
                    labels[i] = labels[i][image_indices]
                    boxes[i] = boxes[i][image_indices, :]
                n_out += n_out_image
    labels = torch.cat(labels)
    return (images, boxes, sequences), (labels, meta)


class Dataset(torch.utils.data.Dataset):
    def __init__(
            self, *, df: pd.DataFrame, transforms: List[Callable],
            resample_empty: bool, classes: Dict[str, int],
            ):
        self.df = df
        self.transforms = transforms
        self.resample_empty = resample_empty
        self.classes = classes

    def __len__(self):
        return len(self.df) * len(self.transforms)

    def __getitem__(self, idx):
        item = self.df.iloc[idx // len(self.transforms)]
        transform = self.transforms[idx % len(self.transforms)]
        del idx
        image = read_image(get_image_path(item))
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
        xy = transform(**xy)
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
