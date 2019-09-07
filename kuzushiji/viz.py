"""
Visualization utils, based on
https://www.kaggle.com/anokas/kuzushiji-visualisation
"""
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

import cv2
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from .data_utils import DATA_ROOT, TRAIN_ROOT, UNICODE_MAP


@lru_cache()
def load_font(fontsize: int):
    """ Load font. Download instructions::

        wget -q --show-progress \
            https://noto-website-2.storage.googleapis.com/pkgs/NotoSansCJKjp-hinted.zip
        unzip -p NotoSansCJKjp-hinted.zip NotoSansCJKjp-Regular.otf \
            > data/NotoSansCJKjp-Regular.otf
        rm NotoSansCJKjp-hinted.zip
    """
    return ImageFont.truetype(
        str(DATA_ROOT / 'NotoSansCJKjp-Regular.otf'),
        size=fontsize, encoding='utf-8')


def visualize_training_data(
        image_path: Path,
        labels: List[Tuple[str, int, int, int, int]],
        fontsize: int = 50,
        with_labels: bool = True):
    """ This function takes in a filename of an image,
    and the labels in the string format given in train.csv,
    and returns an image containing the bounding boxes
    and characters annotated.
    """
    # Read image
    img = Image.open(image_path).convert('RGBA')
    if len(labels) == 0:
        return img

    bbox_canvas = Image.new('RGBA', img.size)
    char_canvas = Image.new('RGBA', img.size)
    # Separate canvases for boxes and chars so a box doesn't cut off a character
    bbox_draw = ImageDraw.Draw(bbox_canvas)
    char_draw = ImageDraw.Draw(char_canvas)
    font = load_font(fontsize)

    for codepoint, x, y, w, h in labels:
        x, y, w, h = int(x), int(y), int(w), int(h)
        char = UNICODE_MAP.get(codepoint, codepoint)

        # Draw bounding box around character, and unicode character next to it
        bbox_draw.rectangle((x, y, x+w, y+h), fill=(255, 255, 255, 0),
                            outline=(255, 0, 0, 255), width=4)
        if with_labels:
            char_draw.text((x + w + fontsize/4, y + h/2 - fontsize), char,
                           fill=(0, 0, 255, 255), font=font)

    img = Image.alpha_composite(Image.alpha_composite(img, bbox_canvas),
                                char_canvas)
    img = img.convert('RGB')  # Remove alpha for saving in jpg format.
    return img


BOX_COLOR = (255, 0, 0)


def visualize_box(image: np.ndarray, bbox, color=BOX_COLOR, thickness=2):
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = \
        int(x_min), int(x_min + w), int(y_min), int(y_min + h)
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max),
                  color=color, thickness=thickness)


def visualize_boxes(image: np.ndarray, boxes, **kwargs):
    image = image.copy()
    for idx, bbox in enumerate(boxes):
        visualize_box(image, bbox, **kwargs)
    return image


def visualize_clf_errors(image_id: str, df: pd.DataFrame):
    """ Visualize classification errors (df comes from errors.csv).
    """
    items = df[df['image_id'] == image_id]
    err_items = items[items['pred'] != items['true']]
    err_items_chars = err_items[
        (err_items['true'] != 'seg_fp') & (err_items['pred'] != 'seg_fp')]
    err_items_seg_fp_fn = err_items[err_items['true'] == 'seg_fp']
    err_items_seg_fp_fp = err_items[err_items['pred'] == 'seg_fp']
    good_items = items[items['pred'] == items['true']]
    good_items_seg_fp = good_items[good_items['true'] == 'seg_fp']
    good_items_chars = good_items[good_items['true'] != 'seg_fp']
    title = (
        f'{image_id} acc={1 - len(err_items) / len(items):.2f} '
        f'bad_chars(r)={len(err_items_chars)} '
        f'bad_segfp_fn(y)={len(err_items_seg_fp_fn)} '
        f'bad_segfp_fp(v)={len(err_items_seg_fp_fp)} '
        f'ok_chars(g)={len(good_items_chars)} '
        f'ok_seg_fp(b)={len(good_items_seg_fp)}')
    image = np.array(Image.open(TRAIN_ROOT / f'{image_id}.jpg').convert('RGB'))
    to_boxes = lambda x: [
        (item.x, item.y, item.w, item.h) for item in x.itertuples()]
    image = visualize_boxes(image, to_boxes(err_items_chars), thickness=4)
    image = visualize_boxes(
        image, to_boxes(err_items_seg_fp_fn), thickness=4, color=(255, 255, 0))
    image = visualize_boxes(
        image, to_boxes(err_items_seg_fp_fp), thickness=4, color=(255, 0, 255))
    image = visualize_boxes(
        image, to_boxes(good_items_seg_fp), thickness=4, color=(0, 0, 255))
    image = visualize_boxes(
        image, to_boxes(good_items_chars), thickness=4, color=(0, 255, 0))
    return image, title
