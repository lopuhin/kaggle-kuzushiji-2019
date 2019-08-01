"""
Visualization utils, based on
https://www.kaggle.com/anokas/kuzushiji-visualisation
"""
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import numpy as np

from .utils import DATA_ROOT, UNICODE_MAP


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


def visualize_training_data(image_path: Path, labels: str, fontsize: int = 50):
    """ This function takes in a filename of an image,
    and the labels in the string format given in train.csv,
    and returns an image containing the bounding boxes
    and characters annotated.
    """
    # Read image
    img = Image.open(image_path).convert('RGBA')
    if not labels:
        return img

    # Convert annotation string to array
    labels = np.array(labels.split(' ')).reshape(-1, 5)

    bbox_canvas = Image.new('RGBA', img.size)
    char_canvas = Image.new('RGBA', img.size)
    # Separate canvases for boxes and chars so a box doesn't cut off a character
    bbox_draw = ImageDraw.Draw(bbox_canvas)
    char_draw = ImageDraw.Draw(char_canvas)
    font = load_font(fontsize)

    for codepoint, x, y, w, h in labels:
        x, y, w, h = int(x), int(y), int(w), int(h)
        char = UNICODE_MAP[codepoint]

        # Draw bounding box around character, and unicode character next to it
        bbox_draw.rectangle((x, y, x+w, y+h), fill=(255, 255, 255, 0),
                            outline=(255, 0, 0, 255))
        char_draw.text((x + w + fontsize/4, y + h/2 - fontsize), char,
                       fill=(0, 0, 255, 255), font=font)

    img = Image.alpha_composite(Image.alpha_composite(img, bbox_canvas),
                                char_canvas)
    img = img.convert('RGB')  # Remove alpha for saving in jpg format.
    return img
