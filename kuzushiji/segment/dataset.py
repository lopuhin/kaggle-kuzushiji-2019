from pathlib import Path
from typing import Callable

from PIL import Image
import pandas as pd
import torch.utils.data


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
        # TODO crop and resize, use albumentations
        if self.transform is not None:
            image = self.transform(image)
        return image
