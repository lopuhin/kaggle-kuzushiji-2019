import re
from pathlib import Path

import pandas as pd
import numpy as np


DATA_ROOT = Path(__file__).parent.parent / 'data'
TRAIN_ROOT = DATA_ROOT / 'train_images'

UNICODE_MAP = {codepoint: char for codepoint, char in
               pd.read_csv(DATA_ROOT / 'unicode_translation.csv').values}


def load_train_df():
    df = pd.read_csv(DATA_ROOT / 'train.csv')
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


def _get_book_id(image_id):
    book_id = re.split(r'[_-]', image_id)[0]
    m = re.search(r'^[a-z]+', book_id)
    if m:
        return m.group()
    else:
        return book_id
