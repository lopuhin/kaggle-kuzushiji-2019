from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch.utils.data
import tqdm

from ..data_utils import load_train_df, DATA_ROOT


TRAIN_TEXTS_PATH = DATA_ROOT / 'train-texts.csv'


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, classes: Dict[str, int], seq_length: int):
        texts = [
            torch.tensor([classes[s] for s in text.split(' ')],
                         dtype=torch.long)
            for text in df['text'].values]
        # FIXME this will oversample text from longer sequences
        # TODO maybe instead text should have "line breaks"?
        self.data = [
            (text[i: i + seq_length],
             text[i + 1: i + seq_length + 1])
            for text in texts
            for i in range(0, text.shape[0] - seq_length)
            if text.shape[0] > seq_length]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_sequences(
        boxes: List[Tuple[float, float, float, float]],
        ) -> List[List[int]]:
    """ Return a list of sequences from bounding boxes.
    """
    # TODO expand tall boxes
    next_indices = {}
    for i, box in enumerate(boxes):
        x0, y0, w, h = box
        x1, y1 = x0 + w, y0 + h
        bx0 = boxes[:, 0]
        bx1 = boxes[:, 0] + boxes[:, 2]
        by0 = boxes[:, 1]
        by1 = boxes[:, 1] + boxes[:, 3]
        w_intersecting = (
            ((bx0 >= x0) & (bx0 <= x1)) |
            ((bx1 >= x0) & (bx1 <= x1)) |
            ((x0 >= bx0) & (x0 <= bx1)) |
            ((x1 >= bx0) & (x1 <= bx1))
        )
        higher = w_intersecting & (by0 < y0)
        higher_indices, = higher.nonzero()
        if higher_indices.shape[0] > 0:
            closest = higher_indices[np.argmax(by1[higher_indices])]
            next_indices[closest] = i
    next_indices_values = set(next_indices.values())
    starts = {i for i in range(len(boxes)) if i not in next_indices_values}
    sequences = []
    for i in starts:
        seq = [i]
        next_idx = next_indices.get(i)
        while next_idx is not None:
            seq.append(next_idx)
            next_idx = next_indices.get(next_idx)
        sequences.append(seq)
    return sequences


def main():
    """ Convert dataset to sequences.
    """
    df = load_train_df()
    data = []
    for item in tqdm.tqdm(df.itertuples(), total=len(df)):
        if not item.labels:
            continue
        labels = np.array(item.labels.split(' ')).reshape(-1, 5)
        sequences = get_sequences(labels[:, 1:].astype(float))
        for seq in sequences:
            data.append({
                'image_id': item.image_id,
                'text': ' '.join(labels[i, 0] for i in seq),
            })
    pd.DataFrame(data).to_csv(TRAIN_TEXTS_PATH, index=None)


if __name__ == '__main__':
    main()