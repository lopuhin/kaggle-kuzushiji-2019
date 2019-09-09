import random
from typing import Dict

import numpy as np
import pandas as pd
import torch.utils.data
import tqdm

from ..data_utils import load_train_df, DATA_ROOT, get_sequences


TRAIN_TEXTS_PATH = DATA_ROOT / 'train-texts.csv'


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, classes: Dict[str, int], seq_length: int):
        self.seq_length = seq_length
        texts = [
            torch.tensor([classes[s] for s in text.split(' ')],
                         dtype=torch.long)
            for text in df['text'].values]
        # TODO use all texts and pad
        self.texts = [t for t in texts if t.shape[0] > seq_length]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        i = random.randint(0, text.shape[0] - self.seq_length - 1)
        return (text[i: i + self.seq_length],
                text[i + 1: i + self.seq_length + 1])


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
