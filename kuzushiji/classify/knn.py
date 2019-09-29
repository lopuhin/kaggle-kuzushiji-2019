import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
import tqdm

from ..data_utils import load_train_df


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('clf_folder')
    arg('--device', default='cuda')
    arg('--limit', type=int, help='evaluate only on some pages')
    args = parser.parse_args()

    clf_folder = Path(args.clf_folder)
    device = torch.device(args.device)
    train_features, train_ys = torch.load(
        clf_folder / 'train_features.pth', map_location='cpu')
    test_features, test_ys = torch.load(
        clf_folder / 'test_features.pth', map_location='cpu')
    train_features = train_features.to(device)
    df_detailed = pd.read_csv(clf_folder / 'detailed.csv.gz')
    df_train = load_train_df()
    image_ids = sorted(set(df_detailed['image_id'].values))
    if args.limit:
        rng = np.random.RandomState(42)
        image_ids = rng.choice(image_ids, args.limit)
        index = torch.tensor(df_detailed['image_id'].isin(image_ids).values)
        test_features = test_features[index]
        test_ys = test_ys[index]
    df_train = df_train[df_train['image_id'].isin(image_ids)]

    cos_sim = nn.CosineSimilarity().to(device)
    pred_ys = []
    for i in tqdm.trange(test_features.shape[0]):
        feature = test_features[i].unsqueeze(0).to(device)
        sim = cos_sim(train_features, feature)
        pred_ys.append(int(train_ys[sim.argmax()]))
    pred_ys = torch.tensor(pred_ys)
    print(f'accuracy:  {(pred_ys == test_ys).float().mean():.4f}')


if __name__ == '__main__':
    main()
