import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
import tqdm


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('clf_folder')
    arg('--device', default='cuda')
    arg('--limit', type=int)
    args = parser.parse_args()

    clf_folder = Path(args.clf_folder)
    device = torch.device(args.device)
    train_features, train_ys = torch.load(
        clf_folder / 'train_features.pth', map_location='cpu')
    test_features, test_ys = torch.load(
        clf_folder / 'test_features.pth', map_location='cpu')
    train_features = train_features.to(device)
    if args.limit:
        rng = np.random.RandomState(42)
        index = torch.tensor(
            rng.randint(0, test_features.shape[0], args.limit))
        test_features = test_features[index]
        test_ys = test_ys[index]
    test_features = test_features.to(device)

    cos_sim = nn.CosineSimilarity().to(device)
    pred_ys = []
    for i in tqdm.trange(test_features.shape[0]):
        feature = test_features[i].unsqueeze(0)  # .to(device)
        sim = cos_sim(train_features, feature)
        pred_ys.append(int(train_ys[sim.argmax()]))
    pred_ys = torch.tensor(pred_ys)
    print(f'accuracy:  {(pred_ys == test_ys).float().mean():.4f}')


if __name__ == '__main__':
    main()
