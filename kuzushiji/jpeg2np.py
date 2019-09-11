import argparse

import numpy as np
import tqdm

from kuzushiji.data_utils import read_image, TRAIN_ROOT, get_image_np_path


def main():
    parser = argparse.ArgumentParser()
    _ = parser.parse_args()
    paths = list(TRAIN_ROOT.glob('*.jpg'))
    for path in tqdm.tqdm(paths):
        np_path = get_image_np_path(path)
        if not np_path.exists():
            image = read_image(path)
            np.save(np_path, image)


if __name__ == '__main__':
    main()
