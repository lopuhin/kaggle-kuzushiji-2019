import argparse
from collections import defaultdict

import pandas as pd

from ..data_utils import SEG_FP


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('detailed', help='predictions on test in "detailed_*" format')
    arg('output', help='path where predictions are written as ground truth')
    arg('--min-gap', type=float, default=5,
        help='min score gap between top and second prediction')
    arg('--max-second-score', type=float, default=10,
        help='max second prediction score')
    arg('--drop-seg-fp', type=int, default=0)
    args = parser.parse_args()
    df = pd.read_csv(args.detailed)

    by_image_id = defaultdict(list)
    n_kept = 0
    for item in df.itertuples():
        top, second, *_ = map(float, item.top_k_logits.split())
        if (top - second >= args.min_gap and
                second <= args.max_second_score and
                (not args.drop_seg_fp or item.pred != SEG_FP)):
            n_kept += 1
            by_image_id[item.image_id].append(
                ' '.join(map(str, [
                    item.pred,
                    int(item.x),
                    int(item.y),
                    int(item.w),
                    int(item.h),
                ])))

    pseudolabeled = pd.DataFrame(
        [{'image_id': image_id, 'labels': ' '.join(labels)}
         for image_id, labels in sorted(by_image_id.items())])
    print(f'{n_kept / len(df):.1%} predictions kept')
    pseudolabeled.to_csv(args.output, index=None)


if __name__ == '__main__':
    main()
