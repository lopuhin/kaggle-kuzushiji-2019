import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd
import tqdm

from ..data_utils import SEG_FP, get_encoded_classes
from .blend import get_pred_dict


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('detailed', nargs='+',
        help='predictions on test in "detailed_*" format')
    arg('output', help='path where predictions are written as ground truth')
    arg('--min-gap', type=float, default=5,
        help='min score gap between top and second prediction')
    arg('--max-second-score', type=float, default=10,
        help='max second prediction score')
    arg('--drop-seg-fp', type=int, default=0)
    args = parser.parse_args()
    if Path(args.output).exists():
        parser.error(f'output {args.output} exists')

    dfs = [pd.read_csv(path) for path in args.detailed]
    classes = get_encoded_classes()
    cls_by_idx = {idx: cls for cls, idx in classes.items()}
    by_image_id = defaultdict(list)
    n_kept = 0
    for items in tqdm.tqdm(zip(*[df.itertuples() for df in dfs]),
                           total=len(dfs[0])):
        item = items[0]
        preds = [get_pred_dict(item, cls_by_idx) for item in items]
        blend = {cls: sum(p.get(cls, 0) for p in preds) / len(preds)
                 for cls in {cls for p in preds for cls in p}}
        pred = max(blend, key=lambda cls: blend[cls])
        top, second, *_ = sorted(blend.values(), reverse=True)
        if (top - second >= args.min_gap and
                second <= args.max_second_score and
                (not args.drop_seg_fp or item.pred != SEG_FP)):
            n_kept += 1
            by_image_id[item.image_id].append(
                ' '.join(map(str, [
                    pred,
                    int(item.x),
                    int(item.y),
                    int(item.w),
                    int(item.h),
                ])))

    pseudolabeled = pd.DataFrame(
        [{'image_id': image_id, 'labels': ' '.join(labels)}
         for image_id, labels in sorted(by_image_id.items())])
    print(f'{n_kept / len(dfs[0]):.1%} predictions kept')
    pseudolabeled.to_csv(args.output, index=None)


if __name__ == '__main__':
    main()
