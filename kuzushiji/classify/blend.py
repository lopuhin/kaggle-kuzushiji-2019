import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd
import tqdm

from ..data_utils import get_encoded_classes, SEG_FP, submission_item, DATA_ROOT


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('detailed', nargs='+')
    arg('output')
    args = parser.parse_args()
    if Path(args.output).exists():
        parser.error(f'output {args.output} exists')
    classes = get_encoded_classes()
    cls_by_idx = {idx: cls for cls, idx in classes.items()}
    dfs = [pd.read_csv(p) for p in args.detailed]
    predictions_by_image_id = defaultdict(list)
    for items in tqdm.tqdm(zip(*[df.itertuples() for df in dfs]),
                           total=len(dfs[0])):
        preds = [get_pred_dict(item, cls_by_idx) for item in items]
        classes = {cls for pred in preds for cls in pred}
        blend_cls = max(classes, key=lambda cls: sum(
            pred.get(cls, 0) for pred in preds))
        if blend_cls != SEG_FP:
            item = items[0]
            predictions_by_image_id[item.image_id].append({
                'cls': blend_cls,
                'center': (item.x + item.w / 2, item.y + item.h / 2),
            })
    submission = [submission_item(image_id, prediction)
                  for image_id, prediction in predictions_by_image_id.items()
                  if prediction]
    sample_submission = pd.read_csv(DATA_ROOT / 'sample_submission.csv')
    empty_pages = (set(sample_submission['image_id']) -
                   set(x['image_id'] for x in submission))
    submission.extend(submission_item(image_id, [])
                      for image_id in empty_pages)
    pd.DataFrame(submission).to_csv(args.output, index=False)


def get_pred_dict(item, cls_by_idx):
    return dict(zip(
        [cls_by_idx[int(idx)] for idx in item.top_k_classes.split()],
        map(float, item.top_k_logits.split())))


if __name__ == '__main__':
    main()
