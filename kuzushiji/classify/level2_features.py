import argparse
from collections import defaultdict
import json
from pathlib import Path

import pandas as pd
import numpy as np
import tqdm

from ..data_utils import SEG_FP, get_encoded_classes
from .blend import get_pred_dict


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('detailed',
        help='json with groups of predictions on test in "detailed_*" format, '
             'each group is blended (see source and level2.sh)')
    arg('output', help='output path for kuzushiji.classify.level2')
    arg('--top-k', type=int, default=3)
    args = parser.parse_args()
    if Path(args.output).exists():
        parser.error(f'output {args.output} exists')

    dfs = [[pd.read_csv(path) for path in group]
           for group in json.loads(args.detailed)]
    classes = dict(get_encoded_classes())
    cls_by_idx = {idx: cls for cls, idx in classes.items()}
    classes[SEG_FP] = -1  # should create better splits

    df = dfs[0][0]
    boxes_by_image_id = get_boxes_by_image_id(df)
    blended_groups = []
    for i, group in enumerate(dfs):
        blended = []
        blended_groups.append(blended)
        for items in tqdm.tqdm(zip(*[df.itertuples() for df in group]),
                               total=len(df), desc=f'blending {i}'):
            preds = [get_pred_dict(item, cls_by_idx) for item in items]
            blended.append((
                items[0],
                {cls: sum(p.get(cls, 0) for p in preds) / len(preds)
                     for cls in {cls for p in preds for cls in p}}
            ))

    output = []
    for i, item_groups in tqdm.tqdm(enumerate(zip(*blended_groups)),
                                    total=len(df)):
        features = {'item': i}
        top_k_classes = set()
        for j, (_, preds) in enumerate(item_groups):
            top_k = sorted(
                preds.items(), key=lambda cs: cs[1], reverse=True)[:args.top_k]
            top_k_classes.update(cls for cls, _ in top_k)
            features.update({f'top_{i}_cls_m{j}': classes[cls]
                             for i, (cls, _) in enumerate(top_k)})
            features.update({f'top_{i}_score_m{j}': score
                             for i, (_, score) in enumerate(top_k)})
        item = item_groups[0][0]
        features['box_overlap'] = get_max_iou(
            item, boxes_by_image_id[item.image_id])
        true = item.true
        if not any(true == cls for cls in top_k_classes):
            true = SEG_FP  # it harms F1 less: one fn instead of fn + fp
        for cls in (top_k_classes | {true, SEG_FP}):
            output.append(dict(
                features,
                candidate_cls=classes[cls],
                y=true == cls))
    print(f'{len(output):,} items')
    pd.DataFrame(output).to_csv(args.output, index=None)


def get_boxes_by_image_id(df):
    boxes_by_image_id = defaultdict(list)
    for item in df.itertuples():
        boxes_by_image_id[item.image_id].append(
            [item.x, item.y, item.x + item.w, item.y + item.h, item.Index])
    return {image_id: np.array(boxes)
            for image_id, boxes in boxes_by_image_id.items()}


def get_max_iou(item, boxes):
    boxes = boxes[boxes[:, 4] != item.Index]  # exclude self
    if boxes.shape[0] == 0:
        return 0
    x1, y1, x2, y2 = (boxes[:, i] for i in range(4))
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    xx1 = np.maximum(item.x, x1)
    yy1 = np.maximum(item.y, y1)
    xx2 = np.minimum(item.x + item.w, x2)
    yy2 = np.minimum(item.y + item.h, y2)
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)
    overlap = (w * h) / area
    return overlap.max()


if __name__ == '__main__':
    main()
