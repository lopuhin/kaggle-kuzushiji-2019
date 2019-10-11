import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import tqdm

from ..data_utils import (
    get_encoded_classes, SEG_FP, submission_item, DATA_ROOT, load_train_df,
    get_target_boxes_labels, from_coco)
from ..utils import print_metrics
from ..metric import score_boxes, get_metrics


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('detailed', nargs='+', help='paths or path=weight items')
    arg('--output')
    arg('--score', action='store_true')
    args = parser.parse_args()
    if not args.score and not args.output:
        parser.error('either --score or --output is required')
    if args.output and Path(args.output).exists():
        parser.error(f'output {args.output} exists')
    classes = get_encoded_classes()
    cls_by_idx = {idx: cls for cls, idx in classes.items()}
    dfs = []
    weights = []
    for detailed in args.detailed:
        if '=' in detailed:
            path, weight = detailed.split('=')
        else:
            path, weight = detailed, 1
        dfs.append(pd.read_csv(path))
        weights.append(float(weight))
    predictions_by_image_id = defaultdict(list)
    for items in tqdm.tqdm(zip(*[df.itertuples() for df in dfs]),
                           total=len(dfs[0])):
        assert len(items) == len(weights)
        preds = [get_pred_dict(item, cls_by_idx, w)
                 for w, item in zip(weights, items)]
        classes = {cls for pred in preds for cls in pred}
        blend_cls = max(classes, key=lambda cls: sum(
            ps.get(cls, 0) for ps in preds))  # 0 default is rather arbitrary
        if blend_cls != SEG_FP:
            item = items[0]
            predictions_by_image_id[item.image_id].append({
                'cls': blend_cls,
                'center': (item.x + item.w / 2, item.y + item.h / 2),
            })
    if args.score:
        print_metrics(score_predictions_by_image_id(predictions_by_image_id))
        return

    submission = submission_from_predictions_by_image_id(
        predictions_by_image_id)
    submission.to_csv(args.output, index=False)


def score_predictions_by_image_id(predictions_by_image_id):
    gt_by_image_id = {item.image_id: item
                      for item in load_train_df().itertuples()}
    scores = []
    for image_id, predictions in predictions_by_image_id.items():
        item = gt_by_image_id[image_id]
        target_boxes, target_labels = get_target_boxes_labels(item)
        target_boxes = torch.from_numpy(target_boxes)
        pred_centers = [p['center'] for p in predictions]
        pred_labels = [p['cls'] for p in predictions]
        scores.append(score_boxes(
            truth_boxes=from_coco(target_boxes).numpy(),
            truth_label=target_labels,
            preds_center=np.array(pred_centers),
            preds_label=np.array(pred_labels),
        ))
    return get_metrics(scores)


def submission_from_predictions_by_image_id(predictions_by_image_id):
    submission = [submission_item(image_id, prediction)
                  for image_id, prediction in predictions_by_image_id.items()
                  if prediction]
    sample_submission = pd.read_csv(DATA_ROOT / 'sample_submission.csv')
    empty_pages = (set(sample_submission['image_id']) -
                   set(x['image_id'] for x in submission))
    submission.extend(submission_item(image_id, [])
                      for image_id in empty_pages)
    return pd.DataFrame(submission)


def get_pred_dict(item, cls_by_idx, weight: float = 1):
    return dict(zip(
        [cls_by_idx[int(idx)] for idx in item.top_k_classes.split()],
        [weight * float(v) for v in item.top_k_logits.split()]))


if __name__ == '__main__':
    main()
