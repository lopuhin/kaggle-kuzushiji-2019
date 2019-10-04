import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import tqdm

from ..data_utils import load_train_df, SEG_FP, get_encoded_classes
from ..utils import print_metrics


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('clf_folder')
    arg('--device', default='cuda')
    arg('--limit', type=int, help='evaluate only on some pages')
    arg('--fp16', type=int, default=1)
    arg('--check-thresholds', type=int)
    arg('--threshold', type=float, default=0.0)
    arg('--zero-seg-fp', type=int, default=0)
    arg('--submission', action='store_true')
    arg('--show-errors', action='store_true')
    args = parser.parse_args()

    clf_folder = Path(args.clf_folder)
    device = torch.device(args.device)
    train_features, train_ys = torch.load(
        clf_folder / 'train_features.pth', map_location='cpu')
    if args.submission:
        test_features, _ = torch.load(
            clf_folder / 'test_features.pth', map_location='cpu')
    else:
        test_features, test_ys = torch.load(
            clf_folder / 'valid_features.pth', map_location='cpu')
    train_features = train_features.to(device)
    if args.submission:
        df_detailed = pd.read_csv(clf_folder / 'test_detailed.csv.gz')
    else:
        df_detailed = pd.read_csv(clf_folder / 'detailed.csv.gz')
    image_ids = sorted(set(df_detailed['image_id'].values))
    if args.limit:
        rng = np.random.RandomState(42)
        image_ids = rng.choice(image_ids, args.limit)
        index = torch.tensor(df_detailed['image_id'].isin(image_ids).values)
        test_features = test_features[index]
        if not args.submission:
            test_ys = test_ys[index]
    df_detailed = df_detailed[df_detailed['image_id'].isin(image_ids)]

    eps = 1e-9
    train_features /= torch.norm(train_features, dim=1).unsqueeze(1) + eps
    test_features /= torch.norm(test_features, dim=1).unsqueeze(1) + eps

    if args.fp16:
        train_features = train_features.half()
        test_features = test_features.half()

    classes = get_encoded_classes()
    seg_fp_id = classes[SEG_FP]
    thresholds = {args.threshold}
    if args.check_thresholds:
        thresholds.update(np.linspace(0.5, 0.9, args.check_thresholds))
    thresholds = sorted(thresholds)

    pred_ys_by_threshold = defaultdict(list)
    train_seg_fp_mask = train_ys == seg_fp_id
    for i in tqdm.trange(test_features.shape[0]):
        feature = test_features[i].unsqueeze(1).to(device)
        sim = torch.mm(train_features, feature).squeeze()
        if args.zero_seg_fp:
            sim[train_seg_fp_mask] = 0
        max_idx = sim.argmax()
        max_sim = sim[max_idx]
        cls = train_ys[max_idx]
        for th in thresholds:
            th_cls = cls
            # TODO maybe also separate threshold for seg_fp?
            if max_sim < th:
                th_cls = seg_fp_id
            pred_ys_by_threshold[th].append(th_cls)
        topk_sim, topk_idx = torch.topk(sim, k=10)
        clf_cls = classes[df_detailed.iloc[i].pred]
        true_cls = test_ys[i]
        if args.show_errors and not (true_cls == cls == clf_cls):
            true_mask = train_ys == true_cls
            if true_mask.any():
                true_sim = sim[true_mask].max()
            else:
                true_sim = 0
            print(
                f'true {true_cls:>4}',
                f'clf {clf_cls:>4}',
                f'knn {cls:>4}',
                f'true_sim={true_sim:.3f}',
                ' '.join(f'{cls_id:>4}={s:.3f}'
                    for cls_id, s in zip(train_ys[topk_idx], topk_sim)),
                )

    pred_ys_by_threshold = {
        th: np.array(pred_ys) for th, pred_ys in pred_ys_by_threshold.items()}

    if args.submission:
        pred_ys = pred_ys_by_threshold[args.threshold]
        cls_by_id = {id: cls for cls, id in classes.items()}
        df_detailed['pred'] = [cls_by_id[id] for id in pred_ys]
        # TODO update top_k_classes and top_k_logits for the blend
        df_detailed.to_csv(clf_folder / 'test_detailed_features.csv.gz',
                           index=None)
        return

    df_train = load_train_df()
    df_train = df_train[df_train['image_id'].isin(image_ids)]
    # fn from missing detections missed by the segmentation model
    fn_segmentation = (
        sum(len(label.split()) // 5 for label in df_train['labels'].values) -
        len(df_detailed))
    clf_metrics = get_metrics(
        true=df_detailed['true'].values,
        pred=df_detailed['pred'].values,
        seg_fp=SEG_FP,
        fn_segmentation=fn_segmentation)
    print('clf baseline')
    print_metrics(clf_metrics)

    true_ids = np.array([classes[cls] for cls in df_detailed['true'].values])
    for th, pred_ys in sorted(pred_ys_by_threshold.items()):
        knn_metrics = get_metrics(
            true=true_ids,
            pred=pred_ys,
            seg_fp=seg_fp_id,
            fn_segmentation=fn_segmentation)
        print(f'knn at threshold={th:.3f}')
        print_metrics(knn_metrics)


def get_metrics(true, pred, seg_fp, fn_segmentation):
    accuracy = (true == pred).mean()
    tp = ((true != seg_fp) & (pred == true)).sum()
    fp = ((pred != seg_fp) & (pred != true)).sum()
    fn = ((true != seg_fp) & (pred != true)).sum() + fn_segmentation
    seg_fp_ratio = (pred == seg_fp).mean()
    if (tp + fp) == 0 or (tp + fn) == 0:
        f1 = 0
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        if precision > 0 and recall > 0:
            f1 = (2 * precision * recall) / (precision + recall)
        else:
            f1 = 0
    return {
        'f1': f1,
        'accuracy': accuracy,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'seg_fp_ratio': seg_fp_ratio,
    }


if __name__ == '__main__':
    main()
