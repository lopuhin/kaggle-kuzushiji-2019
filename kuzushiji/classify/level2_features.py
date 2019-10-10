import argparse
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
    arg('output', help='output path for kuzushiji.classify.level2')
    arg('--top-k', type=int, default=5)
    args = parser.parse_args()
    if Path(args.output).exists():
        parser.error(f'output {args.output} exists')

    dfs = [pd.read_csv(path) for path in args.detailed]
    classes = dict(get_encoded_classes())
    cls_by_idx = {idx: cls for cls, idx in classes.items()}
    classes[SEG_FP] = -1  # should create better splits
    output = []
    for i, items in tqdm.tqdm(enumerate(zip(*[df.itertuples() for df in dfs])),
                              total=len(dfs[0])):
        item = items[0]
        preds = [get_pred_dict(item, cls_by_idx) for item in items]
        blend = {cls: sum(p.get(cls, 0) for p in preds) / len(preds)
                 for cls in {cls for p in preds for cls in p}}
        true = item.true
        top_k = sorted(
            blend.items(), key=lambda cs: cs[1], reverse=True)[:args.top_k]
        features = {'item': i}
        features.update({
            f'top_{i}_cls': classes[cls] for i, (cls, _) in enumerate(top_k)})
        features.update({
            f'top_{i}_score': score for i, (_, score) in enumerate(top_k)})
        if not any(true == cls for cls, _ in top_k):
            true = SEG_FP  # it harms F1 less: one fn instead of fn + fp
        for cls in ({cls for cls, _ in top_k} | {true, SEG_FP}):
            output.append(dict(
                features,
                candidate_cls=classes[cls], 
                y=true == cls))
    print(f'{len(output):,} items')
    pd.DataFrame(output).to_csv(args.output, index=None)


if __name__ == '__main__':
    main()
