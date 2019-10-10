import argparse
from collections import defaultdict
import re

import lightgbm as lgb
import pandas as pd

from ..data_utils import SEG_FP, get_encoded_classes
from .blend import (
    score_predictions_by_image_id, submission_from_predictions_by_image_id)


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('detailed', help='detailed dataframe for test part')
    arg('features', nargs='+', help='features from different folds')
    arg('--num-boost-round', type=int, default=100)
    arg('--save-model')
    arg('--load-model')
    arg('--output')
    args = parser.parse_args()

    print(args.features)
    feature_dfs = [pd.read_csv(path) for path in args.features]
    valid_df = feature_dfs[0]
    assert valid_df.columns[0] == 'item'
    assert valid_df.columns[-1] == 'y'
    feature_cols = valid_df.columns[1:-1]

    def build_features(df):
        df = df[feature_cols].copy()
        for col in feature_cols:
           if re.match('top_\d+_cls$', col):
               df[f'{col}_is_candidate'] = df[col] == df['candidate_cls']
        return df

    valid_features = build_features(valid_df)
    if args.load_model:
        bst = lgb.Booster(model_file=args.load_model)
    else:
        train_df = pd.concat(feature_dfs[1:])
        # TODO check categorical features
        train_data = lgb.Dataset(build_features(train_df), train_df['y'])
        valid_data = lgb.Dataset(valid_features, valid_df['y'],
                                 reference=train_data)
        params = {
            'objective': 'binary',
            'metric': ['auc', 'binary_logloss'],
        }
        bst = lgb.train(
            params=params,
            train_set=train_data,
            num_boost_round=args.num_boost_round,
            valid_sets=[valid_data])
    if args.save_model:
        bst.save_model(args.save_model)

    print('prediction')
    valid_df['y_pred'] = bst.predict(
        valid_features, num_iteration=bst.best_iteration)
    max_by_item = valid_df.iloc[valid_df.groupby('item')['y_pred'].idxmax()]
    max_by_item = max_by_item.reset_index(drop=True)

    print('scoring')
    classes = get_encoded_classes()
    cls_by_idx = {idx: cls for cls, idx in classes.items()}
    cls_by_idx[-1] = SEG_FP
    detailed = pd.read_csv(args.detailed)
    detailed['pred'] = \
        max_by_item['candidate_cls'].apply(cls_by_idx.__getitem__)
    print(f'SEG_FP ratio: {(detailed["pred"] == SEG_FP).mean():.5f}')

    predictions_by_image_id = defaultdict(list)
    for item in detailed.itertuples():
        if item.pred != SEG_FP:
            predictions_by_image_id[item.image_id].append({
                'cls': item.pred,
                'center': (item.x + item.w / 2, item.y + item.h / 2),
            })

    if args.output:
        submission = submission_from_predictions_by_image_id(
            predictions_by_image_id)
        submission.to_csv(args.output, index=False)
    else:
        print(f'accuracy: {(detailed["pred"] == detailed["true"]).mean():.5f}')
        score_predictions_by_image_id(predictions_by_image_id)


if __name__ == '__main__':
    main()
