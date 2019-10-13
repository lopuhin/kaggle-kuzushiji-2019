import argparse
from collections import defaultdict
import re

import lightgbm as lgb
import pandas as pd
import numpy as np
import xgboost as xgb

from ..data_utils import SEG_FP, get_encoded_classes
from ..utils import print_metrics
from ..metric import get_metrics
from .blend import (
    score_predictions_by_image_id, submission_from_predictions_by_image_id)


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('detailed_then_features', nargs='+',
        help='detailed dataframes and the features in the same order')
    arg('--num-boost-round', type=int, default=200)
    arg('--lr', type=float, default=0.1, help='for lightgbm')
    arg('--eta', type=float, default=0.15, help='for xgboost')
    arg('--save-model')
    arg('--load-model')
    arg('--output')
    arg('--n-folds', type=int, default=5)
    arg('--seg-fp-adjust', type=float)
    args = parser.parse_args()
    if len(args.detailed_then_features) % 2 != 0:
        parser.error('number of detailed and features must be equal')
    n = len(args.detailed_then_features) // 2
    detailed_paths, feature_paths = (args.detailed_then_features[:n],
                                     args.detailed_then_features[n:])
    if args.output:
        if not args.load_model:
            parser.error('--output needs --load-model')
    elif len(feature_paths) == 1:
        parser.error('need more than one feature df for train/valid split')
    print('\n'.join(
        f'{f} | {d}' for f, d in zip(detailed_paths, feature_paths)))

    detailed_dfs = [pd.read_csv(path) for path in detailed_paths]
    feature_dfs = [pd.read_csv(path) for path in feature_paths]
    valid_df = feature_dfs[0]
    assert valid_df.columns[0] == 'item'
    assert valid_df.columns[-1] == 'y'
    feature_cols = [
        col for col in valid_df.columns[1:-1] if col not in {
            'width', 'height', 'aspect',
            'candidate_count', 'candidate_count_on_page',
            'candidate_freq_on_page',
        }]
    top_cls_re = re.compile('^top_\d+_cls$')

    def build_features(df):
        df = df[feature_cols].copy()
        for col in feature_cols:
            if top_cls_re.match(col):
                df[f'{col}_is_candidate'] = df[col] == df['candidate_cls']
                # del df[col]
        print(' '.join(df.columns))
        return df

    classes = get_encoded_classes()
    cls_by_idx = {idx: cls for cls, idx in classes.items()}
    cls_by_idx[-1] = SEG_FP

    y_preds = []
    all_metrics = []
    for fold_num in range(args.n_folds):
        print(f'fold {fold_num}')
        detailed = (detailed_dfs[fold_num if len(detailed_dfs) != 1 else 0]
                    .copy())
        valid_df = feature_dfs[fold_num if len(feature_dfs) != 1 else 0].copy()
        valid_features = build_features(valid_df)
        xgb_valid_data = xgb.DMatrix(valid_features, label=valid_df['y'])

        fold_path = lambda path: f'{path}.fold{fold_num}'
        if args.load_model:
            load_path = fold_path(args.load_model)
            print(f'loading from {load_path}')
            lgb_model = lgb.Booster(model_file=load_path)
        else:
            train_df = pd.concat([df for i, df in enumerate(feature_dfs)
                                  if i != fold_num])
            train_features = build_features(train_df)
            lgb_model = train_lgb(
                train_features, train_df['y'],
                valid_features, valid_df['y'],
                lr=args.lr,
                num_boost_round=args.num_boost_round)
            xgb_model = train_xgb(
                train_features, train_df['y'],
                valid_features, valid_df['y'],
                eta=args.eta,
                num_boost_round=args.num_boost_round)
        if args.save_model:
            save_path = fold_path(args.save_model)
            print(f'saving to {save_path}')
            lgb_model.save_model(
                save_path, num_iteration=lgb_model.best_iteration)

        print('prediction')
        lgb_y_pred = lgb_model.predict(
            valid_features, num_iteration=lgb_model.best_iteration)
        xgb_y_pred = xgb_model.predict(
            xgb_valid_data, ntree_limit=xgb_model.best_ntree_limit)
        valid_df['y_pred'] = np.mean([lgb_y_pred, xgb_y_pred], axis=0)
        if args.seg_fp_adjust:
            valid_df.loc[valid_df['candidate_cls'] == -1, 'y_pred'] += \
                args.seg_fp_adjust
        y_preds.append(valid_df['y_pred'].values)
        max_by_item = get_max_by_item(valid_df)

        print('scoring')
        detailed['pred'] = \
            max_by_item['candidate_cls'].apply(cls_by_idx.__getitem__)
        print(f'SEG_FP ratio: {(detailed["pred"] == SEG_FP).mean():.5f}')
        predictions_by_image_id = get_predictions_by_image_id(detailed)

        if not args.output:
            metrics = {
                'accuracy': (detailed["pred"] == detailed["true"]).mean(),
            }
            metrics.update(
                score_predictions_by_image_id(predictions_by_image_id))
            print_metrics(metrics)
            all_metrics.append(metrics)

    if args.output:
        valid_df['y_pred'] = np.mean(y_preds, axis=0)
        max_by_item = get_max_by_item(valid_df)
        detailed['pred'] = \
            max_by_item['candidate_cls'].apply(cls_by_idx.__getitem__)
        predictions_by_image_id = get_predictions_by_image_id(detailed)
        submission = submission_from_predictions_by_image_id(
            predictions_by_image_id)
        submission.to_csv(args.output, index=False)
    else:
        print('\nAll folds:')
        print_metrics(get_metrics(all_metrics))



def train_lgb(train_features, train_y, valid_features, valid_y, *,
              lr, num_boost_round):
    train_data = lgb.Dataset(train_features, train_y)
    valid_data = lgb.Dataset(valid_features, valid_y, reference=train_data)
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'learning_rate': lr,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'feature_fraction': 0.9,
        'min_data_in_leaf': 20,
        'num_leaves': 31,
        'scale_pos_weight': 1.2,
        'lambda_l2': 1,
    }
    print(params)
    return lgb.train(
        params=params,
        train_set=train_data,
        num_boost_round=num_boost_round,
        early_stopping_rounds=20,
        valid_sets=[valid_data],
        verbose_eval=10,
    )


def train_xgb(train_features, train_y, valid_features, valid_y, *,
              eta, num_boost_round):
    train_data = xgb.DMatrix(train_features, label=train_y)
    valid_data = xgb.DMatrix(valid_features, label=valid_y)
    params = {
        'eta': eta,
        'objective': 'binary:logistic',
    }
    print(params)
    eval_list = [(valid_data, 'eval')]
    return xgb.train(
        params, train_data, num_boost_round, eval_list,
        early_stopping_rounds=20,
        verbose_eval=10,
    )


def get_max_by_item(df):
    return (df.iloc[df.groupby('item')['y_pred'].idxmax()]
            .reset_index(drop=True))


def get_predictions_by_image_id(detailed):
    predictions_by_image_id = defaultdict(list)
    for item in detailed.itertuples():
        if item.pred != SEG_FP:
            predictions_by_image_id[item.image_id].append({
                'cls': item.pred,
                'center': (item.x + item.w / 2, item.y + item.h / 2),
            })
    return predictions_by_image_id


if __name__ == '__main__':
    main()
