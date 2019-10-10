import argparse

import lightgbm as lgb
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('features', nargs='+', help='features from different folds')
    arg('--num-boost-round', type=int, defaul=100)
    args = parser.parse_args()

    print(args.features)
    feature_dfs = [pd.read_csv(path) for path in args.features]
    valid_df = feature_dfs[0]
    train_df = pd.concat(feature_dfs[1:])

    assert train_df.columns[-1] == 'y'
    feature_cols = train_df.columns[:-1]
    train_data = lgb.Dataset(train_df[feature_cols], train_df['y'])
    valid_data = lgb.Dataset(valid_df[feature_cols], valid_df['y'],
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
    y_pred = bst.predict(
        valid_df[feature_cols], num_iteration=bst.best_iteration)

    import IPython; IPython.embed()



if __name__ == '__main__':
    main()
