#!/bin/bash

set -ev

python -m kuzushiji.classify.level2_features _runs/clf-fold0/detailed.csv.gz _runs/v100-clf-fold0-wsl16d-mt256-frozen-start/detailed.csv.gz                                                              _runs/level2_train_v3_fold0.csv.gz
python -m kuzushiji.classify.level2_features _runs/clf-fold1/detailed.csv.gz _runs/clf-fold1-wsl8d-mt256-frozen/detailed.csv.gz             _runs/clf-fold1-wsl8d-mt256-frozen-pseudo-ft/detailed.csv.gz _runs/level2_train_v3_fold1.csv.gz
python -m kuzushiji.classify.level2_features _runs/clf-fold2/detailed.csv.gz _runs/clf-fold2-wsl8d-mt256-frozen/detailed.csv.gz             _runs/clf-fold2-wsl8d-mt256-frozen-pseudo-ft/detailed.csv.gz _runs/level2_train_v3_fold2.csv.gz
python -m kuzushiji.classify.level2_features _runs/clf-fold3/detailed.csv.gz _runs/clf-fold3-wsl8d-mt256-frozen/detailed.csv.gz             _runs/clf-fold3-wsl8d-mt256-frozen-pseudo-ft/detailed.csv.gz _runs/level2_train_v3_fold3.csv.gz
python -m kuzushiji.classify.level2_features _runs/clf-fold4/detailed.csv.gz _runs/clf-fold4-wsl8d-mt256-frozen/detailed.csv.gz             _runs/clf-fold4-wsl8d-mt256-frozen-pseudo-ft/detailed.csv.gz _runs/level2_train_v3_fold4.csv.gz

python -m kuzushiji.classify.level2_features \
    _runs/v100-clf-fold0-wsl16d-mt256-frozen-start/test_detailed.csv.gz \
    _runs/clf-fold1-wsl8d-mt256-frozen/test_detailed.csv.gz \
    _runs/clf-fold2-wsl8d-mt256-frozen/test_detailed.csv.gz \
    _runs/clf-fold3-wsl8d-mt256-frozen/test_detailed.csv.gz \
    _runs/clf-fold4-wsl8d-mt256-frozen/test_detailed.csv.gz \
    _runs/clf-fold0/test_detailed.csv.gz \
    _runs/clf-fold1/test_detailed.csv.gz \
    _runs/clf-fold2/test_detailed.csv.gz \
    _runs/clf-fold3/test_detailed.csv.gz \
    _runs/clf-fold4/test_detailed.csv.gz \
    _runs/clf-fold1-wsl8d-mt256-frozen-pseudo-ft/test_detailed.csv.gz \
    _runs/clf-fold2-wsl8d-mt256-frozen-pseudo-ft/test_detailed.csv.gz \
    _runs/clf-fold3-wsl8d-mt256-frozen-pseudo-ft/test_detailed.csv.gz \
    _runs/clf-fold4-wsl8d-mt256-frozen-pseudo-ft/test_detailed.csv.gz \
    level2_test_v2_fold01234-wsl16-fold0-wsl8-fold1234-pseudo-ft-fold1234.csv.gz

python -m kuzushiji.classify.level2 _runs/clf-fold[0-4]/detailed.csv.gz _runs/level2_train_v3_fold[0-4].csv.gz --num-boost-round 50 --save-model bst-nbr50

python -m kuzushiji.classify.level2 _runs/clf-fold0/test_detailed.csv.gz  \
    level2_test_v2_fold01234-wsl16-fold0-wsl8-fold1234-pseudo-ft-fold1234.csv.gz \
    --load-model bst-nbr50 \
    --output level2_test_v2_fold01234-wsl16-fold0-wsl8-fold1234_nbr50_fold0.csv.gz --num-folds 1

python -m kuzushiji.classify.level2 _runs/clf-fold0/test_detailed.csv.gz  \
    level2_test_v2_fold01234-wsl16-fold0-wsl8-fold1234-pseudo-ft-fold1234.csv.gz \
    --load-model bst-nbr50 \
    --output level2_test_v2_fold01234-wsl16-fold0-wsl8-fold1234_nbr50_blend.csv.gz
