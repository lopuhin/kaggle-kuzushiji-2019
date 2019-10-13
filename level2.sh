#!/bin/bash

set -ev

python -m kuzushiji.classify.level2_features  _runs/clf-fold0/detailed.csv.gz _runs/clf-fold0-wsl8d-mt256-frozen/detailed.csv.gz _runs/clf-fold0-wsl8d-mt256-frozen-pseudo-ft/detailed.csv.gz _runs/level2_train_v7_fold0.csv.gz
python -m kuzushiji.classify.level2_features  _runs/clf-fold1/detailed.csv.gz _runs/clf-fold1-wsl8d-mt256-frozen/detailed.csv.gz _runs/clf-fold1-wsl8d-mt256-frozen-pseudo-ft/detailed.csv.gz _runs/level2_train_v7_fold1.csv.gz
python -m kuzushiji.classify.level2_features  _runs/clf-fold2/detailed.csv.gz _runs/clf-fold2-wsl8d-mt256-frozen/detailed.csv.gz _runs/clf-fold2-wsl8d-mt256-frozen-pseudo-ft/detailed.csv.gz _runs/level2_train_v7_fold2.csv.gz
python -m kuzushiji.classify.level2_features  _runs/clf-fold3/detailed.csv.gz _runs/clf-fold3-wsl8d-mt256-frozen/detailed.csv.gz _runs/clf-fold3-wsl8d-mt256-frozen-pseudo-ft/detailed.csv.gz _runs/level2_train_v7_fold3.csv.gz
python -m kuzushiji.classify.level2_features  _runs/clf-fold4/detailed.csv.gz _runs/clf-fold4-wsl8d-mt256-frozen/detailed.csv.gz _runs/clf-fold4-wsl8d-mt256-frozen-pseudo-ft/detailed.csv.gz _runs/level2_train_v7_fold4.csv.gz

python -m kuzushiji.classify.level2 _runs/clf-fold[0-4]/detailed.csv.gz _runs/level2_train_v7_fold[0-4].csv.gz --num-boost-round 200 --save-model bst-v7

python -m kuzushiji.classify.level2_features \
    _runs/clf-fold0-wsl8d-mt256-frozen/test_detailed.csv.gz \
    _runs/clf-fold1-wsl8d-mt256-frozen/test_detailed.csv.gz \
    _runs/clf-fold2-wsl8d-mt256-frozen/test_detailed.csv.gz \
    _runs/clf-fold3-wsl8d-mt256-frozen/test_detailed.csv.gz \
    _runs/clf-fold4-wsl8d-mt256-frozen/test_detailed.csv.gz \
    _runs/clf-fold0/test_detailed.csv.gz \
    _runs/clf-fold1/test_detailed.csv.gz \
    _runs/clf-fold2/test_detailed.csv.gz \
    _runs/clf-fold3/test_detailed.csv.gz \
    _runs/clf-fold4/test_detailed.csv.gz \
    _runs/clf-fold0-wsl8d-mt256-frozen-pseudo-ft/test_detailed.csv.gz \
    _runs/clf-fold1-wsl8d-mt256-frozen-pseudo-ft/test_detailed.csv.gz \
    _runs/clf-fold2-wsl8d-mt256-frozen-pseudo-ft/test_detailed.csv.gz \
    _runs/clf-fold3-wsl8d-mt256-frozen-pseudo-ft/test_detailed.csv.gz \
    _runs/clf-fold4-wsl8d-mt256-frozen-pseudo-ft/test_detailed.csv.gz \
    level2_test_v7_fold01234_wsl8-fold01234-pseudo-ft-fold01234.csv.gz

python -m kuzushiji.classify.level2 _runs/clf-fold0/test_detailed.csv.gz  \
    level2_test_v7_fold01234_wsl8-fold01234-pseudo-ft-fold01234.csv.gz \
    --load-model bst-v7 \
    --output sub_level2_test_v7_fold01234_wsl8-fold01234-pseudo-ft-fold01234_blend.csv.gz