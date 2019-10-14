set -ev

for fold in 0 1 2 3 4;
do
    echo "fold ${fold}"
    # Train segmentation model
    python -m kuzushiji.segment.main \
       --output-dir _runs/segment-fold${fold} --fold ${fold} \
       --model fasterrcnn_resnet152_fpn
    # Out-of-fold predictions
    python -m kuzushiji.segment.main \
        --output-dir _runs/segment-fold${fold} --fold ${fold} \
        --model fasterrcnn_resnet152_fpn \
        --resume _runs/segment-fold${fold}/model_best.pth \
        --test-only
    # Test predictions
    python -m kuzushiji.segment.main \
        --output-dir _runs/segment-fold${fold} --fold ${fold} \
        --model fasterrcnn_resnet152_fpn \
        --resume _runs/segment-fold${fold}/model_best.pth \
        --submission
done

# Concatenate blend results
python -c "
import pandas as pd;
pd.concat([pd.read_csv(f'_runs/segment-fold{i}/clf_gt.csv') for i in range(5)]).to_csv('_runs/segment_clf_gt.csv')
"

# Train classification models from scratch

for fold in 0 1 2 3 4;
do
    echo "fold ${fold}"

    # resnet152, requires a 16 GB GPU, most likely will fit  in 11 GB with --opt-level O1 (weak but still used)
    python -m kuzushiji.classify.main \
        _runs/segment_clf_gt.csv \
        --output-dir _runs/clf-fold${fold} --fold ${fold} \
        --base resnet152
    # Check validation score
    python -m kuzushiji.classify.main \
        _runs/segment_clf_gt.csv \
        --output-dir _runs/clf-fold${fold} --fold ${fold} \
        --base resnet152 \
        --resume _runs/clf-fold${fold}/model_best.pth \
        --print-model 0 \
        --n-tta 4 \
        --test-only > _runs/clf-fold${fold}/validation.txt
    # Create submission
    python -m kuzushiji.classify.main \
        _runs/segment-fold0/test_predictions.csv \
        --output-dir _runs/clf-fold${fold} --fold ${fold} \
        --base resnet152 \
        --resume _runs/clf-fold${fold}/model_best.pth \
        --print-model 0 \
        --n-tta 4 \
        --submission

    # resnext101_32x8d_wsl: smallest of WSL models, still fits fine on 2080ti with fp16 and freezing
    # Takes around 16h to train on 2080ti
    python -m kuzushiji.classify.main \
        _runs/segment_clf_gt.csv \
        --output-dir _runs/clf-fold${fold}-wsl8d-mt256-frozen --fold ${fold} --print-model 0 \
        --max-targets 256 --benchmark 1 --frozen-start 1 \
        --base resnext101_32x8d_wsl --workers 4 --batch-size 12 --lr 16e-3 --opt-level O1
    python -m kuzushiji.classify.main \
        _runs/segment_clf_gt.csv \
        --output-dir _runs/clf-fold${fold}-wsl8d-mt256-frozen --fold ${fold} --print-model 0 \
        --benchmark 1 \
        --base resnext101_32x8d_wsl --workers 4 --batch-size 12 --opt-level O1 \
        --resume _runs/clf-fold${fold}-wsl8d-mt256-frozen/model_best.pth \
        --n-tta 4 --test-only > _runs/clf-fold${fold}-wsl8d-mt256-frozen/validation.txt
    python -m kuzushiji.classify.main \
        _runs/segment-fold0/test_predictions.csv \
        --output-dir _runs/clf-fold${fold}-wsl8d-mt256-frozen --fold ${fold} --print-model 0 \
        --benchmark 1 \
        --base resnext101_32x8d_wsl --workers 4 --batch-size 12 --opt-level O1 \
        --resume _runs/clf-fold${fold}-wsl8d-mt256-frozen/model_best.pth \
        --n-tta 4 --submission

done

# Create pseduolabels
python -m kuzushiji.classify.pseudolabel \
    _runs/clf-fold[0-4]-wsl8d-mt256-frozen/test_detailed.csv.gz \
    _runs/clf-fold[0-4]/test_detailed.csv.gz \
    _runs/pseudolabels.csv.gz

# Train models on pseudolabels

for fold in 0 1 2 3 4;
do
    echo "fold ${fold}"

    # Fine-tuned pseudolabels model (~2h on 2080ti)
    # Note that exact params for this model were not saved, and recovered from memory (logs were saved though)
    python -m kuzushiji.classify.main \
        _runs/segment_clf_gt.csv \
        --output-dir _runs/clf-fold${fold}-wsl8d-mt256-frozen-pseudo-ft --fold ${fold} --print-model 0 \
        --resume _runs/clf-fold${fold}-wsl8d-mt256-frozen/model_best.pth \
        --max-targets 256 --benchmark 1 --frozen-start 1 \
        --base resnext101_32x8d_wsl --workers 4 --batch-size 12 --lr 1.6e-3 --opt-level O1 \
        --pseudolabels _runs/pseudolabels.csv.gz \
        --repeat-train 3 \
        --epochs 5
    python -m kuzushiji.classify.main \
        _runs/segment_clf_gt.csv \
        --output-dir _runs/clf-fold${fold}-wsl8d-mt256-frozen-pseudo-ft --fold ${fold} --print-model 0 \
        --benchmark 1 \
        --base resnext101_32x8d_wsl --workers 4 --batch-size 12 --opt-level O1 \
        --resume _runs/clf-fold${fold}-wsl8d-mt256-frozen-pseudo-ft/model_best.pth \
        --n-tta 4 --test-only
    python -m kuzushiji.classify.main \
        _runs/segment-fold0/test_predictions.csv \
        --output-dir _runs/clf-fold${fold}-wsl8d-mt256-frozen-pseudo --fold ${fold} --print-model 0 \
        --benchmark 1 \
        --base resnext101_32x8d_wsl --workers 4 --batch-size 12 --opt-level O1 \
        --resume _runs/clf-fold${fold}-wsl8d-mt256-frozen-pseudo-ft/model_best.pth \
        --n-tta 4 --submission

    # Model on pseudolabels from scratch (abour 22h on 2080ti)
    python -m kuzushiji.classify.main \
        _runs/segment_clf_gt.csv \
        --output-dir _runs/clf-fold${fold}-wsl8d-mt256-frozen-pseudo --fold ${fold} --print-model 0 \
        --max-targets 256 --benchmark 1 --frozen-start 1 \
        --base resnext101_32x8d_wsl --workers 4 --batch-size 12 --lr 16e-3 --opt-level O1 \
        --pseudolabels _runs/pseudolabels.csv.gz \
        --repeat-train 4
    python -m kuzushiji.classify.main \
        _runs/segment_clf_gt.csv \
        --output-dir _runs/clf-fold${fold}-wsl8d-mt256-frozen-pseudo --fold ${fold} --print-model 0 \
        --benchmark 1 \
        --base resnext101_32x8d_wsl --workers 4 --batch-size 12 --opt-level O1 \
        --resume _runs/clf-fold${fold}-wsl8d-mt256-frozen-pseudo/model_best.pth \
        --n-tta 4 --test-only
    python -m kuzushiji.classify.main \
        _runs/segment-fold0/test_predictions.csv \
        --output-dir _runs/clf-fold${fold}-wsl8d-mt256-frozen-pseudo --fold ${fold} --print-model 0 \
        --benchmark 1 \
        --base resnext101_32x8d_wsl --workers 4 --batch-size 12 --opt-level O1 \
        --resume _runs/clf-fold${fold}-wsl8d-mt256-frozen-pseudo/model_best.pth \
        --n-tta 4 --submission

done

# create final submission with a second level model

bash ./level2.sh
