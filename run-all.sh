set -ev

# Segmentation

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
done

# Concatenate out-of-fold predictions
python -c "
import pandas as pd
pd.concat([pd.read_csv(f'_runs/segment-fold{i}/clf_gt.csv')
           for i in range(5)]).to_csv('_runs/segment_clf_gt.csv')
"

# Blend test predictions
python -m kuzushiji.segment.main \
   --output-dir _runs/segment-blend \
   --model fasterrcnn_resnet152_fpn \
   --submission \
   --resume _runs/segment-fold0/model_best.pth \
   --blend _runs/segment-fold1/model_best.pth \
   --blend _runs/segment-fold2/model_best.pth \
   --blend _runs/segment-fold3/model_best.pth \
   --blend _runs/segment-fold4/model_best.pth

# Classification

for fold in 0 1 2 3 4;
do
    echo "fold ${fold}"
    # Train classification model
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
    # Create submission (sementation: check different folds, blend)
    python -m kuzushiji.classify.main \
        _runs/segment-blend/test_predictions.csv \
        --output-dir _runs/clf-fold${fold} --fold ${fold} \
        --base resnet152 \
        --resume _runs/clf-fold${fold}/model_best.pth \
        --print-model 0 \
        --n-tta 4 \
        --submission
done

# TODO blend
