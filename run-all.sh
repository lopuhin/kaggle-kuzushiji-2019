set -ev

for fold in 0 1 2 3 4;
do
    # Train segmentation model
    python -m kuzushiji.segment.main --output-dir _runs/segment-fold${fold} --fold ${fold} --model fasterrcnn_resnet152_fpn
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


# TODO merge, port the rest

