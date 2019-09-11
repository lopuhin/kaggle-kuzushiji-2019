Kuzushiji Recognition
=====================

https://www.kaggle.com/c/kuzushiji-recognition/

Install
-------

Use Python 3.6.

Install::

    pip install -r requirements.txt
    python setup.py develop

Install libjpeg-turbo as documented at https://github.com/ajkxyz/jpeg4py


Run
---

#. Prepare data for the language model (not used at the moment)::

    python -m kuzushiji.lm.dataset

#. Convert images to numpy (takes about 76 GB extra on disk)::

    python -m kuzushiji.jpeg2np

#. Train segmentation model across all folds::

    python -m kuzushiji.segment.main \
        --output-dir _runs/fold0
    ...

#. Run it with ``--test-only`` to generate OOF ground truth for classification::

    python -m kuzushiji.segment.main \
        --output-dir _runs/fold0 \
        --resume _runs/fold0/model_last.pth \
        --test-only
    ...

#. Join all dataframes into one::

    paths = Path('_runs').glob('fold*/clf_gt.csv')
    df = pd.concat([pd.read_csv(p) for p in paths]
    df.to_csv('_runs/clf_gt.csv')

#. Train classification model::

    python -m kuzushiji.classify.main \
        _runs/clf_gt.csv \
        --output-dir _runs/clf

#. Create test predictions from segmentation model::

    python -m kuzushiji.segment.main \
        --output-dir _runs/fold0 \
        --resume _runs/fold0/model_last.pth \
        --submission

#. Create submission from classification model, using file created by the
   previous command as the first argument::

    python -m kuzushiji.classify.main \
        _runs/fold0/test_predictions.csv \
        --output-dir _runs/clf \
        --resume _runs/clf/model_best.pth \
        --submission

.
