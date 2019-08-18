Kuzushiji Recognition
=====================

https://www.kaggle.com/c/kuzushiji-recognition/

Install
-------

Use Python 3.6.

Install::

    pip install -r requirements.txt
    python setup.py develop

Run
---

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

.
