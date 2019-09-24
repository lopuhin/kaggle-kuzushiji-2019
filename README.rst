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

All steps required for submission are in ``./run-all.sh``.

TODO port it as well, GPU memory requirements.

#. Create submission from classification model, using file created by the
   previous command as the first argument::

    python -m kuzushiji.classify.main \
        _runs/segment-fold0/test_predictions.csv \
        --output-dir _runs/clf \
        --resume _runs/clf/model_best.pth \
        --submission

.
