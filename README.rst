Kuzushiji Recognition
=====================

https://www.kaggle.com/c/kuzushiji-recognition/

.. contents::

Install
-------

Use Python 3.6.

Install::

    pip install -r requirements.txt
    python setup.py develop

Install libjpeg-turbo for your OS as documented at https://github.com/ajkxyz/jpeg4py

Install apex as documented at https://github.com/NVIDIA/apex
(python-only is good enough)

Run
---

All steps required for submission are in heavily commented ``run-all.sh``
and ``level2.sh``. Note that ``run-all.sh`` was never run in one step
(it would take too much time), so may contain errors.
Also all model parameters are in the ``models`` folder
(they come from actual models).

One extra model was used to create pseudolabels which is not used in the final
solution (resnext101_32x16d_wsl on fold0 with batch size 10,
instead of resnext101_32x8d_wsl on the same fold),
but it's contribution is extremely minor and
quality is very similar to resnext101_32x8d_wsl.

All run logs and configs (for classification models) are in ``_runs`` folder.

Overview
--------

TODO

License
-------

License is MIT.
Files under ``detection`` are taken from torchvision with minor modifications,
which is licensed under BSD-3-Clause.
