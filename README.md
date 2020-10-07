# FRBs and CNNs

This project aims to perform a binary classification whether observations are [*Fast Radio Bursts*](https://arxiv.org/abs/1904.07947) or *not* (typically *not* means it is *Radio Frequency Interference*) using Convolutional Neural Networks (CNNs).

Included in this repo are some illustrative scripts. These apply the fundamental building blocks of CNNs to some example images (see the `example_data` folder).

- `NumpyConv.py` shows the result of applying 2 simple *convolutional filters/kernels* across an image
- `NumpyMaxPool.py` shows the result of maxpooling on an image