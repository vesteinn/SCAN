# SCAN

This is a re-implementation of the SCAN paper by Lake and Baroni https://arxiv.org/abs/1711.00350

## Setup

Start by fetching the repository and checking out the submodule to get the dataset

```bash
git clone git@github.com:vesteinn/SCAN.git
cd SCAN
git submodule update --init --recursive
```

Then make sure you have pytorch and tqdm installed into your environment. The project should work both on CPU and GPU.

## Training

To run a full training of the best overall model from the paper simply run the following (the default parameters should correspond to those used in the paper).

```bash
cd src/SCAN/
python train.py \
    --train ../../data/SCAN/simple_split/tasks_train_simple.txt \
    --valid ../../data/SCAN/simple_split/tasks_train_simple.txt \
    --model lstm
```
