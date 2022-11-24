# SCAN

This is a re-implementation of the SCAN paper by Lake and Baroni http://proceedings.mlr.press/v80/lake18a/lake18a.pdf

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
python src/scan/train.py \
    --train data/SCAN/simple_split/tasks_train_simple.txt \
    --valid data/SCAN/simple_split/tasks_train_simple.txt \
    --model lstm
```

Scripts for running all experiments using top-performing architectures and the over-best one are under `./scripts`.

## Experimental setup

### Overall-best architecture

The authors name a single architecture as the *overall-best* one.

* 2-layer LSTM
* 200 hidden unist per layer
* No attention
* Dropout applied at the 0.5 level

* Trained for 100k steps using a batch size of 1
* ADAM optimizer is used with learning rate 1e-3
* Gradients are clipped if norm above 5.0
* Decoding uses teacher forcing with 50% chance

Training (!) accuracy in the paper is reported as at least 99.5% for this model in the key experiments. Top-performing models reach above 95% training accuracy.

## Experiment 1

The reported top-performer score is 99.8 % over the test-set, and that of the overall best model is reported as 99.7% on the test-set (averaged over 5 runs). The model only differs from the *overall-best* one in that there is no drop out (and possibly no gradient clipping, though this is unclear in the paper).

## Experiment 2

## Experiment 3