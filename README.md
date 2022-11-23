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

Accuracy in the paper is reported as above 99.5% for this model in the key experiments.


