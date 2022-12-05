# SCAN

This is a re-implementation of the paper _Generalization without Systematicity: On the Compositional Skills of Sequence-to-Sequence Recurrent Networks_ (SCAN) by Lake and Baroni http://proceedings.mlr.press/v80/lake18a/lake18a.pdf

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
    --valid data/SCAN/simple_split/tasks_test_simple.txt \
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

---

Reproduced experiments from the paper are described below.

## Experiment 1

### 80 / 20 split
The reported top-performer score is 99.8 % over the test-set, and that of the overall best model is reported as 99.7% on the test-set (averaged over 5 runs). The model only differs from the *overall-best* one in that there is no drop out (and possibly no gradient clipping, though this is unclear in the paper).

### Varying number of distinct samples
Besides the 80 / 20 split, other experiments using 1, 2, 4, 8, 16, 32 and 64 % of the training data were made.

> (From paper) With 1% of the commands shown during training (about 210 examples), the network performs poorly at about 5% correct. With 2% coverage, performance improves to about 54% correct on the test set. By 4% coverage, performance is about 93% correct. 

## Experiment 2

### Split by length
In this experiment sequences of length <=22 are used to train and those above for testing.

The best result was achieved using
* Single layer GRU
  - With attention
  - 50-dimensional hidden layer
  - Dropout 0.5

```bash
python src/scan/train.py \
    --train data/SCAN/length_split/tasks_train_length.txt \
    --valid data/SCAN/length_split/tasks_test_length.txt \
    --model gru
    --hidden_dim 50
    --layers 1 
    --use_attention True
```

## Experiment 3

### Split by Jump and Turn Left
In this experiment sequences are divided into two parts according to primitive commands (“jump” and “turn left”).

The best result for TURN LEFT was achieved using
* Single layer GRU
  - With attention
  - 100-dimensional hidden layer
  - Dropout 0.1

```bash
python src/scan/train.py \
    --train data/SCAN/add_prim_split/tasks_train_addprim_turn_left.txt \
    --valid data/SCAN/add_prim_split/tasks_test_addprim_turn_left.txt \
    --model gru
    --dropout 0.1
    --hidden_dim 100 
    --layers 1 
    --use_attention True
```

The best result for JUMP was achieved using
* Single layer LSTM
  - With attention
  - 100-dimensional hidden layer
  - Dropout 0.1

```bash
python src/scan/train.py \
    --train data/SCAN/add_prim_split/tasks_test_addprim_jump.txt \
    --valid data/SCAN/add_prim_split/tasks_test_addprim_jump.txt \
    --model lstm  
    --hidden_dim 100 
    --layers 1 
    --dropout 0.1 
    --use_attention True
```

## Experiment 4

