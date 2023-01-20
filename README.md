# SCAN

This is a re-implementation of the paper _Generalization without Systematicity: On the Compositional Skills of Sequence-to-Sequence Recurrent Networks_ (SCAN) by Lake and Baroni http://proceedings.mlr.press/v80/lake18a/lake18a.pdf

---
Disclaimer: while this code did produce results similar to those in the original paper there may still be issues found in it. 

---

## Setup

Start by fetching the repository and checking out the submodule to get the dataset

```bash
git clone git@github.com:vesteinn/SCAN.git
cd SCAN
git submodule update --init --recursive
```

Then make sure you have `pytorch` (version 1.12 and 1.13 were tested) and `tqdm` installed into your environment. The project should work both on CPU and GPU.

## Training

To run a full training of the best overall model from the paper simply run the following (the default parameters should correspond to those used in the paper).

```bash
python src/scan/train.py \
    --train data/SCAN/simple_split/tasks_train_simple.txt \
    --valid data/SCAN/simple_split/tasks_test_simple.txt \
    --model lstm
```

Scripts for running experiments 1-3 using top-performing architectures and the overall-best one are under `./scripts`.


## Statistics

After finishing running the experiments, run

```bash
cd stats
python get_stats.py
```

This will output a report with the results from the experiments and save some figures to the stats folder.

## Transformer experiments

An encoder-decoder setup is also included, the main implementation is in `src/scan/alttransformer.py`. The model in `src/scan/transformer_model.py` has some bug when used with the code used to train the other RNN's.

Scripts for running experiment 1b and Experiment 2 are found under `scripts/alttransformer`. 

