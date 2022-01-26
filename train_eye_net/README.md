# ROI Prediction Network

This directory contains the training script for our proposed eye segmentation networks.

## Prerequisites

Before running any code, make sure you download the preprocessed data into current directory. There are three important data sources.
* [eye segmentation dataset](https://rochester.box.com/s/y6ryd043x3y1kvsnwlkhssoo42je4eem): This contains the dataset that we used for training.

Follow the root directory and install the required packages.


## Training

After extracting the three zip files into the current directory. Simply run command:
```
 $ python3 train.py --model MODEL
```
In this directory, we proposed three variants of eye segmentation networks, `eye_net`, `eye_net_m` and `pruned_eye_net`. `pruned_eye_net` is the most lightweight model but less accurate, `eye_net` is the most accurate one but with large model size.

To enable GPU training, please add `--useGPU`option.
```
  $ python3 train.py --model MODEL --useGPU
```
More Options can be seen by running command:
```
 $ python3 train.py --help

usage: train.py [-h] [--dataset DATASET] [--bs BS] [--epochs EPOCHS]
                [--workers WORKERS] [--model MODEL] [--evalsplit EVALSPLIT]
                [--lr LR] [--save SAVE] [--seed SEED] [--load LOAD] [--resume]
                [--test] [--savemodel] [--testrun] [--expname EXPNAME]
                [--useGPU USEGPU]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     name of dataset
  --bs BS
  --epochs EPOCHS       Number of epochs
  --workers WORKERS     Number of workers
  --model MODEL         model name
  --evalsplit EVALSPLIT
                        eval spolit
  --lr LR               Learning rate
  --save SAVE           save folder name
  --seed SEED           random seed
  --load LOAD           load checkpoint file name
  --resume              resume train from load chkpoint
  --test                test only
  --savemodel           checkpoint save the model
  --testrun             test run with few dataset
  --expname EXPNAME     extra explanation of the method
  --useGPU              Set it as False if GPU is unavailable
```
