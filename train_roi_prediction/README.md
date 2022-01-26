# ROI Prediction Network

This directory contains the training script for the ROI prediction network.

## Prerequisites

Before running any code, make sure you download the preprocessed data into current directory. There are three important data sources.
* [eye segmentation data mask](https://rochester.box.com/s/y6ryd043x3y1kvsnwlkhssoo42je4eem): this contains the eye segmentation masks which are used to extract edge info.
* [event image data](https://rochester.box.com/s/vbu9f40yu1h580zhp811j9fx38luw2ee): this contains preprocessed event images.
* [bbox ground truth](https://rochester.box.com/s/a2cfyyg2gc9v1d0bevqxfxfm6cd7ipvx): this is the ground truth for this task.

## Training

After extracting the three zip files into the current directory. Simply run command:
```
 $ python3 train.py
```
More Options can be seen by running command:
```
 $ python3 train.py --help

usage: train.py [-h] [--device DEVICE] [--num_workers NUM_WORKERS]
                [--disable_edge_info] [--mask_data_path MASK_DATA_PATH]
                [--event_data_path EVENT_DATA_PATH]
                [--bbox_data_path BBOX_DATA_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --device DEVICE       device to run on, 'cpu' or 'cuda', only apply to
                        pytorch, default: CPU
  --num_workers NUM_WORKERS
                        number of workers for the data loader
  --disable_edge_info   disable using edge information in bbox prediction
  --mask_data_path MASK_DATA_PATH
                        eye segmentation map data path
  --event_data_path EVENT_DATA_PATH
                        event map data path
  --bbox_data_path BBOX_DATA_PATH
                        bbox data path
```
