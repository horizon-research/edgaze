#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is the data loader for training ROI prediction network

Three data sources are needed here:
    1. ground truth eye segmentation results
    2. camera emulated event images
    3. ground truth bounding box

@author: Yu Feng
"""

import glob
import random
import numpy as np
import torch
import cv2
import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class EyeDataset(Dataset):
    """boudning box prediction data loader"""

    def __init__(self, folder_path, event_folder_path, bbox_folder_path):
        """
            init function
            args:
                folder_path: str, the data path for eye segmentation ground truth,
                            this is one input for training bbox prediction
                event_folder_path: str, the path for emulated event map, this is another
                                  input for training
                bbox_folder_path: str, the data path for ground truth bbox.
        """
        self.folder_path = folder_path
        self.event_folder_path = event_folder_path

        # for image transformation
        self.tensor_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize([0.5], [0.5])])

        self.bbox_list, self.bboxs = self.load_bbox(bbox_folder_path)

    def load_bbox(self, bbox_folder_path):
        """load bbox list, it is used for getitem function"""
        bbox_files = sorted(glob.glob('%s/*.txt' % bbox_folder_path))

        bbox_list = []
        bboxs = {}
        for i in range(len(bbox_files)):
            bbox_name = "%s/S_%d.txt" % (bbox_folder_path, i)
            if not os.path.isfile(bbox_name):
                continue
            bbox_file = open(bbox_name, "r")
            bboxs[i] = []
            cnt = 1
            for line in bbox_file.readlines():
                line = line.rstrip()
                bboxs[i].append([int(num) for num in line.split(" ")])
                bbox_list.append((i, cnt))
                cnt += 1
            bbox_list.pop()


        return bbox_list, bboxs

    def __getitem__(self, index):
        """get item function"""
        # find sequence index and frame index
        serial_num, idx = self.bbox_list[index]
        # generate the input path
        event_path = "%s/S_%d/event_frames/%d.npy" % (self.event_folder_path, serial_num, idx)
        mask_path = "%s/S_%d/%d.npy" % (self.folder_path, serial_num, idx-1)
        # print(event_path, len(self.event_files), index)
        # load previous bbox, it is also input to bbox prediction network
        prev_bbox = np.array(self.bboxs[serial_num][idx-1]).copy().astype(float)
        # check if the index is out of bound
        if idx >= len(self.bboxs[serial_num]):
            print(serial_num, idx)

        # laod current bbox
        bbox = np.array(self.bboxs[serial_num][idx]).copy().astype(float)

        event = np.load(event_path)
        event = np.expand_dims(event, axis=0)
        event = torch.from_numpy(event).float()

        shape = event.shape
        masks = np.load(mask_path)
        dim = (masks.shape[1]//2, masks.shape[0]//2)
        masks = cv2.resize(masks.astype("uint8"), dim, interpolation = cv2.INTER_NEAREST)
        edge_img = cv2.Canny(np.array(masks),0,3)/255.
        edge_img = np.expand_dims(edge_img, axis=0).astype(float)
        edge_img = torch.from_numpy(edge_img).float()

        # normalize bbox
        bbox[0] = (bbox[0] - shape[2]) / float(shape[2])
        bbox[1] = (bbox[1] - shape[2]) / float(shape[2])
        bbox[2] = (bbox[2] - shape[1]) / float(shape[1])
        bbox[3] = (bbox[3] - shape[1]) / float(shape[1])

        prev_bbox[0] = (prev_bbox[0] - shape[2]) / float(shape[2])
        prev_bbox[1] = (prev_bbox[1] - shape[2]) / float(shape[2])
        prev_bbox[2] = (prev_bbox[2] - shape[1]) / float(shape[1])
        prev_bbox[3] = (prev_bbox[3] - shape[1]) / float(shape[1])

        return event, edge_img, bbox, prev_bbox, event_path

    def __len__(self):
        return 100 #len(self.bbox_list)
