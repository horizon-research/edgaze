#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is the training script to train the bbox prediction network.

@author: Yu Feng
"""

import os
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from classifier import Classifier2D
from dataloader import EyeDataset


def option():
    parser = argparse.ArgumentParser()

    # options
    parser.add_argument(
        "--device",
        type=str,
        help="device to run on, 'cpu' or 'cuda', only apply to pytorch, default: CPU",
        default="cpu",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="number of workers for the data loader",
    )
    parser.add_argument(
        "--disable_edge_info",
        default=False,
        action="store_true",
        help="disable using edge information in bbox prediction",
    )
    parser.add_argument(
        "--mask_data_path",
        type=str,
        default="openEDS",
        help="eye segmentation map data path",
    )
    parser.add_argument(
        "--event_data_path",
        type=str,
        default="openEDS_events",
        help="event map data path",
    )
    parser.add_argument(
        "--bbox_data_path",
        type=str,
        default="bbox",
        help="bbox data path",
    )

    # parse
    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = option()

    inputHeight = 400
    inputWidth = 640
    use_edge_info = not args.disable_edge_info
    num_workers = args.num_workers
    device = args.device
    folder_path = args.mask_data_path
    event_folder_path = args.event_data_path
    bbox_folder_path = args.bbox_data_path

    os.makedirs('model_weights',exist_ok=True)

    classifer = Classifier2D(use_edge_info=use_edge_info, dropout=True, prob=0.2)
    classifer = classifer.to(torch.device(device))

    eye_dataset = EyeDataset(folder_path=folder_path, 
                             event_folder_path=event_folder_path,
                             bbox_folder_path=bbox_folder_path)
    dataloader = DataLoader(eye_dataset, batch_size=8, shuffle=False, num_workers=num_workers)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(classifer.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(250):
        avg_loss = 0.
        cnt = 0.
        for batch_idx, data in enumerate(dataloader):
            event_img, edge_img, labels, prev_bboxs, event_path = data

            # load to targeted device
            event_img = event_img.to(device)
            edge_img = edge_img.to(device)
            labels = labels.to(device)
            prev_bboxs = prev_bboxs.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = classifer((event_img, edge_img, prev_bboxs.float()))
            # print(outputs.shape, labels.shape)

            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            cnt += 1

            # print out result every 1000 iterations
            if batch_idx % 10 == 0:
                outputs = outputs[0]
                bbox = labels[0]
                print("epoch[%d/%d]" % (epoch, batch_idx), loss.item(), 
                      [int(outputs[0] * (inputWidth//2) + inputWidth//2), 
                       int(outputs[1] * (inputWidth//2) + inputWidth//2),
                       int(outputs[2] * (inputHeight//2) + inputHeight//2),
                       int(outputs[3] * (inputHeight//2) + inputHeight//2)], 
                       [int(bbox[0] * (inputWidth//2) + inputWidth//2), 
                       int(bbox[1] * (inputWidth//2) + inputWidth//2),
                       int(bbox[2] * (inputHeight//2) + inputHeight//2),
                       int(bbox[3] * (inputHeight//2) + inputHeight//2)])
                print(event_path[0])
            
        print("epoch[%d] loss: %f" % (epoch, avg_loss/cnt))

        # save model weights every 5 epoches.
        if epoch % 5 == 4:
            torch.save(classifer.state_dict(), 'model_weights/G030_c32_%d.pth' % epoch)




