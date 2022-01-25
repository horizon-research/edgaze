#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simple bbox prediction network
it consists of three CONV layers and two FC layers.

@author: Yu Feng
"""

import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class Classifier2D(nn.Module):
    def __init__(
            self, 
            use_edge_info=True, 
            in_channels=1, 
            out_channels=4, 
            channel_size=32, 
            concat=True, 
            dropout=False, 
            prob=0, 
            down_size=2
    ):
        super(Classifier2D, self).__init__()

        self.use_edge_info = use_edge_info

        if use_edge_info:
            in_channels += 1

        self.conv1 = nn.Conv2d(in_channels, channel_size, kernel_size=(3,3), padding=(1,1))
        self.conv2 = nn.Conv2d(channel_size, channel_size, kernel_size=(3,3),padding=(1,1))
        self.conv3 = nn.Conv2d(channel_size, 1, kernel_size=(3,3),padding=(1,1))
        self.max_pool = nn.AvgPool2d(kernel_size=down_size)

        self.fc1 = nn.Linear(1004, 32)
        self.fc2 = nn.Linear(32, out_channels)

        self.relu = nn.LeakyReLU()
        self.down_size = down_size
        self.dropout = dropout
        self.dropout1 = nn.Dropout(p=prob)
        self.dropout2 = nn.Dropout(p=prob)
        self.dropout3 = nn.Dropout(p=prob)

        self._initialize_weights()
    
    def forward(self, x):
        x, x_e, bbox = x

        if self.use_edge_info:
            x = torch.cat((x, x_e), axis=1)
        if self.down_size != None:
            x = self.max_pool(x)
            
        if self.dropout:
            x1 = self.relu(self.dropout1(self.conv1(x)))
            x1 = self.max_pool(x1)
            x2 = self.relu(self.dropout2(self.conv2(x1)))
            x2 = self.max_pool(x2)
            x3 = self.relu(self.dropout3(self.conv3(x2)))
            x3 = torch.flatten(x3, 1)

            x3 = torch.cat((x3, bbox), axis=1)
            x4 = self.relu(self.fc1(x3))
            out = self.fc2(x4)
        else:
            x1 = self.relu(self.conv1(x))
            x1 = self.max_pool(x1)
            x2 = self.relu(self.conv2(x1))
            x2 = self.max_pool(x2)
            x3 = self.relu(self.conv3(x2))
            x3 = torch.flatten(x3, 1)

            x3 = torch.cat((x3, bbox), axis=1)
            x4 = self.relu(self.fc1(x3))
            out = self.fc2(x4)

        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

