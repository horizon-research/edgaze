#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One variant of our eye segmentation network

@author: Yu Feng
"""

import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class URNet2D_down_block(nn.Module):
    def __init__(self,input_channels,output_channels,down_size,expand_scale=4,dropout=False,prob=0):
        super(URNet2D_down_block, self).__init__()
        self.conv1 = nn.Conv2d(input_channels,output_channels*expand_scale,kernel_size=(1,1),padding=(0,0))
        self.conv21 = nn.Conv2d(output_channels*expand_scale,
                                output_channels*expand_scale,
                                kernel_size=(3,3),
                                padding=(1,1),
                                groups=output_channels*expand_scale)
        self.conv22 = nn.Conv2d(input_channels+output_channels*expand_scale,
                                output_channels,
                                kernel_size=(1,1),
                                padding=(0,0))
        self.max_pool = nn.AvgPool2d(kernel_size=down_size)            
        
        self.relu = nn.LeakyReLU()
        self.down_size = down_size
        self.dropout = dropout
        self.dropout1 = nn.Dropout(p=prob)
        self.dropout2 = nn.Dropout(p=prob)
        self.dropout3 = nn.Dropout(p=prob)
        self.bn = torch.nn.BatchNorm2d(num_features=output_channels)
    
    def forward(self, x):
        if self.down_size != None:
            x = self.max_pool(x)
            
        if self.dropout:
            x1 = self.relu(self.dropout1(self.conv1(x)))
            x21 = self.relu(self.dropout2(self.conv21(x1)))
            x21 = torch.cat((x,x21),dim=1)
            out = self.relu(self.dropout3(self.conv22(x21)))
        else:
            x1 = self.relu(self.conv1(x))
            x21 = self.relu(self.conv21(x1))
            x21 = torch.cat((x,x21),dim=1)
            out = self.relu(self.conv22(x21))
        return self.bn(out)
    
    
class URNet2D_up_block_concat(nn.Module):
    def __init__(self,skip_channels,input_channels,output_channels,up_stride,expand_scale=4,dropout=False,prob=0):
        super(URNet2D_up_block_concat, self).__init__()
        self.conv11 = nn.Conv2d(skip_channels+input_channels,output_channels*expand_scale,kernel_size=(1,1),padding=(0,0))
        self.conv12 = nn.Conv2d(output_channels*expand_scale,
                                output_channels*expand_scale,
                                kernel_size=(3,3),
                                padding=(1,1),
                                groups=output_channels*expand_scale)
        self.conv21 = nn.Conv2d(skip_channels+input_channels+output_channels*expand_scale,
                                output_channels,
                                kernel_size=(1,1),
                                padding=(0,0))
        self.conv22 = nn.Conv2d(output_channels,
                                output_channels,
                                kernel_size=(3,3),
                                padding=(1,1),
                                groups=output_channels)
        self.relu = nn.LeakyReLU()
        self.up_stride = up_stride
        self.dropout = dropout
        self.dropout1 = nn.Dropout(p=prob)
        self.dropout2 = nn.Dropout(p=prob)
        self.dropout3 = nn.Dropout(p=prob)

    def forward(self,prev_feature_map,x):
        x = nn.functional.interpolate(x,scale_factor=self.up_stride,mode='nearest')
        x = torch.cat((x,prev_feature_map),dim=1)
        if self.dropout:
            x11 = self.relu(self.dropout1(self.conv11(x)))
            x1 = self.relu(self.dropout2(self.conv12(x11)))
            x21 = torch.cat((x,x1),dim=1)
            out = self.relu(self.dropout3(self.conv22(self.conv21(x21))))
        else:
            x11 = self.relu(self.conv11(x))
            x1 = self.relu(self.conv12(x11))
            x21 = torch.cat((x,x1),dim=1)
            out = self.relu(self.conv22(self.conv21(x21)))
        return out
    
class URNet2DM(nn.Module):
    def __init__(self,in_channels=1,out_channels=4,channel_size=24,concat=True,dropout=False,prob=0):
        super(URNet2DM, self).__init__()
        expand_scale = 4
        in_channels = [1, 16, 16, 24, 32, 32, 32, 24, 16, 16]
        skip_channels = [32, 24, 16, 16]

        self.down_block1 = URNet2D_down_block(input_channels=in_channels[0],output_channels=in_channels[1],
                                                 down_size=None,expand_scale=expand_scale,dropout=dropout,prob=prob)
        self.down_block2 = URNet2D_down_block(input_channels=in_channels[1],output_channels=in_channels[2],
                                                 down_size=(2,2),expand_scale=expand_scale,dropout=dropout,prob=prob)
        self.down_block3 = URNet2D_down_block(input_channels=in_channels[2],output_channels=in_channels[3],
                                                 down_size=(2,2),expand_scale=expand_scale,dropout=dropout,prob=prob)
        self.down_block4 = URNet2D_down_block(input_channels=in_channels[3],output_channels=in_channels[4],
                                                 down_size=(2,2),expand_scale=expand_scale,dropout=dropout,prob=prob)
        self.down_block5 = URNet2D_down_block(input_channels=in_channels[4],output_channels=in_channels[5],
                                                 down_size=(2,2),expand_scale=expand_scale,dropout=dropout,prob=prob)

        self.up_block1 = URNet2D_up_block_concat(skip_channels=skip_channels[0],input_channels=in_channels[5],
                                                    output_channels=in_channels[6],up_stride=(2,2),
                                                    expand_scale=expand_scale,dropout=dropout,prob=prob)
        self.up_block2 = URNet2D_up_block_concat(skip_channels=skip_channels[1],input_channels=in_channels[6],
                                                    output_channels=in_channels[7],up_stride=(2,2),
                                                    expand_scale=expand_scale,dropout=dropout,prob=prob)
        self.up_block3 = URNet2D_up_block_concat(skip_channels=skip_channels[2],input_channels=in_channels[7],
                                                    output_channels=in_channels[8],up_stride=(2,2),
                                                    expand_scale=expand_scale,dropout=dropout,prob=prob)
        self.up_block4 = URNet2D_up_block_concat(skip_channels=skip_channels[3],input_channels=in_channels[8],
                                                    output_channels=in_channels[9],up_stride=(2,2),
                                                    expand_scale=expand_scale,dropout=dropout,prob=prob)

        self.out_conv1 = nn.Conv2d(in_channels=in_channels[9],out_channels=out_channels,kernel_size=1,padding=0)
        self.concat = concat
        self.dropout = dropout
        self.dropout1 = nn.Dropout(p=prob)
        
        self._initialize_weights()
        
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
                
    def forward(self,x):
        self.x1 = self.down_block1(x)
        self.x2 = self.down_block2(self.x1)
        self.x3 = self.down_block3(self.x2)
        self.x4 = self.down_block4(self.x3)
        self.x5 = self.down_block5(self.x4)
        self.x6 = self.up_block1(self.x4, self.x5)
        self.x7 = self.up_block2(self.x3, self.x6)
        self.x8 = self.up_block3(self.x2, self.x7)
        self.x9 = self.up_block4(self.x1, self.x8)
        if self.dropout:
            out = self.out_conv1(self.dropout1(self.x9))
        else:
            out = self.out_conv1(self.x9)
                       
        return out

if __name__ == '__main__':
    model = URNet2DM()
    inputs = torch.rand(1, 1, 640, 400)
    model.eval()
    output = model(inputs)

