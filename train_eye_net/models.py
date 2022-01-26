#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model factory in a format of dictionary.

@author: Yu Feng
"""

from eye_net import URNet2D
from eye_net_m import URNet2DM
from pruned_eye_net import pruned_eye_net


model_dict = {}
pruned_net_cfg = [8, 10, 32,   11, 45, 32,   24, 55, 32,   33, 65, 32,   36, 68, 32,
       34, 97, 28, 28,   38, 98, 25, 25,    29, 85, 24, 24,   16, 75, 10, 10]

model_dict['eye_net'] = URNet2D(dropout = True, prob = 0.2)
model_dict['eye_net_m'] = URNet2DM(dropout = True, prob = 0.2)
model_dict['pruned_eye_net'] = pruned_eye_net(cfg = pruned_net_cfg)


