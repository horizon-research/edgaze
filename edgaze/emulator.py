#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is simplified event generator to generate events between two 
temporally-adjacent images.

@author: Yu Feng
"""

import cv2
import numpy
import os

COLOR_MAX = 255


class SimpleEmulator(object):
    """This is a simplified event emulator

    """
    def __init__(self,
                 blur_size=3,
                 clip_val=10,
                 threshold_ratio=0.3):
        """ Init.

        Args:
            blur_size: the gaussian blur size apply to input images
            clip_val: clip all low pixel value to this number
            threshold ratio: the threshold ratio to activate an event
        """
        self.blur_size = blur_size
        self.clip_val = clip_val
        self.ratio = threshold_ratio

        self.clipped_prev_frame = None

    def generate_events(self, curr_frame):
        """
            generate events as an list, and diff map as a numpy array
        """
        # blur the image first
        curr_frame = cv2.blur(curr_frame, (self.blur_size, self.blur_size))
        clipped_curr_frame = numpy.clip(curr_frame, self.clip_val, COLOR_MAX)

        events = []
        diff = None

        if self.clipped_prev_frame is not None:
            diff = clipped_curr_frame - self.clipped_prev_frame
            ratio_frame = numpy.abs(diff)/self.clipped_prev_frame
            diff = numpy.clip(ratio_frame, self.ratio, 1) - self.ratio
            
            for row in range(diff.shape[0]):
                for col in range(diff.shape[1]):
                    if diff[row, col] > 0:
                        diff[row, col] = 1
                        events.append([0, col, row, 0])

        self.clipped_prev_frame = clipped_curr_frame

        events == numpy.array(events)

        return events, diff

    def calc_density(self, events, bbox, shape, scale):
        """
            calculate the density within the bbox, in percentage (%).
        """
        cnt = 0

        for event in events:
            row = event[1]*scale
            col = event[2]*scale
            if row >= bbox["x_min"] and row <= bbox["x_max"] and \
                col >= bbox["y_min"] and col <= bbox["y_max"]:
                cnt += 1

        bbox_area = (bbox["x_max"] - bbox["x_min"]) * (bbox["y_max"] - bbox["y_min"])
        density = float(cnt)/bbox_area

        return density*100

