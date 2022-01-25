#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is a bounding box predictor that using SIFT feature descriptor to
track and predict region of interest (ROI) of the current frame.

@author: Yu Feng
"""

import numpy
import sys
import os
import cv2

sys.path.append(os.path.dirname(__file__))

import classifier


class BBoxPredictor(object):
    """docstring for BBoxPredictor"""
    def __init__(
            self,
            scaledown,
            preview,
            density_threshold=0.01, 
            input_width=640,
            input_height=400
    ):
        """Init.

            Args:
                scaledown: int, the dimension scale down number
                preview: bool, whether preview the segmentation result.
                density_threshold: the threshold to determine whether performing 
                                   DNN eye segmentation on current input image
                input_width: input image width
                input_height: input image height
        """
        super(BBoxPredictor, self).__init__()
        self.scaledown = scaledown
        self.density_threshold = density_threshold
        self.input_width = input_width
        self.input_height = input_height
        self.preview = preview

        self.EYE_LABEL_THRESHOLD = 1

        # construct sift detector
        self.orb = cv2.SIFT_create()
        # BFMatcher with default params
        self.bf = cv2.BFMatcher()
        # some initial configs
        self.prev_kp = None
        self.prev_desp = None

        self.prev_bbox = None
        self.prev_seg_result = None

    def predict(self, curr_img):
        """
            predict bounding box and return bbox result in dict.
        """
        # find keypoints and their descriptors
        curr_kp, curr_desp = self.orb.detectAndCompute(curr_img.astype("uint8"), None)

        if self.prev_bbox is None:
            self.prev_kp = curr_kp
            self.prev_desp = curr_desp
            return None

        matches = self.bf.knnMatch(self.prev_desp,  curr_desp, k=2)

        # find good match
        good_matches = []
        for pair in matches:
            try:
                m, n = pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
            except ValueError:
                pass

        fallback_bbox = {
            "x_min": 0,
            "x_max": self.input_width,
            "y_min": 0,
            "y_max": self.input_height,
        }

        if len(good_matches) < 10:
            self.prev_kp = curr_kp
            self.prev_desp = curr_desp
            # print("return fallback_bbox", len(good_matches))
            return fallback_bbox

        # Extract location of good matches
        src_points = numpy.zeros((len(good_matches), 2), dtype=numpy.float32)
        dst_points = numpy.zeros((len(good_matches), 2), dtype=numpy.float32)

        for i, match in enumerate(good_matches):
            src_points[i, :] = self.prev_kp[match.queryIdx].pt
            dst_points[i, :] = curr_kp[match.trainIdx].pt

        M, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC)
        pts = numpy.float32(
            [[self.prev_bbox["x_min"], self.prev_bbox["y_min"]], 
             [self.prev_bbox["x_min"], self.prev_bbox["y_max"]], 
             [self.prev_bbox["x_max"], self.prev_bbox["y_max"]], 
             [self.prev_bbox["x_max"], self.prev_bbox["y_min"]],
            ]
        ).reshape(-1,1,2)

        try:
            dst = cv2.perspectiveTransform(pts, M)
        except Exception as e:
            dst = pts

        # average two corner coordinates to find the mini/max of the bbox
        x_min = int((dst[0][0][0]+dst[1][0][0])/2)
        y_min = int((dst[0][0][1]+dst[3][0][1])/2)
        x_max = int((dst[2][0][0]+dst[3][0][0])/2)
        y_max = int((dst[1][0][1]+dst[2][0][1])/2)

        # print(pts, dst)
        self.prev_kp = curr_kp
        self.prev_desp = curr_desp
        if x_max <= x_min or y_max <= y_min:
            return fallback_bbox

        if x_min < 0 or x_max >= 640 or y_min < 0 or y_max >= 400:
            return fallback_bbox
        else:
            return {
                "x_min": x_min,
                "x_max": x_max,
                "y_min": y_min,
                "y_max": y_max,
            }

    def update(self, seg_result, density):
        if self.prev_bbox is None:
            bbox = self.find_bbox(seg_result)
            self.prev_bbox = bbox
            self.prev_seg_result = seg_result

            return False, seg_result

        else:
            filtering_result = None
            able_warp = False
            if density < self.density_threshold:
                able_warp = True
                filtering_result = self.prev_seg_result
            else:
                filtering_result = seg_result

            self.prev_seg_result = filtering_result

            return able_warp, filtering_result

    def find_bbox(self, img):
        """
            This is a function to find a bbox from
            a DNN-predicted segmentation result.
            return:
                bbox in a dict
        """
        shape = img.shape

        bbox = {"x_min": shape[1], "x_max": 0, "y_min": shape[0], "y_max": 0}

        bboxs = []
        for c in range(shape[1]):
            check = False
            for r in range(shape[0]):
                if img[r, c] >= self.EYE_LABEL_THRESHOLD:
                    bbox["x_min"] = min(bbox["x_min"], c)
                    bbox["y_min"] = min(bbox["y_min"], r)
                    bbox["x_max"] = max(bbox["x_max"], c)
                    bbox["y_max"] = max(bbox["y_max"], r)
                    check = True

            if not check and bbox["x_max"] > 0:
                bboxs.append(bbox)
                bbox = {"x_min": shape[1], "x_max": 0, "y_min": shape[0], "y_max": 0}

        if len(bboxs) == 0:
            return {"x_min": 0, "x_max": shape[1], "y_min": 0, "y_max": shape[0]}

        # find the biggest bbox as the final bbox
        best_bbox = bboxs[0]
        for bbox in bboxs:
            area = (bbox["x_max"] - bbox["x_min"]) * (bbox["y_max"] - bbox["y_min"])
            best_area = (best_bbox["x_max"] - best_bbox["x_min"]) * (
                best_bbox["y_max"] - best_bbox["y_min"]
            )

            if area > best_area:
                best_bbox = dict(bbox)

        return dict(best_bbox)






