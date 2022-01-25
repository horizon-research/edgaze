#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is a bounding box predictor that using a DNN algorithm to
track and predict region of interest (ROI) of the current frame.

@author: Yu Feng
"""

import torch
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
            device,
            scaledown,
            preview,
            bbox_model_path,
            density_threshold=0.01, 
            disable_edge_info=False,
            input_width=640,
            input_height=400
        ):
        """Init.

        Args:
            device: str, device to run eye segmentation model
            scaledown: int, the dimension scale down number
            preview: bool, whether preview the segmentation result
            bbox_model_path: str, loading path for bbox prediction network
            density_threshold: the threshold to determine whether performing 
                               DNN eye segmentation on current input image
            disable_edge_info: bool, disable edge input to the bbox predictor
            input_width: input image width
            input_height: input image height
        """

        super(BBoxPredictor, self).__init__()
        self.device = torch.device(device)
        self.scaledown = scaledown
        self.density_threshold = density_threshold
        self.input_width = input_width
        self.input_height = input_height
        self.preview = preview
        self.disable_edge_info = disable_edge_info

        if disable_edge_info:
            self.classifier = classifier.Classifier2D(use_edge_info=False)
            self.classifier.load_state_dict(
                torch.load(
                    bbox_model_path,
                    map_location=torch.device(device)
                )
            )

        else:
            self.classifier = classifier.Classifier2D()
            self.classifier.load_state_dict(
                torch.load(
                    bbox_model_path,
                    map_location=torch.device(device)
                )
            )

        self.classifier = self.classifier.to(torch.device(device))
        self.classifier.eval()

        self.EYE_LABEL_THRESHOLD = 1

        self.prev_bbox = None
        self.prev_bbox_dict = None
        self.curr_bbox_dict = None
        self.prev_seg_result = None

    def predict(self, event):
        """
            use event map to predict bbox of the current frame.
        """
        if self.prev_bbox is None:
            return None

        event = numpy.expand_dims(event, axis=0)
        event = torch.from_numpy(event).float().to(self.device).unsqueeze(0)

        if self.disable_edge_info:
            pred_bbox = self.classifier((event, None, self.prev_bbox)).squeeze()
        else:
            resized_prev_res = cv2.resize(self.prev_seg_result, 
                                          (320, 200), 
                                          interpolation = cv2.INTER_AREA)
            edge_img = cv2.Canny(resized_prev_res.astype("uint8"), 0, 3)/255.
            edge_img = numpy.expand_dims(edge_img, axis=0)
            edge_img = torch.from_numpy(edge_img).float().to(self.device).unsqueeze(0)
            # print(edge_img.shape, event.shape, self.prev_bbox.shape)
            pred_bbox = self.classifier((event, edge_img, self.prev_bbox)).squeeze()
            # print(pred_bbox.shape, pred_bbox)

        curr_bbox = [int(pred_bbox[0] * (self.input_width//2) + self.input_width//2), 
                     int(pred_bbox[1] * (self.input_width//2) + self.input_width//2),
                     int(pred_bbox[2] * (self.input_height//2) + self.input_height//2),
                     int(pred_bbox[3] * (self.input_height//2) + self.input_height//2)] 

        curr_bbox[0] = max(0, (curr_bbox[0]//16-1)*16)
        curr_bbox[1] = min(self.input_width, (curr_bbox[1]//16+1)*16)
        curr_bbox[2] = max(0, (curr_bbox[2]//16-1)*16)
        curr_bbox[3] = min(self.input_height, (curr_bbox[3]//16+1)*16)

        self.prev_bbox = pred_bbox.unsqueeze(0)

        new_bbox = {
            "x_min": curr_bbox[0],
            "x_max": curr_bbox[1],
            "y_min": curr_bbox[2],
            "y_max": curr_bbox[3],
        }
        if self.curr_bbox_dict is not None:
            self.prev_bbox_dict = self.curr_bbox_dict

        self.curr_bbox_dict = new_bbox
        print(new_bbox)

        return new_bbox


    def update(self, seg_result, density):
        """
            based on the current event density to generate the segmentation results
            return:
                1. whether use extrapolation
                2. generated segmentation result
        """
        if self.prev_bbox is None:
            bbox = self.find_bbox(seg_result)
            prev_bbox = [float(bbox["x_min"]-self.input_width//2)/(self.input_width//2), 
                         float(bbox["x_max"]-self.input_width//2)/(self.input_width//2),
                         float(bbox["y_min"]-self.input_height//2)/(self.input_height//2),
                         float(bbox["y_max"]-self.input_height//2)/(self.input_height//2)]

            self.prev_bbox_dict = bbox
            self.prev_bbox = torch.from_numpy(numpy.array(prev_bbox)).float().to(self.device).unsqueeze(0)
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
            a DNN-predicted segmentation result
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






