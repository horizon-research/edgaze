#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is the pipeline interface to harness other modules in the eye segmentation stage

@author: Yu Feng
"""
import os
import sys

# add this path to import other modules in the current directory
sys.path.append(os.path.dirname(__file__))

import atexit

import bbox_predictor
import bbox_sift
import cv2
import emulator
import eye_segmentation
import numpy
import torch


#%%

FULL_DENSITY = 100


class Framework(object):
    def __init__(
        self,
        model_name,
        model_path,
        device,
        scaledown,
        record_path,
        use_sift=False,
        disable_edge_info=False,
        bbox_model_path="model_weights/G030_c32_best.pth",
        blur_size=3,
        clip_val=10,
        threshold_ratio=0.3,
        density_threshold=0.05,
        preview=False,
    ):
        """Init.

        Args:
            model_name: str, the name of eye segmentation model
            model_path: str, loading path for model
            device: str, device to run eye segmentation model
            scaledown: int, the dimension scale down number
            record_path: str, the result recording path
            use_sift: bool, enable sift feature descriptor 
                      to tracking the bbox instead of DNN model
            disable_edge_info: bool, disable edge input to the bbox predictor
            blur_size: the gaussian blur size apply to input images
            clip_val: clip all low pixel value to this number
            threshold_ratio: the threshold ratio to activate a event
            density_threshold: the threshold to determine whether performing 
                               DNN eye segmentation on current input image
            preview: bool, whether preview the segmentation result.
        """

        self.simple_emulator = emulator.SimpleEmulator(
            blur_size=blur_size, 
            clip_val=clip_val, 
            threshold_ratio=threshold_ratio
        )

        self.eye_segmentation = eye_segmentation.EyeSegmentation(
            model_name=model_name, 
            model_path=model_path, 
            device=device
        )

        if not use_sift:
            self.bbox_predictor = bbox_predictor.BBoxPredictor(
                    device, scaledown, preview, 
                    bbox_model_path=bbox_model_path,
                    density_threshold=density_threshold,
                    disable_edge_info=disable_edge_info)
        else:
            self.bbox_predictor = bbox_sift.BBoxPredictor(
                    scaledown, preview, 
                    density_threshold=density_threshold)

        self.scaledown = scaledown
        self.preview = preview
        self.use_sift = use_sift

        os.makedirs(record_path, exist_ok=True)

        self.bbox_record_file = open(
            os.path.join(record_path, "bbox_record_file.txt"), "w+"
        )
        self.warp_record = open(os.path.join(record_path, "warp_record.txt"), "w+")

        atexit.register(self.cleanup)

    def cleanup(self):
        """cleanup and close files"""
        self.bbox_record_file.close()
        self.warp_record.close()

    def downsize(self, img, size):
        """downsample the image"""
        img_shape = img.shape
        dim = (img_shape[1] // size, img_shape[0] // size)
        fx = 1.0 / size
        fy = 1.0 / size
        resized_img = cv2.resize(
            src=img, dsize=dim, fx=fx, fy=fy, interpolation=cv2.INTER_AREA
        )

        return resized_img

    def predict(self, framename):
        """
            predict eye segmentation
            the main logic of the pipeline, first emulate events,
            then, predict bounding box and whether to execute DNN,
            lastly, predict eye segmentation result.
        """

        img = cv2.imread(framename, cv2.IMREAD_GRAYSCALE)
        img = img.astype(numpy.float32)
        img_no_resize = numpy.array(img)
        img = self.downsize(img, self.scaledown)

        img = img.astype(numpy.float32)
        events, event_map = self.simple_emulator.generate_events(img)
        if self.use_sift:
            img = self.downsize(img_no_resize, self.scaledown)
            new_bbox = self.bbox_predictor.predict(img)
        else:
            new_bbox = self.bbox_predictor.predict(event_map)

        density = FULL_DENSITY
        if new_bbox is not None:
            density = self.simple_emulator.calc_density(
                events, new_bbox, img.shape, self.scaledown
            )
        else:
            new_bbox = {
                "x_min": 0,
                "x_max": img_no_resize.shape[1],
                "y_min": 0,
                "y_max": img_no_resize.shape[0],
            }

        self.bbox_record_file.write(
            "%d %d %d %d\n"
            % (
                new_bbox["x_min"],
                new_bbox["x_max"],
                new_bbox["y_min"],
                new_bbox["y_max"],
            )
        )

        filter_result = self.eye_segmentation.predict_filtering(
            img_no_resize.astype(numpy.uint8), new_bbox
        )

        able_warp, filter_result = self.bbox_predictor.update(filter_result, density)

        if self.preview:
            comb = numpy.hstack((img_no_resize/255., filter_result/4))
            cv2.imshow("warpped & inference", comb)
            cv2.waitKey(30)

        self.warp_record.write("%f %s\n" % (density, able_warp))

        filter_result = self.eye_segmentation.extract_pupil(
            filter_result, "filter pupil"
        )

        return filter_result
