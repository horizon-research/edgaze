#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is a class for Eye segmentation application, the class can process 
a stream of images and store the result to the designated directory.

In addition, it also providee an interface to process only the bbox portion,
using our API.

@author: Yu Feng
"""
import os
import sys

# add this path to import densenet, don't change it
sys.path.append(os.path.dirname(__file__))

import cv2
import eye_net
import eye_net_m
import pruned_eye_net
import numpy
import PIL
import skimage
import torch
import torchvision

# some constant configurations
COLOR_MAX = 255
COLOR_CAP = 256
EYE_CLASS = 1
IMAGE_MOD = 16
BBOX_EXTRA_SPACE = 20
CLIP_LIMIT = 1.5
TILE_GRID_SIZE = 8


class EyeSegmentation(object):
    def __init__(self, model_name, model_path, device="cpu", preview=False):
        """Init.

        Args:
            model_path: str, loading path for model
            device: str, device to run eye segmentation model
            preview: bool, whether preview the segmentation result
        """
        # load model
        if not os.path.exists(model_path):
            print(model_path)
            print("model path not found !!!")
            exit(1)

        self.preview = preview

        self.device = torch.device(device)

        # construct torch model
        if model_name == "eye_net":
            self.model = eye_net.URNet2D()
        elif model_name == "eye_net_m":
            self.model = eye_net_m.URNet2DM()
        elif model_name == "pruned_eye_net":
            # this is the configuration of pruned eye-net
            cfg = [8, 10, 32,   11, 45, 32,   24, 55, 32,   33, 65, 32,   36, 68, 32,
                   34, 97, 28, 28,   38, 98, 25, 25,    29, 85, 24, 24,   16, 75, 10, 10]
            self.model = pruned_eye_net.pruned_eye_net(cfg=cfg)
        else:
            print([model_name])
            raise NotImplementedError

        self.model = self.model.to(self.device)

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()

        # for image transformation
        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.clahe = cv2.createCLAHE(
            clipLimit=CLIP_LIMIT, tileGridSize=(TILE_GRID_SIZE, TILE_GRID_SIZE)
        )

    def get_img(self, img):
        """read and finish the pre-processing on image"""

        H, W = img.shape[0], img.shape[1]

        # Fixed gamma value for pprocessng raw image
        table = float(COLOR_MAX) * (numpy.linspace(0, 1, COLOR_CAP) ** 0.8)
        img = cv2.LUT(numpy.array(img), table)

        img = self.clahe.apply(numpy.array(numpy.uint8(img)))
        img = PIL.Image.fromarray(img)

        # normalize pixel value to [-1, 1]
        img = self.transform(img)

        return numpy.array(img)

    def extract_pupil(self, predict, name):
        """
            this function extract pupil from segmentation map,
            pupil result is used in the later gaze prediction process.
        """
        predict = numpy.array(predict)
        bbox = self.find_bbox(predict)
        if numpy.max(predict) > 0:
            predict = predict / numpy.max(predict)
        blank_img = numpy.zeros_like(predict)
        blank_img[
            bbox["y_min"] : bbox["y_max"], bbox["x_min"] : bbox["x_max"]
        ] = predict[bbox["y_min"] : bbox["y_max"], bbox["x_min"] : bbox["x_max"]]

        predict = blank_img

        low_pass_filter = predict < EYE_CLASS
        predict[low_pass_filter] = 0

        if self.preview:
            cv2.imshow(name, predict)
            cv2.waitKey(30)

        predict = numpy.expand_dims(predict, axis=0)

        return predict

    def get_predictions_pytorch(self, output):
        """return eye category prediction for each pixel"""
        bs, c, h, w = output.size()
        values, indices = output.cpu().max(1)
        indices = indices.view(bs, h, w)  # bs x h x w
        return indices

    def predict(self, framename):
        """predict eye segmentation using the entire input image"""

        print(framename)
        img = cv2.imread(framename, cv2.IMREAD_GRAYSCALE)
        img = img.astype(numpy.float32)
        print("eye segmentation baseline", img.shape)
        img = skimage.color.rgb2gray(img).squeeze().astype(numpy.uint8)
        img = self.get_img(img)

        data = torch.tensor(img, device=self.device).float().unsqueeze(0)
        output = self.model(data)
        predict = numpy.array(self.get_predictions_pytorch(output))

        predict = predict[0]
        predict = self.extract_pupil(predict, "original pupil")

        return predict

    def predict_filtering(self, img, bbox):
        """predict eye segmentation with filtering bbox

        Args:
            img: 2D image.
            bboxs: bouding boxs for cropping
            scale: bbox processing scale
        """
        print(img.shape)
        img_shape = img.shape
        img = self.get_img(img)
        if img.ndim == 2:
            img = numpy.expand_dims(img, axis=0)
        result_frame = numpy.zeros_like(img[0])

        # increase the bbox size by reducing the y_min
        y_min = max(bbox["y_min"] - BBOX_EXTRA_SPACE, 0)
        x_min = bbox["x_min"]
        y_max = min((bbox["y_max"] + 1), img_shape[0])
        x_max = min((bbox["x_max"] + 1), img_shape[1])

        # make the cropped image dimension to be multiple of 16
        y_max = y_max - ((y_max - y_min) % IMAGE_MOD)
        x_max = x_max - ((x_max - x_min) % IMAGE_MOD)
   
        # crop the original image based on bbox
        frame = img[:, y_min:y_max, x_min:x_max]
        data = torch.tensor(frame, device=self.device).float().unsqueeze(0)
        # predict
        output = self.model(data)
        predict = self.get_predictions_pytorch(output)
        result_frame[y_min:y_max, x_min:x_max] = predict[0]

        self.current_filter_result = result_frame

        return result_frame

    def find_bbox(self, img):
        """find the region most likely to be the eye and find its bbox

        Args:
            img: output from the eye segmentation
        """
        shape = img.shape

        bbox = {"x_min": shape[1], "x_max": 0, "y_min": shape[0], "y_max": 0}

        bboxs = []
        for c in range(shape[1]):
            check = False
            for r in range(shape[0]):
                if img[r, c] >= EYE_CLASS:
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

        # find the biggest region to be the bbox
        best_bbox = bboxs[0]
        for bbox in bboxs:
            area = (bbox["x_max"] - bbox["x_min"]) * (bbox["y_max"] - bbox["y_min"])

            best_area = (best_bbox["x_max"] - best_bbox["x_min"]) * (
                best_bbox["y_max"] - best_bbox["y_min"]
            )

            if area > best_area:
                best_bbox = dict(bbox)

        return dict(best_bbox)
