#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is the main program to invoke the rest of the gaze estimation code

@author: Yu Feng
"""
import argparse
import glob
import os
import sys

import deepvog
import edgaze

def option():
    parser = argparse.ArgumentParser()
    # Data input settings
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="openEDS",
        help="path to openEDS dataset, default: openEDS",
    )
    parser.add_argument(
        "--sequence", type=int, help="The sequence number of the dataset", default=1
    )
    parser.add_argument(
        "--all_sequence",
        default=False,
        action="store_true",
        help="Evaluate all sequences in dataset",
    )
    parser.add_argument(
        "--preview",
        default=False,
        action="store_true",
        help="preview the result, default: False",
    )

    # model information
    parser.add_argument(
        "--model",
        type=str,
        default="eye_net_m",
        help="dnn model, support eye_net, eye_net_m, pruned_eye_net",
    )
    parser.add_argument(
        "--pytorch_model_path",
        type=str,
        help="pytorch model path",
        default="model_weights/EyeModel_best_mIoU.pth",
    )
    parser.add_argument(
        "--disable_edge_info",
        default=False,
        action="store_true",
        help="disable using edge information in bbox prediction",
    )
    parser.add_argument(
        "--use_sift",
        default=False,
        action="store_true",
        help="use sift descriptor to extrapolate bbox",
    )
    parser.add_argument(
        "--bbox_model_path",
        type=str,
        default="model_weights/G030_c32_best.pth",
        help="bbox model path",
    )
    parser.add_argument(
        "--device",
        type=str,
        help="device to run on, 'cpu' or 'cuda', only apply to pytorch, default: CPU",
        default="cpu",
    )

    # camera setting
    parser.add_argument(
        "--video_shape",
        nargs=2,
        type=int,
        default=[400, 640],
        help="video_shape, [HEIGHT, WIDTH] default: [400, 640]",
    )
    parser.add_argument(
        "--sensor_size",
        nargs=2,
        type=float,
        default=[3.6, 4.8],
        help="sensor_shape, [HEIGHT, WIDTH] default: [3.6, 4.8]",
    )
    parser.add_argument(
        "--focal_length", type=int, default=6, help="camera focal length, default: 6"
    )

    # ROI-related settings
    parser.add_argument(
        "--mode",
        type=str,
        default="org",
        help="processing mode, org: use baseline [default], filter: use smart camera filter",
    )
    parser.add_argument(
        "--scaledown",
        type=int,
        default=2,
        help="scaledown when tracking bbox to reduce computation, default: 2",
    )
    parser.add_argument(
        "--blur_size",
        type=int,
        default=3,
        help="blur the input image when tracking bbox",
    )
    parser.add_argument(
        "--clip_val",
        type=int,
        default=10,
        help="clip value in event emulator, clip all low pixel value to this number, default: 10",
    )
    parser.add_argument(
        "--threshold_ratio",
        type=float,
        default=0.3,
        help="threshold ratio to activate an event, default: 0.3",
    )
    parser.add_argument(
        "--density_threshold",
        type=float,
        default=0.05,
        help="threshold ratio to warp result, default: 0.05",
    )

    # output path
    parser.add_argument("--output_path", help="save folder name", default="output_demo")
    parser.add_argument("--suffix", help="save folder name", default="")

    # parse
    args = parser.parse_args()

    return args


def make_dirs(prefix, suffix):
    video_dir = "%s/videos" % prefix
    os.makedirs(video_dir, exist_ok=True)
    eye_model_dir = "%s/fit_models" % prefix
    os.makedirs(eye_model_dir, exist_ok=True)
    result_dir = "%s/results%s" % (prefix, suffix)
    os.makedirs(result_dir, exist_ok=True)


def main():

    args = option()

    if args.mode != "org" and args.mode != "filter":
        print("[ERROR]: '--mode' only support 'org' and 'filter'.")
        exit()

    # some initial setting
    video_shape = (args.video_shape[0], args.video_shape[1])
    sensor_size = (args.sensor_size[0], args.sensor_size[1])
    focal_length = args.focal_length

    model_path = args.pytorch_model_path
    use_sift = args.use_sift
    disable_edge_info = args.disable_edge_info

    keyword = args.mode
    result_prefix = args.output_path
    result_suffix = args.suffix

    # make output directory
    make_dirs(result_prefix, result_suffix)

    # this dataset has 200 sequences of data in total
    seq_start = 0
    seq_end = len(glob.glob(args.dataset_path + "/*"))

    if not args.all_sequence:
        seq_start = args.sequence
        seq_end = args.sequence + 1

    for i in range(seq_start, seq_end):
        # try:
        # iterate the entire datasets
        dataset = "%s/S_%d" % (args.dataset_path, i)
        log_record_path = "%s/S_%d%s" % (result_prefix, i, result_suffix)
        output_record_path = "%s/results%s/%s_result_S_%d.csv" % (
            result_prefix,
            result_suffix,
            keyword,
            i,
        )
        video_name = "%s/videos/%s_S_%d.avi" % (result_prefix, keyword, i)
        model_name = "eye_fit_models/S_%d.json" % (i)

        # Load our pre-trained network for baseline
        model = edgaze.eye_segmentation.EyeSegmentation(
            model_name=args.model,
            model_path=model_path,
            device=args.device,
            preview=args.preview,
        )

        # Init our smart camera filter instance
        filter_model = edgaze.pipeline.Framework(
            model_name=args.model,
            model_path=model_path,
            device=args.device,
            scaledown=args.scaledown,
            record_path=log_record_path,
            use_sift=use_sift,
            disable_edge_info=disable_edge_info,
            bbox_model_path=args.bbox_model_path,
            blur_size=args.blur_size,
            clip_val=args.clip_val,
            threshold_ratio=args.threshold_ratio,
            density_threshold=args.density_threshold,
            preview=args.preview,
        )

        # Initialize the class. It requires information of your camera's focal
        # length and sensor size, which should be available in product manual.
        inferer = deepvog.gaze_inferer(
            model, filter_model, focal_length, video_shape, sensor_size
        )
        # # Fit an eyeball model.
        # inferer.process(dataset, mode="Fit", keyword=keyword)

        # # store the eye model
        # inferer.save_eyeball_model(model_name)

        # load the eyeball model
        inferer.load_eyeball_model(model_name)

        # infer gaze
        inferer.process(
            dataset,
            mode="Infer",
            keyword=keyword,
            output_record_path=output_record_path,
            output_video_path=video_name,
        )
        # except Exception as e:
        #     print(e)


if __name__ == "__main__":
    main()
