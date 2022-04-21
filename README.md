# Real Time Gaze Tracking with Event Driven Eye Segmentation

## What's in it

In the main directory, this repo contains the methods that we propose in our IEEE VR 2022 paper [Real Time Gaze Tracking with Event Driven Eye Segmentation](https://www.cs.rochester.edu/horizon/pubs/vr22.pdf). We propose a very lightweight method to predict region-of-interests (ROIs) for continuous eye tracking and reduce the average eye tracking execution time up to 5 times. The To run our code, please follow the instructions below.

For more details about how to train the eye segmentation networks and ROI prediction network, please checkout the links below:
* [Efficient Eye segmentation](https://github.com/horizon-research/edgaze/tree/main/train_eye_net) go to directory `train_eye_net`
* [ROI prediction network](https://github.com/horizon-research/edgaze/tree/main/train_roi_prediction): go to directory `train_roi_prediction`

## Prerequisites

To run our framework, it is better to have a Python distribution (we recommend [Anaconda](https://www.anaconda.com/)) and the following Python packages with Python3.7:

```
Python3.7
numpy
scikit-video              : conda install -c conda-forge sk-video 
scikit-image              : conda install scikit-image
Pillow
cv2                       : pip install opencv-python opencv-contrib-python 
Pytorch                   : conda install pytorch torchvision torchaudio cpuonly -c pytorch
keras
matplotlib                : conda install -c conda-forge matplotlib
```

One of the evaluation datasets we use is the Facebook OpenEDS 2020 dataset [openEDS](https://research.fb.com/programs/openeds-2020-challenge/) (Track-2). The dataset has sequential eye sequences, but do not have the ground truth. We generate the ground truth labels, and package them with the dataset. You can download the dataset at this [Box folder](https://rochester.box.com/s/vwiiv4ahe6hrftf8lbulpdwxefngudeo) or simply use this [link](https://rochester.box.com/s/y6ryd043x3y1kvsnwlkhssoo42je4eem). See the paper for how we generate the ground truth labels.
* [Image Data](https://rochester.box.com/s/y6ryd043x3y1kvsnwlkhssoo42je4eem): which contains the sequential image data and ground truth.
* [Bounding Box](https://rochester.box.com/s/a2cfyyg2gc9v1d0bevqxfxfm6cd7ipvx): which contains the bounding box to train the bounding box predictor.
* [Events](https://rochester.box.com/s/vbu9f40yu1h580zhp811j9fx38luw2ee): whcih contains the emulated event maps.

## Usage

The easiest way to run our code is to via `main.py`:
```
 $ python main.py --help

usage: main.py [-h] [--dataset_path DATASET_PATH] [--sequence SEQUENCE]
               [--all_sequence] [--preview] [--model MODEL]
               [--pytorch_model_path PYTORCH_MODEL_PATH] [--disable_edge_info]
               [--use_sift] [--bbox_model_path BBOX_MODEL_PATH]
               [--device DEVICE] [--video_shape VIDEO_SHAPE VIDEO_SHAPE]
               [--sensor_size SENSOR_SIZE SENSOR_SIZE]
               [--focal_length FOCAL_LENGTH] [--mode MODE]
               [--scaledown SCALEDOWN] [--blur_size BLUR_SIZE]
               [--clip_val CLIP_VAL] [--threshold_ratio THRESHOLD_RATIO]
               [--density_threshold DENSITY_THRESHOLD]
               [--output_path OUTPUT_PATH] [--suffix SUFFIX]

optional arguments:
  -h, --help            show this help message and exit
  --dataset_path DATASET_PATH
                        path to openEDS dataset, default: openEDS
  --sequence SEQUENCE   The sequence number of the dataset
  --all_sequence        Evaluate all sequences in dataset
  --preview             preview the result, default: False
  --model MODEL         dnn model, support eye_net, eye_net_m, pruned_eye_net
  --pytorch_model_path PYTORCH_MODEL_PATH
                        pytorch model path
  --disable_edge_info   disable using edge information in bbox prediction
  --use_sift            use sift descriptor to extrapolate bbox
  --bbox_model_path BBOX_MODEL_PATH
                        bbox model path
  --device DEVICE       device to run on, 'cpu' or 'cuda', only apply to
                        pytorch, default: CPU
  --video_shape VIDEO_SHAPE VIDEO_SHAPE
                        video_shape, [HEIGHT, WIDTH] default: [400, 640]
  --sensor_size SENSOR_SIZE SENSOR_SIZE
                        sensor_shape, [HEIGHT, WIDTH] default: [3.6, 4.8]
  --focal_length FOCAL_LENGTH
                        camera focal length, default: 6
  --mode MODE           processing mode, org: use baseline [default], filter:
                        use smart camera filter
  --scaledown SCALEDOWN
                        scaledown when tracking bbox to reduce computation,
                        default: 2
  --blur_size BLUR_SIZE
                        blur the input image when tracking bbox
  --clip_val CLIP_VAL   clip value in event emulator, clip all low pixel value
                        to this number, default: 10
  --threshold_ratio THRESHOLD_RATIO
                        threshold ratio to activate an event, default: 0.3
  --density_threshold DENSITY_THRESHOLD
                        threshold ratio to warp result, default: 0.05
  --output_path OUTPUT_PATH
                        save folder name
  --suffix SUFFIX       save folder name
```

In this repo, we support different eye segmentation networks, here is the supported networks with their corresponding model weights. All weights reside in directory `model_weights`:
| Model Name     | Name in code   | Weights Name             |
| -------------- |--------------- | ------------------------ |
| eye_net        | eye_net        | eye_net.pkl              |
| Ours(L)        | eye_net_m      | eye_net_m.pkl            |
| Ours(S)        | pruned_eye_net | pruned_eye_net.pkl       |

Please use the correct weight with the eye segmentation network. When choosing networks, using the `Name in code` to define the eye segmentation model.

There are several main options:
 1. `pytorch_model_path`/`tensorflow_model_path`: currently, we have already provide pre-trained pytorch/tensorflow model, but you can use your own model too.
 2. `sequence`: in OpenEDS dataset, it contains 20 demo sequences in the directory, you can use this flag to choose one sequence to run.
 3. `all_sequence`: use this flag to run all sequences of data.
 4. `mode`: we provide two modes. `org` is the reference baseline without ROI prediction and processing, `filter` enables ROI-based prediction and inference.
 5. `device`: define the hardware to run, support two options: cpu and cuda, default is cpu.
 6. `density_threshold`: define the event density threshold for extrapolation. When the event density is lower than this threhsold, it will perform extrapolation. 
 7. `preview`: this flag can preview some of results.
 8. `output_path`: this will output the result to a defined directory.

For example, run one sequence of data,
```
 $ 	python3 main.py --sequence 1 \
			--mode filter \
			--model eye_net_m \
			--density_threshold 0.01 \
			--dataset openEDS \
			--pytorch_model_path model_weights/eye_net_m.pkl \
			--device cpu \
			--bbox_model_path model_weights/G030_c32_best.pth \
			--preview
```

This command will run the sequence data `S_1` from directory `openEDS` with model `eye_net_m`. The `--bbox_model_path` defines the ROI prediction network weights. The `--mode filter` shows that we use ROI prediction and extrapolation in this case. 

To run one sequence with full-frame image with `eye_net_m`,
```
 $ 	python3 main.py --sequence 1 \
			--mode org \
			--model eye_net_m \
			--dataset openEDS \
			--pytorch_model_path model_weights/eye_net_m.pkl \
			--device cpu \
			--preview
```
