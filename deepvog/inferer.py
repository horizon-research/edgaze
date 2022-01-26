import numpy as np
import os
import cv2
import glob
import warnings
import skvideo.io as skv
from skimage.color import rgb2gray
from skimage.transform import resize
from .unprojection import reproject
from .eyefitter import SingleEyeFitter
from .utils import save_json, load_json, convert_vec2angle31
from .visualisation import draw_circle, draw_ellipse, draw_line, VideoManager

class gaze_inferer(object):
    def __init__(self, model, filter_model, flen, ori_video_shape, sensor_size, infer_gaze_flag=True):
        """
        Initialize necessary parameters and load deep_learning model
        
        Args:
            model: Deep learning model that perform image segmentation. Pre-trained model is provided at https://github.com/pydsgz/DeepVOG/model/DeepVOG_model.py, simply by loading load_DeepVOG() with "DeepVOG_weights.h5" in the same directory. If you use your own model, it should take input of grayscale image (m, 240, 320, 3) with value float [0,1] and output (m, 240, 320, 3) with value float [0,1] where (m, 240, 320, 1) is the pupil map.
            
            flen (float): Focal length of camera in mm. You can look it up at the product menu of your camera
            
            ori_video_shape (tuple or list or np.ndarray): Original video shape from your camera, (height, width) in pixel. If you cropped the video before, use the "original" shape but not the cropped shape
            
            sensor_size (tuple or list or np.ndarray): Sensor size of your camera, (height, width) in mm. For 1/3 inch CMOS sensor, it should be (3.6, 4.8). Further reference can be found in https://en.wikipedia.org/wiki/Image_sensor_format and you can look up in your camera product menu
        

        """
        # Assertion of shape
        try:
            assert ((isinstance(flen, int) or isinstance(flen, float)))
            assert (isinstance(ori_video_shape, tuple) or isinstance(ori_video_shape, list) or isinstance(
                ori_video_shape, np.ndarray))
            assert (isinstance(sensor_size, tuple) or isinstance(sensor_size, list) or isinstance(sensor_size,
                                                                                                  np.ndarray))
            assert (isinstance(sensor_size, tuple) or isinstance(sensor_size, list) or isinstance(sensor_size,
                                                                                                  np.ndarray))
        except AssertionError:
            print("At least one of your arguments does not have correct type")
            raise TypeError

        # Parameters dealing with camera and video shape
        self.flen = flen
        self.ori_video_shape, self.sensor_size = np.array(ori_video_shape).squeeze(), np.array(sensor_size).squeeze()
        self.mm2px_scaling = np.linalg.norm(self.ori_video_shape) / np.linalg.norm(self.sensor_size)

        self.model = model
        self.filter_model = filter_model
        self.confidence_fitting_threshold = 0.96
        self.eyefitter = SingleEyeFitter(focal_length=self.flen * self.mm2px_scaling,
                                         pupil_radius=2 * self.mm2px_scaling,
                                         initial_eye_z=50 * self.mm2px_scaling,
                                         image_shape=ori_video_shape)
        self.infer_gaze_flag = infer_gaze_flag

    def load_gt(self, filename, preview=True):

        predict = np.load(filename)

        if np.max(predict) > 0:
            predict = predict / np.max(predict)

        low_pass_filter = predict < 1
        predict[low_pass_filter] = 0

        if preview:
            cv2.imshow("gt", predict)
            cv2.waitKey(30)

        predict = np.expand_dims(predict, axis=0)

        return predict

    def process(self, video_src, mode, keyword="", output_record_path="",
                batch_size=1, output_video_path="", 
                heatmap=False, print_prefix=""):
        """

        Parameters
        ----------
        video_src : str
            Path of the video from which you want to (1) fit the eyeball model or (2) infer the gaze.
        mode : str
            There are two modes: "Fit" or "Infer". "Fit" will fit an eyeball model from the video source.
            "Infer" will infer the gaze from the video source.
        batch_size : int
            Batch size. Recommended >= 32.
        output_record_path : str
            Path of the csv file of your gaze estimation result. Only matter if the mode == "Infer".
        output_video_path : str
            Path of the output visualization video. If mode == "Fit", it draws segmented pupil ellipse.
            If mode == "Infer", it draws segmented pupil ellipse and gaze vector. if output_video_path == "",
            no visualization will be produced.
        heatmap : bool
            If True, show heatmap in the visualization video. If False, no heatmap will be shown.
        print_prefix : str
            What to print before the progress text.

        Returns
        -------
        None

        """
        # Get video information (path strings, frame reader, video's shapes...etc)
    
        vid_h = self.ori_video_shape[0]
        vid_w = self.ori_video_shape[1]

        self.vid_manager = VideoManager(output_record_path=output_record_path,
                                        output_video_path=output_video_path, heatmap=heatmap)

        # (predict) Check if the eyeball model is imported
        if mode == "Infer":
            self._check_eyeball_model_exists()

        image_scaling_factor = 1


        # Correct eyefitter's parameters in accordance with the image resizing
        self.eyefitter.focal_length = self.flen * self.mm2px_scaling * image_scaling_factor
        self.eyefitter.pupil_radius = 2 * self.mm2px_scaling * image_scaling_factor

        framenames = []
        frame_len = len(glob.glob(video_src + "/*.png"))
        for i in range(frame_len):
            framenames.append(video_src + "/%d.png" % i)

        # Set batch-wise operation details
        initial_frame, final_frame = 0, frame_len
        X_batch = np.zeros((1, vid_h, vid_w, 3))

        # Start looping for batch-wise processing
        for idx, framename in enumerate(framenames):
            print("\r%s%s %s (%d/%d)" % (print_prefix, mode, video_src, idx + 1, frame_len), end="\n", flush=True)

            frame = cv2.imread(framename)
            frame_preprocessed = self._preprocess_image(frame, vid_h, vid_w)
            X_batch[0, :, :, :] = frame_preprocessed

            Y_batch = None
            if keyword == "filter":
                Y_batch = self.filter_model.predict(framename)
            elif keyword == "gt":
                print(framename[:-3]+"npy")
                Y_batch = self.load_gt(framename[:-3]+"npy")
            else:
                Y_batch = self.model.predict(framename)

            if mode == "Fit":
                self._fitting_batch(X_batch=X_batch,
                                    Y_batch=Y_batch)
            elif mode == "Infer":
                _, _, _ = self._infer_batch(X_batch=X_batch,
                                            Y_batch=Y_batch,
                                            idx=idx)

        if mode == "Fit":
            # Fit eyeball models. Parameters are stored as internal attributes of Eyefitter instance.
            _ = self.eyefitter.fit_projected_eye_centre(ransac=True, max_iters=100) #, min_distance=10*frame_len)
            _, _ = self.eyefitter.estimate_eye_sphere()

            # Issue error if eyeball model still does not exist after fitting.
            if (self.eyefitter.eye_centre is None) or (self.eyefitter.aver_eye_radius is None):
                raise TypeError("Eyeball model was not fitted. You may need -v or -m argument to check whether the pupil segmentation works properly.")
        print()

    def save_eyeball_model(self, path):
        """
        Save eyeball model parameters in json format.
        
        Args:
            path (str): path of the eyeball model file.
        """

        if (self.eyefitter.eye_centre is None) or (self.eyefitter.aver_eye_radius is None):
            print("3D eyeball model not found. You may need -v or -m argument to check whether the pupil segmentation works properly.")
            raise
        else:
            save_dict = {"eye_centre": self.eyefitter.eye_centre.tolist(),
                         "aver_eye_radius": self.eyefitter.aver_eye_radius}
            save_json(path, save_dict)

    def load_eyeball_model(self, path):
        """
        Load eyeball model parameters of json format from path.
        
        Args:
            path (str): path of the eyeball model file.
        """
        loaded_dict = load_json(path)
        if (self.eyefitter.eye_centre is not None) or (self.eyefitter.aver_eye_radius is not None):
            warnings.warn("3D eyeball exists and reloaded")

        self.eyefitter.eye_centre = np.array(loaded_dict["eye_centre"])
        self.eyefitter.aver_eye_radius = loaded_dict["aver_eye_radius"]


    def _fitting_batch(self, X_batch, Y_batch):

        if self.vid_manager.output_video_flag:
            # Convert video frames to 8 bit integer format for drawing the output video frames
            video_frames_batch = np.around(X_batch * 255).astype(int)
            vid_frame_shape_2d = (video_frames_batch.shape[1], video_frames_batch.shape[2])

        for batch_idx, (X_each, Y_each) in enumerate(zip(X_batch, Y_batch)):
            pred_each = Y_each
            # print(pred_each.shape)
            _, _, _, _, ellipse_info = self.eyefitter.unproject_single_observation(pred_each)
            (rr, cc, centre, w, h, radian, ellipse_confidence) = ellipse_info

            # If visualization is true, initialize output frame
            if self.vid_manager.output_video_flag:
                vid_frame = video_frames_batch[batch_idx,]

            # Fit each observation to eyeball model
            if centre is not None:
                if (ellipse_confidence > self.confidence_fitting_threshold):
                    self.eyefitter.add_to_fitting()

                # Draw ellipse and pupil centre on input video if visualization is enabled
                if self.vid_manager.output_video_flag:
                    ellipse_centre_np = np.array(centre)

                    # Draw pupil ellipse
                    vid_frame = draw_ellipse(output_frame=vid_frame, frame_shape=vid_frame_shape_2d,
                                             ellipse_info=ellipse_info, color=[255, 255, 0])
                    # Draw small circle at the ellipse centre
                    vid_frame = draw_circle(output_frame=vid_frame, frame_shape=vid_frame_shape_2d,
                                            centre=ellipse_centre_np, radius=5, color=[0, 255, 0])

                    self.vid_manager.write_frame_with_condition(vid_frame=vid_frame, pred_each=pred_each)
            else:
                # Draw original input frame when no ellipse is found
                if self.vid_manager.output_video_flag:
                    self.vid_manager.write_frame_with_condition(vid_frame=vid_frame, pred_each=pred_each)


    def _fitting_batch_bkup(self, X_batch, Y_batch):

        if self.vid_manager.output_video_flag:
            # Convert video frames to 8 bit integer format for drawing the output video frames
            video_frames_batch = np.around(X_batch * 255).astype(int)
            vid_frame_shape_2d = (video_frames_batch.shape[1], video_frames_batch.shape[2])

        for batch_idx, (X_each, Y_each) in enumerate(zip(X_batch, Y_batch)):
            pred_each = Y_each[:, :, 1]
            _, _, _, _, ellipse_info = self.eyefitter.unproject_single_observation(pred_each)
            (rr, cc, centre, w, h, radian, ellipse_confidence) = ellipse_info

            # If visualization is true, initialize output frame
            if self.vid_manager.output_video_flag:
                vid_frame = video_frames_batch[batch_idx,]

            # Fit each observation to eyeball model
            if centre is not None:
                if (ellipse_confidence > self.confidence_fitting_threshold):
                    self.eyefitter.add_to_fitting()

                # Draw ellipse and pupil centre on input video if visualization is enabled
                if self.vid_manager.output_video_flag:
                    ellipse_centre_np = np.array(centre)

                    # Draw pupil ellipse
                    vid_frame = draw_ellipse(output_frame=vid_frame, frame_shape=vid_frame_shape_2d,
                                             ellipse_info=ellipse_info, color=[255, 255, 0])
                    # Draw small circle at the ellipse centre
                    vid_frame = draw_circle(output_frame=vid_frame, frame_shape=vid_frame_shape_2d,
                                            centre=ellipse_centre_np, radius=5, color=[0, 255, 0])

                    self.vid_manager.write_frame_with_condition(vid_frame=vid_frame, pred_each=pred_each)
            else:
                # Draw original input frame when no ellipse is found
                if self.vid_manager.output_video_flag:
                    self.vid_manager.write_frame_with_condition(vid_frame=vid_frame, pred_each=pred_each)

    def _infer_batch(self, X_batch, Y_batch, idx):

        if self.vid_manager.output_video_flag:
            # Convert video frames to 8 bit integer format for drawing the output video frames
            video_frames_batch = np.around(X_batch * 255).astype(int)
            vid_frame_shape_2d = (video_frames_batch.shape[1], video_frames_batch.shape[2])

        for batch_idx, (X_each, Y_each) in enumerate(zip(X_batch, Y_batch)):

            frame = idx + batch_idx + 1
            pred_each = Y_each
            _, _, _, _, ellipse_info = self.eyefitter.unproject_single_observation(pred_each)
            (rr, cc, centre, w, h, radian, ellipse_confidence) = ellipse_info
            # If visualization is true, initialize output frame for drawing
            if self.vid_manager.output_video_flag:
                vid_frame = video_frames_batch[batch_idx,]

            # If ellipse fitting is successful, i.e. an ellipse is located, AND gaze inference is ENABLED
            if (centre is not None) and self.infer_gaze_flag:
                p_list, n_list, _, consistence = self.eyefitter.gen_consistent_pupil()
                p1, n1 = p_list[0], n_list[0]
                px, py, pz = p1[0, 0], p1[1, 0], p1[2, 0]
                x, y = convert_vec2angle31(n1)
                positions = (px, py, pz, centre[0], centre[1])  # Pupil 3D positions and 2D projected positions
                gaze_angles = (x, y)  # horizontal and vertical gaze angles
                inference_confidence = (ellipse_confidence, consistence)
                self.vid_manager.write_results(frame_id=frame, pupil2D_x=centre[0], pupil2D_y=centre[1], gaze_x=x,
                                               gaze_y=y, confidence=ellipse_confidence, consistence=consistence)

                if self.vid_manager.output_video_flag:
                    # # Code below is for drawing video
                    ellipse_centre_np = np.array(centre)
                    projected_eye_centre = reproject(self.eyefitter.eye_centre,
                                                     self.eyefitter.focal_length)  # shape (2,1)
                    # The lines below are for translation from camera coordinate system (centred at image centre)
                    # to numpy's indexing frame. You substract the vector by the half of the video's 2D shape.
                    # Col = x-axis, Row = y-axis
                    projected_eye_centre += np.array(vid_frame_shape_2d[::-1]).reshape(-1, 1) / 2

                    vid_frame = self._draw_vis_on_frame(vid_frame, vid_frame_shape_2d, ellipse_info, ellipse_centre_np,
                                                        projected_eye_centre, gaze_vec=n1)

                    self.vid_manager.write_frame_with_condition(vid_frame=vid_frame, pred_each=pred_each)

            # If ellipse fitting is successful, i.e. an ellipse is located, AND gaze inference is DISABLED
            elif (centre is not None) and (not self.infer_gaze_flag):
                positions, gaze_angles, inference_confidence = None, None, None
                self.vid_manager.write_results(frame_id=frame, pupil2D_x=centre[0], pupil2D_y=centre[1], gaze_x=np.nan,
                                               gaze_y=np.nan, confidence=ellipse_confidence, consistence=np.nan)
                if self.vid_manager.output_video_flag:
                    ellipse_centre_np = np.array(centre)

                    # Draw pupil ellipse
                    vid_frame = draw_ellipse(output_frame=vid_frame, frame_shape=vid_frame_shape_2d,
                                             ellipse_info=ellipse_info, color=[255, 255, 0])
                    # Draw small circle at the ellipse centre
                    vid_frame = draw_circle(output_frame=vid_frame, frame_shape=vid_frame_shape_2d,
                                            centre=ellipse_centre_np, radius=5, color=[0, 255, 0])
                    self.vid_manager.write_frame_with_condition(vid_frame=vid_frame, pred_each=pred_each)

            # IF ellipse fitting is unsuccessful.
            else:
                # If ellipse cannot be found, fill the outputs with None's
                positions, gaze_angles, inference_confidence = None, None, None

                self.vid_manager.write_results(frame_id=frame, pupil2D_x=np.nan, pupil2D_y=np.nan, gaze_x=np.nan,
                                               gaze_y=np.nan, confidence=np.nan, consistence=np.nan)

                # Draw original input frame when no ellipse is found
                if self.vid_manager.output_video_flag:
                    self.vid_manager.write_frame_with_condition(vid_frame=vid_frame, pred_each=pred_each)

        return positions, gaze_angles, inference_confidence

    def _check_eyeball_model_exists(self):
        try:
            if self.infer_gaze_flag:
                assert isinstance(self.eyefitter.eye_centre, np.ndarray)
                assert self.eyefitter.eye_centre.shape == (3, 1)
                assert self.eyefitter.aver_eye_radius is not None
            else:
                pass
        except AssertionError as e:
            print(
                "3D eyeball mode is not found. Gaze inference cannot continue. Please fit/load an eyeball model first")
            raise e

    # @staticmethod
    # def _inspectVideoShape(w, h):
    #     if (w, h) == (240, 320):
    #         return True
    #     else:
    #         return False

    @staticmethod
    def _computeCroppedShape(ori_video_shape, crop_size):
        video = np.zeros(ori_video_shape)
        cropped = video[crop_size[0]:crop_size[1], crop_size[2], crop_size[3]]
        return cropped.shape

    @staticmethod
    def _preprocess_image(img, vid_h, vid_w):
        """
        
        Args:
            img (numpy array): unprocessed image with shape (w, h, 3) and values int [0, 255]
        Returns:
            output_img (numpy array): processed grayscale image with shape ( 240, 320, 1) and values float [0,1]
        """
        output_img = np.zeros((vid_h, vid_w, 1))
        img = img / 255
        img = rgb2gray(img)
        img = resize(img, (vid_h, vid_w))
        output_img[:, :, :] = img.reshape(vid_h, vid_w, 1)
        return output_img

    @staticmethod
    def _draw_vis_on_frame(vid_frame, vid_frame_shape_2d, ellipse_info, ellipse_centre_np, projected_eye_centre,
                           gaze_vec):

        # Draw pupil ellipse
        vid_frame = draw_ellipse(output_frame=vid_frame, frame_shape=vid_frame_shape_2d,
                                 ellipse_info=ellipse_info, color=[255, 255, 0])

        # Draw from eyeball centre to ellipse centre (just connecting two points)
        vec_with_length = ellipse_centre_np - projected_eye_centre.squeeze()
        vid_frame = draw_line(output_frame=vid_frame, frame_shape=vid_frame_shape_2d, o=projected_eye_centre,
                              l=vec_with_length, color=[0, 0, 255])

        # Draw gaze vector originated from ellipse centre
        vid_frame = draw_line(output_frame=vid_frame, frame_shape=vid_frame_shape_2d, o=ellipse_centre_np,
                              l=gaze_vec * 50, color=[255, 0, 0])

        # Draw small circle at the ellipse centre
        vid_frame = draw_circle(output_frame=vid_frame, frame_shape=vid_frame_shape_2d,
                                centre=ellipse_centre_np, radius=5, color=[0, 255, 0])
        return vid_frame
