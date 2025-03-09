import torch
from typing import List
from typing import Dict
from typing import Union
import numpy as np
import pandas as pd
from pathlib import Path
import math
import kornia
from kornia.filters import gaussian_blur2d, bilateral_blur
from enum import Enum

from PIL import Image
from PIL.PngImagePlugin import PngInfo
import cv2

from tqdm import tqdm

import traceback

from senpi.constants import *

# CURRENTLY, SMALLEST DIVISION OF TIME IS NANOSECONDS (ANYTHING SMALLER GETS TURNED TO ZEROS)
class ImgFramesFromEventsGenerator:
    """
    A class to generate image frames from event data using the method described in the following paper:
    Cedric Scheerlinck, Nick Barnes, Robert Mahony, "Continuous-time Intensity Estimation Using Event Cameras", Asian Conference on Computer Vision (ACCV), Perth, 2018.
    (corresponding GitHub: https://github.com/cedric-scheerlinck/dvs_image_reconstruction)

    Attributes:
        width (int): Width of the event camera and generated image/video frame(s).
        height (int): Height of the event camera and generated image/video frame(s).
        cutoff_frequency (Union[int, float]): The cutoff frequency for the intensity estimation (the gain/alpha as described in the paper).
        on_contrast_threshold (float): Contrast threshold for ON events.
        off_contrast_threshold (float): Contrast threshold for OFF events.
        most_recent_time_surface (torch.Tensor): Tensor to store the most recent event time for each pixel.
        cur_frame (torch.Tensor): Tensor to store the current frame.
        frame_list (List[torch.Tensor]): List of generated frames.
        timestamp_list (List[float]): List of timestamps for the generated frames.
        min_intensity (float): Minimum intensity for the output frames.
        max_intensity (float): Maximum intensity for the output frames.
    """

    def __init__(self, width: int, height: int, on_contrast_threshold: float=0.2, off_contrast_threshold: float=-0.2, cutoff_frequency: Union[int, float]= 5, \
                 min_intensity: float=-0.5, max_intensity: float=1.5): # default values are not set in stone (event-camera dependent); currently based on paper
        """
        Initialize the ImgFramesFromEventsGenerator with given parameters.

        :param width: Width of the event camera and generated image/video frame(s).
        :type width: int
        :param height: Height of the event camera and generated image/video frame(s).
        :type height: int
        :param on_contrast_threshold: ON (positive) event contrast threshold (based on event camera specifications).
        :type on_contrast_threshold: float
        :param off_contrast_threshold: OFF (negative) event contrast threshold (based on event camera specifications).
        :type off_contrast_threshold: float
        :param cutoff_frequency: Cutoff frequency for the intensity estimation.
        :type cutoff_frequency: Union[int, float]
        :param min_intensity: Minimum intensity for normalization.
        :type min_intensity: float
        :param max_intensity: Maximum intensity for normalization.
        :type max_intensity: float
        """
        self.width = width
        self.height = height
        self.cutoff_frequency = cutoff_frequency
        self.on_contrast_threshold = on_contrast_threshold
        self.off_contrast_threshold = off_contrast_threshold
        self.most_recent_time_surface = torch.zeros((height, width)) # dtype?
        self.cur_frame = torch.zeros((height, width))
        self.frame_list = []
        self.timestamp_list = []
        self.min_intensity = min_intensity
        self.max_intensity = max_intensity

    def reset(self):
        """
        Reset the internal state of the generator, which includes the time surface, current frame tensor, the frame list, and the timestamp list.
        """
        self.most_recent_time_surface = torch.zeros((self.height, self.width)) # dtype -> set to be same as events.dtype in
                                                                               # `generate_frames`
        self.cur_frame = torch.zeros((self.height, self.width), dtype=torch.float32)
        self.frame_list = []
        self.timestamp_list = []

    def generate_frames(self, events: torch.Tensor, gen_frames_mode: str="events", events_per_frame: int=1500, order: List[str]=None, \
                        time_unit: Union[TimeUnit, str]=TimeUnit.MICROSECONDS, enable_gpu: bool=False, accumulate_frame_list: bool=False, \
                        frames_tensor_list_device: torch.device="cpu", spatial_smoothing_method: str="", kernel_size: int=5, \
                        spatial_filter_sigma: float=-1.0, save_images: bool=False, generate_video: bool=False, fps: int=200, \
                        output_parent_dir: Union[Path, str]="", video_file_name: str=None, first_event_start_timestamp: bool=False):
        """
        Generate frames from event data.

        :param events: The input events.
        :type events: torch.Tensor
        :param gen_frames_mode: Mode of frame generation, either "events" or "fps", defaults to "events". Events mode will aggregate `events_per_frame` events per frame and FPS mode will aggregate events within time steps given by 1 / `fps` for each frame.
        :type gen_frames_mode: str, optional
        :param events_per_frame: Number of events per frame, defaults to 1500.
        :type events_per_frame: int, optional
        :param order: Order of columns in the events tensor, defaults to ['b', 't', 'x', 'y', 'p'].
        :type order: List[str], optional
        :param time_unit: Unit of time for events, defaults to TimeUnit.MICROSECONDS.
        :type time_unit: Union[TimeUnit, str], optional
        :param enable_gpu: Whether to use GPU (if available), defaults to False. If True, then `events`, `self.most_recent_time_surface`, and `self.cur_frame` will be moved to the GPU for the duration of the frame generation process and its associated operations. Note that, in some instances setting this parameter to True might actually cause the program to run slower if transferring the data between CPU and GPU imposes a significant enough overhead.
        :type enable_gpu: bool, optional
        :param accumulate_frame_list: Whether to accumulate generated frames in a list (list of torch.Tensor), defaults to False.
        :type accumulate_frame_list: bool, optional
        :param frames_tensor_list_device: Device that each frame in `self.frame_list` should be stored on, defaults to "cpu". 
        :type frames_tensor_list_device: torch.device, optional
        :param spatial_smoothing_method: Method for spatial smoothing, defaults to "". Available smoothing methods are "gaussian" and "bilateral."
        :type spatial_smoothing_method: str, optional
        :param kernel_size: Kernel size for spatial smoothing, defaults to 5.
        :type kernel_size: int, optional
        :param spatial_filter_sigma: Sigma for spatial smoothing, defaults to -1.0, which will prompt its calculation based on `kernel_size`.
        :type spatial_filter_sigma: float, optional
        :param save_images: Whether to save generated images, defaults to False.
        :type save_images: bool, optional
        :param generate_video: Whether to generate video from frames, defaults to False.
        :type generate_video: bool, optional
        :param fps: Frames per second for the generated video, defaults to 200.
        :type fps: int, optional
        :param output_parent_dir: Directory to save the output, defaults to "".
        :type output_parent_dir: Union[Path, str], optional
        :param video_file_name: Name of the output video file, defaults to None.
        :type video_file_name: str, optional
        :param first_event_start_timestamp: Whether the first event's timestamp should be the beginning of the video, defaults to False.
        :type first_event_start_timestamp: bool, optional
        """
        self.reset()

        if gen_frames_mode not in ["events", "fps"]:
            if not isinstance(gen_frames_mode, str):
               raise TypeError("Argument of invalid type provided for time_unit.")
            else:
               raise ValueError(f"Video mode \"{gen_frames_mode}\" is not supported.")
        
        if ((time_unit == TimeUnit.SECONDS.value) or (time_unit.value == TimeUnit.SECONDS.value)):
          time_constant = 1e0
        elif ((time_unit == TimeUnit.MILLISECONDS.value) or (time_unit.value == TimeUnit.MILLISECONDS.value)):
          time_constant = 1e-3
        elif ((time_unit == TimeUnit.MICROSECONDS.value) or (time_unit.value == TimeUnit.MICROSECONDS.value)):
          time_constant = 1e-6
        elif ((time_unit == TimeUnit.NANOSECONDS.value) or (time_unit.value == TimeUnit.NANOSECONDS.value)):
          time_constant = 1e-9
        else:
          if (isinstance(time_unit, TimeUnit) or isinstance(time_unit, str)):
            raise ValueError(f"Time unit \"{time_unit}\" is not supported.")
          else:
            raise TypeError("Argument of invalid type provided for time_unit.")

        if order is None:
            order = ['b', 't', 'x', 'y', 'p']
            print(f"Warning: No order provided. Assuming default order {order}.")

        self.most_recent_time_surface = self.most_recent_time_surface.to(events.dtype) # Converting time surface to same dtype as events

        dev = "cpu"
        if enable_gpu:
            dev = "cuda" if torch.cuda.is_available() else "cpu"

        events = events.to(dev) # Added (if enable_gpu is enabled and cuda is available, events will be moved to gpu)
        self.most_recent_time_surface = self.most_recent_time_surface.to(dev)
        self.cur_frame = self.cur_frame.to(dev)

        data_dir = Path(output_parent_dir) / "gen_data"

        # call gen frames num events or generate frames temporal resolution here
        if gen_frames_mode == "events":
           self.__gen_frames_events(events=events, events_per_frame=events_per_frame, data_dir=data_dir, order=order, time_constant=time_constant, \
                                        spatial_smoothing_method=spatial_smoothing_method, kernel_size=kernel_size, spatial_filter_sigma=spatial_filter_sigma, \
                                        save_images=save_images, accumulate_frame_list=accumulate_frame_list, frames_tensor_list_device=frames_tensor_list_device, \
                                        time_unit=time_unit, gen_video=generate_video, fps=fps, video_file_name=video_file_name)
        else:
           self.__gen_frames_fps(events=events, fps=fps, data_dir=data_dir, order=order, time_constant=time_constant, \
                                        spatial_smoothing_method=spatial_smoothing_method, kernel_size=kernel_size, spatial_filter_sigma=spatial_filter_sigma, \
                                        save_images=save_images, accumulate_frame_list=accumulate_frame_list, frames_tensor_list_device=frames_tensor_list_device, \
                                        time_unit=time_unit, gen_video=generate_video, video_file_name=video_file_name, first_event_start_timestamp=first_event_start_timestamp)

    def __gen_frames_events(self, events: torch.Tensor, events_per_frame: int, data_dir: Path, order: List[str], time_constant: float, \
                               spatial_smoothing_method: str, kernel_size: int, spatial_filter_sigma: float, save_images: bool, \
                               accumulate_frame_list: bool, frames_tensor_list_device: torch.device, time_unit: Union[TimeUnit, str], \
                               gen_video: bool, fps: int, video_file_name: str):
        """
        Generate frames based on the number of events per frame.

        :param events: The input events.
        :type events: torch.Tensor
        :param events_per_frame: Number of events per frame.
        :type events_per_frame: int
        :param data_dir: Directory to save the output.
        :type data_dir: Path
        :param order: Order of columns in the events tensor.
        :type order: List[str]
        :param time_constant: Time constant based on the time unit.
        :type time_constant: float
        :param spatial_smoothing_method: Method for spatial smoothing.
        :type spatial_smoothing_method: str
        :param kernel_size: Kernel size for spatial smoothing.
        :type kernel_size: int
        :param spatial_filter_sigma: Sigma for spatial smoothing.
        :type spatial_filter_sigma: float
        :param save_images: Whether to save generated images.
        :type save_images: bool
        :param accumulate_frame_list: Whether to accumulate generated frames in a list.
        :type accumulate_frame_list: bool
        :param frames_tensor_list_device: Device for storing accumulated frames.
        :type frames_tensor_list_device: torch.device
        :param time_unit: Unit of time for events.
        :type time_unit: Union[TimeUnit, str]
        :param gen_video: Whether to generate video from frames.
        :type gen_video: bool
        :param fps: Frames per second for the generated video.
        :type fps: int
        :param video_file_name: Name of the output video file.
        :type video_file_name: str
        """
        param_ind = {order[ind] : ind for ind in range(len(order))}
        thresholds = [self.off_contrast_threshold, self.on_contrast_threshold]

        frame_num = 0
        NUM_EVENTS = len(events)
        LAST_EVENT_IDX = NUM_EVENTS - 1

        if gen_video:
          data_dir = Path(data_dir)
          data_dir.mkdir(parents=True, exist_ok=True)
          if (video_file_name is None) or (not isinstance(video_file_name, str)):
            video_file_name = "video.mp4"
          if not video_file_name.endswith(".mp4"):
            video_file_name += ".mp4"
          video_path = data_dir / video_file_name
          fourcc = cv2.VideoWriter_fourcc(*'mp4v')
          video = cv2.VideoWriter(str(video_path), fourcc, fps, (self.width, self.height), isColor=False)
        
        print(f"Total number of iterations: {NUM_EVENTS}.")

        for i in tqdm((i for i in range(NUM_EVENTS))): # generator comprehension
            cur_t = events[i, param_ind['t']].item()
            cur_x = int(events[i, param_ind['x']])
            cur_y = int(events[i, param_ind['y']])
            cur_p = int(events[i, param_ind['p']])

            dt = cur_t - self.most_recent_time_surface[cur_y, cur_x]

            if dt < 0:
                print("Non-increasing time sequence detected. Aborting frame generation and initiating reset.")
                self.reset()
                return

            # converting dt to seconds (regardless of initial units)
            self.cur_frame[cur_y, cur_x] *= torch.exp(-self.cutoff_frequency * dt * time_constant)

            p_index = int(cur_p * ((0.5 * cur_p) + 0.5))
            self.cur_frame[cur_y, cur_x] += thresholds[p_index]
            self.most_recent_time_surface[cur_y, cur_x] = cur_t

            if (i % events_per_frame == 0) or (i == LAST_EVENT_IDX):
                dt_matrix = cur_t - self.most_recent_time_surface
                self.cur_frame *= torch.exp(-self.cutoff_frequency * dt_matrix * time_constant)
                self.most_recent_time_surface.fill_(cur_t)
                ret_frame = self.convert_log_frame_to_intensity_img(log_frame=self.cur_frame, spatial_smoothing_method=spatial_smoothing_method, kernel_size=kernel_size, \
                                                                    spatial_filter_sigma=spatial_filter_sigma)
                cur_timestamp = float(cur_t * time_constant)

                if accumulate_frame_list:
                   ret_frame = ret_frame.to(frames_tensor_list_device)
                   self.frame_list.append(ret_frame)
                   self.timestamp_list.append(cur_timestamp)

                if save_images:
                    imgs_sub_dir = Path("images")
                    imgs_final_dir = data_dir / imgs_sub_dir
                    imgs_final_dir.mkdir(parents=True, exist_ok=True)

                    img_final_path = imgs_final_dir / f"frame_{frame_num}.png"
                    ret_fr_np = ret_frame.detach().cpu().numpy()
                    img = Image.fromarray(ret_fr_np, mode='L')

                    metadata = PngInfo()
                    metadata.add_text("timestamp", f"{cur_timestamp: .9f}")

                    # adding timestamp to image metadata (in seconds)
                    # can use exiftool to view image metadata locally
                    img.save(img_final_path, pnginfo=metadata)

                if gen_video:
                  #  frame_np = ret_frame.detach().cpu().numpy()
                    if save_images:
                       video.write(ret_fr_np)
                    else:
                      video.write(ret_frame.detach().cpu().numpy())

                frame_num += 1

        if gen_video:
           video.release()
           print(f"Video saved at {video_path}")


    def __gen_frames_fps(self, events: torch.Tensor, fps: int, data_dir: Path, order: List[str], time_constant: float, \
                               spatial_smoothing_method: str, kernel_size: int, spatial_filter_sigma: float, save_images: bool, \
                               accumulate_frame_list: bool, frames_tensor_list_device: torch.device, time_unit: Union[TimeUnit, str], \
                               gen_video: bool, video_file_name: str, first_event_start_timestamp: bool):
        """
        Generate frames based on frames per second (FPS).

        :param events: The input events.
        :type events: torch.Tensor
        :param fps: Frames per second for the generated video.
        :type fps: int
        :param data_dir: Directory to save the output.
        :type data_dir: Path
        :param order: Order of columns in the events tensor.
        :type order: List[str]
        :param time_constant: Time constant based on the time unit.
        :type time_constant: float
        :param spatial_smoothing_method: Method for spatial smoothing.
        :type spatial_smoothing_method: str
        :param kernel_size: Kernel size for spatial smoothing.
        :type kernel_size: int
        :param spatial_filter_sigma: Sigma for spatial smoothing.
        :type spatial_filter_sigma: float
        :param save_images: Whether to save generated images.
        :type save_images: bool
        :param accumulate_frame_list: Whether to accumulate generated frames in a list.
        :type accumulate_frame_list: bool
        :param frames_tensor_list_device: Device for storing accumulated frames.
        :type frames_tensor_list_device: torch.device
        :param time_unit: Unit of time for events.
        :type time_unit: Union[TimeUnit, str]
        :param gen_video: Whether to generate video from frames.
        :type gen_video: bool
        :param video_file_name: Name of the output video file.
        :type video_file_name: str
        """
        # SHOULD BE REAL-TIME
        # CURRENTLY ONLY SUPPORTS NANOSECOND RESOLUTION --> LATER MAKE IT SO THAT THE RESOLUTION ISN'T STATIC
        if ((time_unit == TimeUnit.SECONDS.value) or (time_unit.value == TimeUnit.SECONDS.value)):
          nano_to_same_unit = 1e-9
        elif ((time_unit == TimeUnit.MILLISECONDS.value) or (time_unit.value == TimeUnit.MILLISECONDS.value)):
          nano_to_same_unit = 1e-6
        elif ((time_unit == TimeUnit.MICROSECONDS.value) or (time_unit.value == TimeUnit.MICROSECONDS.value)):
          nano_to_same_unit = 1e-3
        elif ((time_unit == TimeUnit.NANOSECONDS.value) or (time_unit.value == TimeUnit.NANOSECONDS.value)):
          nano_to_same_unit = 1e0

        param_ind = {order[ind] : ind for ind in range(len(order))}
        thresholds = [self.off_contrast_threshold, self.on_contrast_threshold]

        # temporal resolution
        temp_res_seconds = 1 / fps # seconds per frame (time step)
        frame_num = 0

        event_ptr = 0 # index of event that has its timestamp greater than the current time
        nanosecond_time_constant = int(1e9 * time_constant)
        # print(f"Nanosecond time constant: {nanosecond_time_constant}")

        OFFSET_TIME = -events[0, param_ind['t']].item() if first_event_start_timestamp else 0
        OFFSET_TIME_NANO = int(OFFSET_TIME * nanosecond_time_constant)
        LAST_EVENT_TIME = int(events[-1, param_ind['t']].item() * nanosecond_time_constant) + OFFSET_TIME_NANO
        TEMP_RES = int(temp_res_seconds * 1e9)
        LAST_VID_TIME_PLUS_ONE = LAST_EVENT_TIME + TEMP_RES - (LAST_EVENT_TIME % TEMP_RES) + 1 # can  use divmod rather than operations (%, //) for large numbers??

        NUM_EVENTS = len(events)
        LAST_EVENT_IDX = NUM_EVENTS - 1

        if TEMP_RES == 0:
           print("Temporal resolution (1 / FPS) is too small. Aborting frame generation.")
           self.reset()
           return

        if gen_video:
          data_dir = Path(data_dir)
          data_dir.mkdir(parents=True, exist_ok=True)
          if (video_file_name is None) or (not isinstance(video_file_name, str)):
            video_file_name = "video.mp4"
          if not video_file_name.endswith(".mp4"):
            video_file_name += ".mp4"
          video_path = data_dir / video_file_name
          fourcc = cv2.VideoWriter_fourcc(*'mp4v')
          # video = cv2.VideoWriter(str(video_path), fourcc, fps, (self.width, self.height), isColor=True)
          video = cv2.VideoWriter(str(video_path), fourcc, fps, (self.width, self.height), isColor=False)

        print(f"Total number of iterations: {(LAST_VID_TIME_PLUS_ONE // TEMP_RES) + (0 if (LAST_VID_TIME_PLUS_ONE % TEMP_RES == 0) else 1)}.")

        # last_time = -1
        for cur_time in tqdm((cur_time for cur_time in range(0, LAST_VID_TIME_PLUS_ONE, TEMP_RES))):
            if (event_ptr > LAST_EVENT_IDX):
               break
            # Debugging
            # print(f"Current time (nanoseconds): {cur_time}")
            # print(f"Current time (same unit): {cur_time * nano_to_same_unit}")
            # print("-"*20)
            # print("Inside inner loop:")
            while ((event_ptr < NUM_EVENTS) and (int((events[event_ptr, param_ind['t']].item() + OFFSET_TIME) * nanosecond_time_constant) <= cur_time)):
                cur_t = events[event_ptr, param_ind['t']].item() + OFFSET_TIME
                # print(f"Current time (event): {cur_t}")
                cur_x = int(events[event_ptr, param_ind['x']])
                cur_y = int(events[event_ptr, param_ind['y']])
                cur_p = int(events[event_ptr, param_ind['p']])

                dt = cur_t - self.most_recent_time_surface[cur_y, cur_x]

                if dt < 0:
                    # print("DECREASING SEQUENCE")
                    # print(f"Current time (nanoseconds): {cur_time}")
                    # print(f"Current time (same unit): {cur_time * nano_to_same_unit}")
                    # print(f"dt: {dt}")
                    # print(f"{cur_t}, {self.most_recent_time_surface[cur_y, cur_x]}, {cur_y}, {cur_x}")
                    # print(f"Event pointer: {event_ptr}")
                    print("Non-increasing time sequence detected. Aborting frame generation and initiating reset.")
                    self.reset()
                    return

                # keeping time difference in seconds
                self.cur_frame[cur_y, cur_x] *= torch.exp(-self.cutoff_frequency * dt * time_constant)

                p_index = int(cur_p * ((0.5 * cur_p) + 0.5))
                self.cur_frame[cur_y, cur_x] += thresholds[p_index]
                self.most_recent_time_surface[cur_y, cur_x] = cur_t

                # last_time = cur_t
                event_ptr += 1
            # print(f"Last time less than {cur_time * nano_to_same_unit}: {last_time}")
            # print(f"Time after last time (same time unit/microseconds): {int(events[event_ptr, param_ind['t']].item())}")
            # print(f"Time after last time (nanoseconds): {int(events[event_ptr, param_ind['t']].item() * nanosecond_time_constant)}")
            # print(f"Current time (nanoseconds): {cur_time}")
            # cur_time_same_unit = (cur_time * 1e-9 * (1 / time_constant)) # float arithmetic and precision is suboptimal here
            cur_time_same_unit = cur_time * nano_to_same_unit
            dt_matrix = cur_time_same_unit - self.most_recent_time_surface
            self.cur_frame *= torch.exp(-self.cutoff_frequency * dt_matrix * time_constant)

            # Debugging
            # print(f"Current time same unit: {cur_time_same_unit}")

            self.most_recent_time_surface.fill_(cur_time_same_unit)
            ret_frame = self.convert_log_frame_to_intensity_img(log_frame=self.cur_frame, spatial_smoothing_method=spatial_smoothing_method, kernel_size=kernel_size, \
                                                                spatial_filter_sigma=spatial_filter_sigma)
            cur_timestamp = float(cur_time * 1e-9)

            if accumulate_frame_list:
                ret_frame = ret_frame.to(frames_tensor_list_device)
                self.frame_list.append(ret_frame)
                self.timestamp_list.append(cur_timestamp) # in seconds

            if save_images:
                imgs_sub_dir = Path("images")
                imgs_final_dir = data_dir / imgs_sub_dir
                imgs_final_dir.mkdir(parents=True, exist_ok=True)

                img_final_path = imgs_final_dir / f"frame_{frame_num}.png"
                img = Image.fromarray(ret_frame.detach().cpu().numpy(), mode='L')

                metadata = PngInfo()
                metadata.add_text("timestamp", f"{cur_timestamp: .9f}")

                # adding timestamp to image metadata (in seconds)
                # can use exiftool to view image metadata locally
                img.save(img_final_path, pnginfo=metadata)

            if gen_video:
              # color_frame = cv2.applyColorMap(ret_frame.cpu().numpy(), colormap=cv2.COLORMAP_TWILIGHT_SHIFTED)
              # video.write(color_frame)
              video.write(ret_frame.detach().cpu().numpy()) # repeated operation with `save_images` ?

            frame_num += 1
            # print("-" * 20)

        if gen_video:
           video.release()
           print(f"Video saved at {video_path}")

    def convert_log_frame_to_intensity_img(self, log_frame: torch.Tensor, spatial_smoothing_method: str="", \
                                           kernel_size: int=5, spatial_filter_sigma: float=-1.0) -> torch.Tensor:
        """
        Convert a log intensity frame to an intensity image.

        This function applies exponential transformation to the log intensity frame,
        scales it to the intensity range, and optionally applies spatial smoothing.

        :param log_frame: The log intensity frame.
        :type log_frame: torch.Tensor
        :param spatial_smoothing_method: Method for spatial smoothing, either "gaussian" or "bilateral".
        :type spatial_smoothing_method: str
        :param kernel_size: Size of the kernel for spatial smoothing.
        :type kernel_size: int
        :param spatial_filter_sigma: Sigma value for spatial smoothing. If -1, it will be calculated based on the kernel size.
        :type spatial_filter_sigma: float
        :return: The intensity image.
        :rtype: torch.Tensor
        """
        LOG_INTENSITY_OFFSET = math.log(1.5)
        ret_frame = log_frame + LOG_INTENSITY_OFFSET # + -> not a direct reference / "deepcopy"; out-of-place
        ret_frame = torch.exp(ret_frame) # also an out-of-place operation
        ret_frame -= 1

        # the above is probably right (yielded better results); no intuition however
        # ret_frame = torch.exp(log_frame) # checked this (simply exp(log_frame)); appears to be incorrect intensity

        intensity_range = self.max_intensity - self.min_intensity
        ret_frame -= self.min_intensity
        ret_frame = (ret_frame / intensity_range) * 255.0

        if spatial_smoothing_method == "gaussian":
            if spatial_filter_sigma < 0:
                spatial_filter_sigma = (0.3 * ((kernel_size-1)*0.5 - 1)) + 0.8
            ret_frame = gaussian_blur2d(input=ret_frame.unsqueeze(0).unsqueeze(0), kernel_size=kernel_size, sigma=(spatial_filter_sigma, spatial_filter_sigma))
            ret_frame = ret_frame.squeeze()
        elif spatial_smoothing_method == "bilateral":
            if spatial_filter_sigma < 0:
                spatial_filter_sigma = (0.3 * ((kernel_size-1)*0.5 - 1)) + 0.8 # NOT OFFICIAL (NEED TO CHECK THIS):
                filter_sigma = spatial_filter_sigma * (kernel_size * kernel_size) # bilateral_sigma = spatial_filter_sigma_*25 in referenced/OG code
            ret_frame = bilateral_blur(input=ret_frame.unsqueeze(0).unsqueeze(0), kernel_size=kernel_size, sigma_color=filter_sigma, sigma_space=(filter_sigma, filter_sigma))
            ret_frame = ret_frame.squeeze()

        ret_frame = ret_frame.to(torch.uint8)
        return ret_frame

    def create_video_from_frame_list(self, data_dir: Union[Path, str], fps: int, video_file_name: str=None):
        """
        Create a video from the accumulated frames in the frame list.

        This function saves the frames as a video file in the specified directory with the given frame rate.

        :param data_dir: Directory to save the video file.
        :type data_dir: Union[Path, str]
        :param fps: Frames per second for the video.
        :type fps: int
        :param video_file_name: Name of the generated video file. If None, defaults to "video.mp4".
        :type video_file_name: str, optional
        """
        if len(self.frame_list) == 0:
            print("No frames to generate video.")
            return

        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        if (video_file_name is None) or (not isinstance(video_file_name, str)):
            video_file_name = "video.mp4"
        if not video_file_name.endswith(".mp4"):
            video_file_name += ".mp4"
        video_path = data_dir / video_file_name

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # Might need to add higher degree of configurability for FPS
        video = cv2.VideoWriter(str(video_path), fourcc, fps, (self.width, self.height), isColor=False)

        print("Generating video...")

        for frame in tqdm(self.frame_list):
            # frame_np = frame.detach().cpu().numpy().astype(np.uint8)
            # video.write(cv2.cvtColor(frame_np, cv2.COLOR_GRAY2BGR)) (grayscale -> RGB frames???)
            video.write(frame.detach().cpu().numpy())

        video.release()
        print(f"Video saved at {video_path}")

# Notes:
# Can follow Cederic's Jupyter notebook (asw as actual GitHub representation, of course)
# Can use Kornia to convert to different representations (image.convertTo(image_out, CV_8UC1, 255/intensity_range))?

# X. Lagorce, G. Orchard, F. Galluppi, B. E. Shi, and R. B. Benosman,
# “HOTS: A hierarchy of event-based time-surfaces for pattern recognition,”
# IEEE Trans. Pattern Anal. Mach. Intell., vol. 39, no. 7, pp. 1346–1359,
# Jul. 2017