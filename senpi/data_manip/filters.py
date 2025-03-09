import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import List, Union
from tqdm import tqdm
from senpi.data_io.basic_utils import get_np_dtype_from_torch_dtype
from senpi.data_io.basic_utils import get_torch_dtype_from_np_dtype
import inspect
import matplotlib.pyplot as plt
# from senpi.data_manip.computation import __auto_str

class Filter:
    """
    A base class for filtering event data.

    This class should be subclassed to implement specific filtering methods for event data.
    """

    def __call__(self, events: Union[pd.DataFrame, np.ndarray, torch.Tensor], **kwargs) -> Union[pd.DataFrame, np.ndarray, torch.Tensor]:
        """
        Apply the filter to the event data.

        :param events: The event data to filter.
        :type events: Union[pd.DataFrame, np.ndarray, torch.Tensor]
        :param kwargs: Additional parameters for the filter method.
        :return: The filtered event data.
        :rtype: Union[pd.DataFrame, np.ndarray, torch.Tensor]
        """
        return self.filter_events(events=events, **kwargs)
    
    def __str__(self):
        """
        Return a string representation of the Filter instance.

        :return: A string representation of the instance's attributes.
        :rtype: str
        """
        attrs = vars(self)
        attrs_str = ', '.join(f"{key}={value}" for key, value in attrs.items())
        return f"{self.__class__.__name__}({attrs_str})"

    def filter_events(self, events: Union[pd.DataFrame, np.ndarray, torch.Tensor], **kwargs) -> Union[pd.DataFrame, np.ndarray, torch.Tensor]:
        """
        Filter the event data.

        This method should be implemented by subclasses to define the specific filtering behavior.

        :param events: The event data to filter.
        :type events: Union[pd.DataFrame, np.ndarray, torch.Tensor]
        :param kwargs: Additional parameters for the filter method.
        :return: The filtered event data.
        :rtype: Union[pd.DataFrame, np.ndarray, torch.Tensor]
        :raises NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def filter_frames_tensor(self, frames: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Filter the frames data.

        This method should be implemented by subclasses to define the specific filtering behavior for frame volumes.

        :param frames: The frame data to filter.
        :type frames: torch.Tensor
        :param kwargs: Additional parameters for the filter method.
        :return: The filtered frame volume data.
        :rtype: torch.Tensor
        :raises NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError("Subclasses must implement this method")

class PolarityFilter(Filter):
    """
    A class to filter event-based vision data by event polarity.
    """
    def __init__(self, filter_type: bool):
        """
        Initialize the polarity filter.

        :param filter_type: The type of events to filter (True for positive events, False for negative events).
        :type filter_type: bool
        """
        self.filter_type = filter_type  # True -> get only ON polarity events, False -> get only OFF polarity events
    
    def __call__(self, events: Union[pd.DataFrame, np.ndarray, torch.Tensor], order: List[str]=None, binary_polarity_mode: bool=False, \
                    reset_indices: bool=True) -> Union[pd.DataFrame, np.ndarray, torch.Tensor]:
        return self.filter_events(events=events, order=order, binary_polarity_mode=binary_polarity_mode, reset_indices=reset_indices)

    def filter_events(self, events: Union[pd.DataFrame, np.ndarray, torch.Tensor], order: List[str]=None, binary_polarity_mode: bool=False, \
               reset_indices: bool=True) -> Union[pd.DataFrame, np.ndarray, torch.Tensor]:
        """
        Filter the inputted events by the polarity given by `self.filter_type`.

        :param events: The events to filter.
        :type events: Union[pd.DataFrame, np.ndarray, torch.Tensor]
        :param order: The order of the columns of an inputted events torch.Tensor. By default, this is set to None and further handled by helper methods.
        :type order: List[str], optional
        :param binary_polarity_mode: Whether the polarities are represented by 0s and 1s (True) or -1s and 1s (False).
        :type binary_polarity_mode: bool, optional
        :param reset_indices: Whether to reset the indices of the returned Pandas dataframe (if a Pandas dataframe is inputted) after filtering it.
        :type reset_indices: bool, optional
        :return: The filtered events.
        :rtype: Union[pd.DataFrame, np.ndarray, torch.Tensor]
        """
        
        if isinstance(events, torch.Tensor):
            return self.filter_events_tensor(events, order=order, binary_polarity_mode=binary_polarity_mode)
        elif isinstance(events, pd.DataFrame):
            return self.filter_events_pd(events, binary_polarity_mode=binary_polarity_mode, reset_indices=reset_indices)
        else:
            return self.filter_events_np(events, binary_polarity_mode=binary_polarity_mode)
    
    def filter_events_pd(self, events: pd.DataFrame, binary_polarity_mode: bool=False, reset_indices: bool=True) -> pd.DataFrame:
        """
        Filter a Pandas DataFrame of events by polarity.

        :param events: The events to filter.
        :type events: pd.DataFrame
        :param binary_polarity_mode: Whether the polarities are represented by 0s and 1s (True) or -1s and 1s (False).
        :type binary_polarity_mode: bool, optional
        :param reset_indices: Whether to reset the indices of the returned DataFrame after filtering.
        :type reset_indices: bool, optional
        :return: The filtered events.
        :rtype: pd.DataFrame
        """
        polarity_entries = 1 if self.filter_type else (0 if binary_polarity_mode else -1)
        if reset_indices:
            return events[events['p'] == polarity_entries].reset_index(drop=True)
        return events[events['p'] == polarity_entries]
    
    def filter_events_np(self, events: np.ndarray, binary_polarity_mode: bool=False) -> np.ndarray:
        """
        Filter a numpy array of events by polarity.

        :param events: The events to filter.
        :type events: np.ndarray
        :param binary_polarity_mode: Whether the polarities are represented by 0s and 1s (True) or -1s and 1s (False).
        :type binary_polarity_mode: bool, optional
        :return: The filtered events.
        :rtype: np.ndarray
        """
        polarity_entries = 1 if self.filter_type else (0 if binary_polarity_mode else -1)
        return events[events['p'] == polarity_entries]
    
    def filter_events_tensor(self, events: torch.Tensor, order: List[str]=None, binary_polarity_mode: bool=False) -> torch.Tensor:
        """
        Filter a torch.Tensor of events by polarity.

        :param events: The events to filter.
        :type events: torch.Tensor
        :param order: The order of the columns of the inputted events tensor. Defaults to ['b', 't', 'x', 'y', 'p'].
        :type order: List[str], optional
        :param binary_polarity_mode: Whether the polarities are represented by 0s and 1s (True) or -1s and 1s (False).
        :type binary_polarity_mode: bool, optional
        :return: The filtered events.
        :rtype: torch.Tensor
        """
        if order is None:
            order = ['b', 't', 'x', 'y', 'p']
            print(f"Warning: No order provided. Assuming default order {order}.")
        polarity_entries = 1 if self.filter_type else (0 if binary_polarity_mode else -1)
        param_ind = {order[ind]: ind for ind in range(len(order))}
        return events[(events[:, param_ind['p']] == polarity_entries).nonzero().squeeze()]
    
    def filter_frames_tensor(self, frames: torch.Tensor, inplace: bool) -> torch.Tensor:
        """
        Filter a torch.Tensor frame volume of events by polarity.
        
        :param frames: The frames to filter. Expected format: (N, C, X, Y), where N is the number of frames (each frame displays an accumulation of events for every delta t window) and C is the number of channels (should be 1).
        :type frames: torch.Tensor
        :param inplace: Whether to apply the filter in place or out of place.
        :type inplace: bool
        :return: The filtered events.
        :rtype: torch.Tensor
        """
        if inplace:
            if self.filter_type:
                frames.clamp_(min = 0, max = 1)
            else:
                frames.clamp_(min = -1, max = 0)
            frames *= -1
            return frames
        
        if self.filter_type:
            return torch.clamp(frames, min = 0, max = 1)
        return torch.clamp(frames, min = -1, max = 0)

    def set_filter_type(self, filter_type: bool):
        """
        Set the type of polarity to filter.

        :param filter_type: The type of events to filter (True for positive events, False for negative events).
        :type filter_type: bool
        """
        self.filter_type = filter_type

class BAFilter(Filter):
    """
    A class to filter event-based vision data using a background activity (BA) filter based on temporal and spatial constraints.
    E-MLB: https://ieeexplore.ieee.org/document/10078400
    Based on the following paper(s) (as referenced in E-MLB): 
    
    Paper 1: 
    https://www.scirp.org/reference/referencespapers?referenceid=1176812
    PDF: https://web.archive.org/web/20170809001804/http://www.ini.uzh.ch/~tobi/wiki/lib/exe/fetch.php?media=delbruckeventbasedvision_gcoe_symp_2008.pdf
    T. Delbruck, “Frame-free dynamic digital vision,” in Proc. Intl. Symp. Secure-Life Electron., Adv. Electron. Qual. Life Soc., 2008, vol. 1, pp. 21-26.

    Paper 2:
    D. Czech and G. Orchard, "Evaluating noise filtering for event-based asynchronous change detection image sensors," 
    2016 6th IEEE International Conference on Biomedical Robotics and Biomechatronics (BioRob), Singapore, 2016, pp. 19-24, doi: 10.1109/BIOROB.2016.7523452.

    :param time_threshold: The time threshold for filtering events.
    :type time_threshold: Union[int, float]
    :param height: The height of the event frame.
    :type height: int
    :param width: The width of the event frame.
    :type width: int
    """
    def __init__(self, time_threshold: Union[int, float], height: int, width: int, device: torch.device=None):
        self.time_threshold = time_threshold
        self.height = height
        self.width = width
    
    def __call__(self, events: Union[pd.DataFrame, np.ndarray, torch.Tensor], order: List[str]=None, \
                    reset_indices: bool=True) -> Union[pd.DataFrame, np.ndarray, torch.Tensor]:
        return self.filter_events(events=events, order=order, reset_indices=reset_indices)
    
    def filter_events(self, events: Union[pd.DataFrame, np.ndarray, torch.Tensor], order: List[str]=None, reset_indices: bool=True) -> Union[pd.DataFrame, np.ndarray, torch.Tensor]:
        """
        Filter the inputted events using the BA filter. Supports events in the form of a pandas DataFrame, NumPy array, or PyTorch tensor.

        :param events: The events to filter.
        :type events: Union[pd.DataFrame, np.ndarray, torch.Tensor]
        :param order: The order of columns for the inputted events tensor. Defaults to None.
        :type order: List[str], optional
        :param reset_indices: Whether to reset the indices of the returned pandas DataFrame after filtering (if the input is a DataFrame). Defaults to True.
        :type reset_indices: bool, optional
        :return: The filtered events.
        :rtype: Union[pd.DataFrame, np.ndarray, torch.Tensor]
        """
        if isinstance(events, torch.Tensor):
            return self.filter_events_tensor(events, order=order)
        elif isinstance(events, pd.DataFrame):
            return self.filter_events_pd(events, reset_indices=reset_indices)
        else:
            return self.filter_events_np(events)
    
    def filter_events_pd(self, events: pd.DataFrame, reset_indices: bool) -> pd.DataFrame:
        """
        Filter a Pandas DataFrame of events using the BA filter.

        :param events: The events to filter.
        :type events: pd.DataFrame
        :param reset_indices: Whether to reset the indices of the returned DataFrame after filtering.
        :type reset_indices: bool, optional
        :return: The filtered events.
        :rtype: pd.DataFrame
        """
        filter_state = np.full((2, self.height, self.width), -self.time_threshold, dtype=events['t'].dtype)
        # 0, 0
        dr = [0, 0, 1, 1, 1, -1, -1, -1]
        dc = [1, -1, -1, 0, 1, -1, 0, 1]
        
        def include_event(row):
            cur_X = int(row['x'])
            cur_Y = int(row['y'])
            cur_T = row['t']
            cur_P = int(row['p'])
            cur_P_ind = int(cur_P * ((0.5 * cur_P) + 0.5))

            for dir in range(len(dr)):
                n_Y = cur_Y + dr[dir]
                n_X = cur_X + dc[dir]
                if ((n_Y < 0) or (n_Y >= self.height) or (n_X < 0) or (n_X >= self.width)):
                    continue
                filter_state[cur_P_ind, n_Y, n_X] = cur_T
            
            return ((cur_T - filter_state[cur_P_ind, cur_Y, cur_X]) < self.time_threshold)

        include_mask = events.apply(include_event, axis=1)

        if reset_indices:
            return events[include_mask].reset_index(drop=True)
        
        return events[include_mask]
    
    def filter_events_np(self, events: np.ndarray) -> np.ndarray:
        """
        Filter a numpy array of events using the BA filter.

        :param events: The events to filter.
        :type events: np.ndarray
        :return: The filtered events.
        :rtype: np.ndarray
        """
        filter_state = np.full((2, self.height, self.width), -self.time_threshold, events['t'].dtype)  # initializing state with all -1s

        events_reshaped = events.reshape(-1, 1)

        dr = np.array([0, 0, 1, 1, 1, -1, -1, -1])
        dc = np.array([1, -1, -1, 0, 1, -1, 0, 1])

        def include_event(event):
            cur_X = int(event['x'][0])
            cur_Y = int(event['y'][0])
            cur_T = event['t'][0]
            cur_P = int(event['p'][0])
            cur_P_ind = int(cur_P * ((0.5 * cur_P) + 0.5))

            for dir in range(len(dr)):
                n_Y = cur_Y + dr[dir]
                n_X = cur_X + dc[dir]
                if ((n_Y < 0) or (n_Y >= self.height) or (n_X < 0) or (n_X >= self.width)):
                    continue
                filter_state[cur_P_ind, n_Y, n_X] = cur_T

            return ((cur_T - filter_state[cur_P_ind, cur_Y, cur_X]) < self.time_threshold)

        include_mask = np.apply_along_axis(include_event, axis=1, arr=events_reshaped)  # events is assumed to be a structured numpy array, as usual

        return events[include_mask]
    
    def filter_events_tensor(self, events: torch.Tensor, order: List[str]=None) -> torch.Tensor:
        """
        Filter a torch.Tensor of events using the BA filter.

        :param events: The events to filter.
        :type events: torch.Tensor
        :param order: The order of the columns of the inputted events tensor. Defaults to ['b', 't', 'x', 'y', 'p'].
        :type order: List[str], optional
        :return: The filtered events.
        :rtype: torch.Tensor
        """
        if order is None:
            order = ['b', 't', 'x', 'y', 'p']
            print(f"Warning: No order provided. Assuming default order {order}.")
        temp_df = pd.DataFrame(events.detach().cpu().numpy(), columns=order)
        temp_df = self.filter_events_pd(temp_df, reset_indices=True)
        return torch.tensor(temp_df.to_numpy(), dtype=events.dtype, device=events.device)
    
    def filter_frames_tensor(self, frames: torch.Tensor, inplace: bool, device: torch.device=None, polarity_agnostic: bool=False):
        """
        Apply the BA filter to a 4D tensor of frames (frame volume of events).

        :param frames: A 4D torch.Tensor of frames to filter. Expected format: (N, C, X, Y), where N is the number of frames, C is the number of channels (should be 1), and (X, Y) are the frame dimensions.
        :type frames: torch.Tensor
        :param inplace: Whether to apply the filter in place or out of place.
        :type inplace: bool
        :param device: The device to run the filter on (e.g., CPU or GPU). If not provided, defaults to "cuda" if available.
        :type device: torch.device, optional
        :param polarity_agnostic: If True, the filter will treat both positive and negative polarities equally by applying the filtering operation on the absolute values of the input frames. 
                                   If False, the filter will handle positive and negative polarities independently, applying polarity-specific masks during filtering.
        :type polarity_agnostic: bool, optional
        :return: The filtered frames as a torch.Tensor.
        :rtype: torch.Tensor
        """

        device = ("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        if not torch.is_tensor(frames): frames = torch.from_numpy(frames, dtype=torch.float)  # convert from numpy if not tensor input
        if len(frames.shape) == 2: frames = frames[None, None, :]  # if fed in a 2D image, expand dim to 4
        if len(frames.shape) == 3: frames = frames[:, None, :]  # interpreted to be (N, X, Y); inserting a channel dimension
        assert len(frames.shape) == 4, "Input stack must be of shape [Frames, Channels, X, Y]"
        assert frames.shape[1] == 1, "Channel dimension must be 1"
        frames = frames.to(device)  # ensure on correct device

        if polarity_agnostic:
            abs_frames = torch.abs(frames)
            mask = self._get_filter_mask(frames=abs_frames, device=device)
            if inplace:
                frames *= mask
                return frames
            return mask * frames
        else:
            abs_frames = torch.abs(frames.detach()) #! detaching from computational graph so that gradients don't propagate backwards on the masks themselves
            intermediate_mask = torch.ones_like(frames.detach())

            abs_frames[frames < 0] = 0
            mask1 = self._get_filter_mask(frames=abs_frames, device=device)
            intermediate_mask[frames < 0] = 0
            mask1 = torch.clamp(mask1 + intermediate_mask, min=0, max=1).to(device) # mask for negative events

            abs_frames[frames < 0] = 1
            abs_frames[frames > 0] = 0
            intermediate_mask[frames < 0] = 1
            intermediate_mask[frames > 0] = 0
            mask2 = self._get_filter_mask(frames=abs_frames, device=device)
            mask2 = torch.clamp(mask2 + intermediate_mask, min=0, max=1).to(device) # mask for positive events

            if inplace:
                frames *= mask1 * mask2
                return frames
            return mask1 * mask2 * frames

    def _get_filter_mask(self, frames: torch.Tensor, device: torch.device=None) -> torch.Tensor:
        device = ("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        if not torch.is_tensor(frames): frames = torch.from_numpy(frames, dtype=torch.float)  # convert from numpy if not tensor input
        if len(frames.shape) == 2: frames = frames[None, None, :]  # if fed in a 2D image, expand dim to 4
        if len(frames.shape) == 3: frames = frames[:, None, :]  # interpreted to be (N, X, Y); inserting a channel dimension
        assert len(frames.shape) == 4, "Input stack must be of shape [Frames, Channels, X, Y]"
        assert frames.shape[1] == 1, "Channel dimension must be 1"
        frames = frames.to(device)  # ensure on correct device

        skern = 3

        t_frames_2 = torch.abs(frames)  # return magnitude of all events
        N, C, X, Y = t_frames_2.shape  # determine the nominal shape of the input

        t_frames_2 = t_frames_2.permute(1, 2, 3, 0)  # permute so time axis is first
        t_frames_2 = t_frames_2.view(-1, N)[:, None, :]  # reshape such that [minibatches, channels, timepoints]

        t_kern = torch.ones(self.time_threshold)[None, None, :] # in real version, just replace frames
        t_kern = t_kern.to(device)

        st_acc_frames = torch.nn.functional.conv1d(F.pad(t_frames_2, (self.time_threshold - 1, 0), "constant", 0), t_kern, stride=1).to(device)  # perform 1D conv

        st_acc_frames = st_acc_frames.squeeze(0).view(C, X, Y, N)  # reshape result
        st_acc_frames = st_acc_frames.permute(3, 0, 1, 2)  # This is a spatial map WEIGHTED by their temporal neighbors at each pixel.

        # Restoring t_frames_2 to original view
        #? t_frames_2 = t_frames_2.squeeze(0).view(C, X, Y, N)
        #? t_frames_2 = t_frames_2.permute(3, 0, 1, 2) #

        # Second step: Determine weighted spatial support in a rolling window. This can be done by looking at the unfolded image along the XY dimension unwrapping 
        # by a 3x3 neighborhood
        # setup parameters for torch unfold
        unfold_params = dict(
            kernel_size = (skern, skern),  # kernel size in x,y
            dilation=1, # no dilation
            padding= (skern // 2),  # zero padding will reduce errors along edge at the cost of some activity
            stride=1  #no stride
        )

        unfold = torch.nn.Unfold(**unfold_params)  # create an instance of the unfold class
        st_acc_frames = unfold(st_acc_frames)  # since have inverse operation, can overwrite original
        
        # st_acc_frames -> [N, (skern * skern), (width * height)]
        # Applying mask to discard (zero out) the center element of each 3x3 time-accumulated kernel
        mid = skern**2//2
        filt = torch.ones((skern**2)).to(device)
        filt[mid] = 0
        filt = filt[None, :, None]
        # print(st_acc_frames.shape, filt.shape)
        mask1, _ = torch.max(st_acc_frames*filt, dim=1, keepdim=True)  # remove central value in each window then use max to determine if any nonzeroes exist. Normalize by maximum possible value, tkern
        mask1 = mask1.squeeze().view(N, C, X, Y).to(device)
        mask1 = mask1.clamp(min=0, max=1).to(device)
        return mask1.to(device)

        # mask1 = mask1 / self.time_threshold #! SHOULD BE UNCOMMENTED IF USING CEILING APPROACH BELOW

        #? mask1 = torch.nn.functional.pad(mask1, pad=(0, 0, (skern**2-1)//2, (skern**2-1)//2)).to(device)
        #? st_acc_frames = mask1 * st_acc_frames

        # FOLD STEP
        # Spatial sum of temporally accumulated pixels (event density matrix)
        #? fold = torch.nn.Fold(output_size = (X, Y), **unfold_params) # sum of each kernel unit is stored (fold documentation)
        #? st_acc_frames = fold(st_acc_frames)  # fold adds each replica together so end result is pixel * win_size
        
        #? st_acc_frames = torch.clamp(st_acc_frames, min=0, max=1) #! below should have the same functionality?
        
        # st_acc_frames = torch.ceil(st_acc_frames / self.time_threshold) #! above should have the same functionality?
        #! UNCOMMENT THE NORMALIZATION LINE ABOVE (/ self.time_threshold) IF USING THE CEILING IMPLEMENTATION
        #! / by self.time_threshold over here not needed for above due to it being
        #! performed on the mask prior to the fold?
        
        # if inplace:
        #     frames *= st_acc_frames
        #     return frames
        
        # return st_acc_frames * frames

    def set_time_threshold(self, time_threshold: Union[int, float]):
        """
        Set the time threshold for the BA filter.

        :param time_threshold: The new time threshold.
        :type time_threshold: Union[int, float]
        """
        self.time_threshold = time_threshold
    
    def set_height(self, height: int):
        """
        Set the height of the event camera frame.

        :param height: The new height.
        :type height: int
        """
        self.height = height
    
    def set_width(self, width: int):
        """
        Set the width of the event camera frame.

        :param width: The new width.
        :type width: int
        """
        self.width = width

class IEFilter(Filter):
    """
    A class to filter event-based vision data using an inceptive event (IE) filter based on temporal and spatial constraints.

    E-MLB: https://ieeexplore.ieee.org/document/10078400
    Based on the following paper (as referenced in E-MLB): https://arxiv.org/abs/2002.11656
    R. Baldwin, M. Almatrafi, J. R. Kaufman, V. Asari, and K. Hirakawa, “Inceptive event time-surfaces for object classification using neuromorphic
    cameras,” in Proc. Int. Conf. Image Anal. Recognit., 2019, pp. 395-403.
    
    Will only classify events `e_i` as signal if `t_i - t_{i - 1} > thresh_negative` and `t_{i + 1} - t_i < thresh_positive` where `t_i \in T(x, y, p)`.
    Can set either threshold to `None` if you do not want the restriction corresponding to that threshold to be imposed on the data.
    """
    def __init__(self, thresh_negative: Union[float, int], thresh_positive: Union[float, int], height: int, width: int):
        self.width = width
        self.height = height
        self.thresh_negative = thresh_negative
        self.thresh_positive = thresh_positive
    
    def __call__(self, events: Union[pd.DataFrame, np.ndarray, torch.Tensor], order: List[str]=None, \
                 reset_indices: bool=True) -> Union[pd.DataFrame, np.ndarray, torch.Tensor]:
        return self.filter_events(events=events, order=order, reset_indices=reset_indices)

    def filter_events(self, events: Union[pd.DataFrame, np.ndarray, torch.Tensor], order: List[str]=None, \
               reset_indices: bool=True) -> Union[pd.DataFrame, np.ndarray, torch.Tensor]:
        """
        Filter the input events using the IE filter based on temporal constraints defined by `thresh_negative` and `thresh_positive`.
        
        :param events: The events to filter, which can be a Pandas DataFrame, NumPy array, or PyTorch Tensor.
        :type events: Union[pd.DataFrame, np.ndarray, torch.Tensor]
        :param order: The order of the columns for torch.Tensor input. If `None`, assumes a default order handled by helper methods.
        :type order: List[str], optional
        :param reset_indices: Whether to reset the indices of the filtered DataFrame if input is a Pandas DataFrame.
        :type reset_indices: bool, optional
        :return: The filtered events in the same format as input (Pandas DataFrame, NumPy array, or PyTorch Tensor).
        :rtype: Union[pd.DataFrame, np.ndarray, torch.Tensor]
        """
        
        if isinstance(events, torch.Tensor):
            return self.filter_events_tensor(events, order=order)
        elif isinstance(events, pd.DataFrame):
            return self.filter_events_pd(events, reset_indices=reset_indices)
        else:
            return self.filter_events_np(events)
    def filter_events_pd(self, events: pd.DataFrame, reset_indices: bool) -> pd.DataFrame:
        """
        Filter a Pandas DataFrame of events using the IE filter. Assumes that row indices of the data frame have not been modified.

        :param events: The events to filter.
        :type events: pd.DataFrame
        :param reset_indices: Whether to reset the indices of the returned DataFrame after filtering.
        :type reset_indices: bool, optional
        :return: The filtered events.
        :rtype: pd.DataFrame
        """
        class NumberRef:
            false_signal_num = 0

        filter_state = np.full((2, self.height, self.width), -1, dtype=events['t'].dtype)
        index_state = np.full((2, self.height, self.width), -1, dtype=np.int64)
        false_signal_indices = np.empty(len(events), dtype=np.int64)

        def include_event(row, num_ref: NumberRef):
            cur_X = int(row['x'])
            cur_Y = int(row['y'])
            cur_T = row['t']
            cur_P = int(row['p'])
            cur_P_ind = int(cur_P * ((0.5 * cur_P) + 0.5))
            cur_row_ind = int(row.name)

            if filter_state[cur_P_ind, cur_Y, cur_X] < 0:
                filter_state[cur_P_ind, cur_Y, cur_X] = cur_T
                index_state[cur_P_ind, cur_Y, cur_X] = cur_row_ind
                return True
            
            if (self.thresh_negative is not None) and ((cur_T - filter_state[cur_P_ind, cur_Y, cur_X]) <= self.thresh_negative):
                filter_state[cur_P_ind, cur_Y, cur_X] = cur_T
                index_state[cur_P_ind, cur_Y, cur_X] = cur_row_ind
                return False
            
            if (self.thresh_positive is not None) and ((cur_T - filter_state[cur_P_ind, cur_Y, cur_X]) >= self.thresh_positive):
                false_signal_indices[num_ref.false_signal_num] = index_state[cur_P_ind, cur_Y, cur_X]
                num_ref.false_signal_num += 1

            filter_state[cur_P_ind, cur_Y, cur_X] = cur_T
            index_state[cur_P_ind, cur_Y, cur_X] = cur_row_ind
            
            # condition_1 = True
            # condition_2 = True
            # if cur_row_ind > 0:
            #     condition_1 = (self.thresh_negative is None) or ((cur_T - events.loc[cur_row_ind - 1]['t']) > self.thresh_negative)
            # if cur_row_ind < (len(events) - 1):
            #     condition_2 = (self.thresh_positive is None) or ((events.loc[cur_row_ind + 1]['t'] - cur_T) < self.thresh_positive)

            # return (condition_1 and condition_2)

            return True

        include_mask = events.apply(lambda row: include_event(row, NumberRef), axis=1)
        include_mask[false_signal_indices[:NumberRef.false_signal_num]] = False

        if reset_indices:
            return events[include_mask].reset_index(drop=True)
        
        return events[include_mask]
    
    def filter_events_np(self, events: np.ndarray) -> np.ndarray:
        """
        Filter a numpy array of events using the IE filter.

        :param events: The events to filter.
        :type events: np.ndarray
        :return: The filtered events.
        :rtype: np.ndarray
        """
        events_reshaped = events.reshape(-1, 1)

        NUM_EVENTS = len(events)
        NUM_EVENTS_MINUS_ONE = NUM_EVENTS - 1
        THRESH_NEGATIVE = self.thresh_negative
        THRESH_POSITIVE = self.thresh_positive

        filter_state = np.full((2, self.height, self.width), -1, dtype=events['t'].dtype)
        index_state = np.full((2, self.height, self.width), -1, dtype=np.int64)
        false_signal_indices = np.empty(len(events), dtype=np.int64)

        class NumRef:
            ctr = 0
            false_signal_num = 0

        # class RowTracker: # avoids copying the NumPy structured array in order to add an index column
        #     def __init__(self):
        #         self.ctr = 0
        #         self.false_signal_num = 0
        def include_event(row, num_ref: NumRef):
            cur_T = row['t'][0]
            cur_X = int(row['x'][0])
            cur_Y = int(row['y'][0])
            cur_P = int(row['p'][0])
            cur_P_ind = int(cur_P * ((0.5 * cur_P) + 0.5))

            if filter_state[cur_P_ind, cur_Y, cur_X] < 0:
                filter_state[cur_P_ind, cur_Y, cur_X] = cur_T
                index_state[cur_P_ind, cur_Y, cur_X] = num_ref.ctr
                num_ref.ctr += 1
                return True
            
            if (THRESH_NEGATIVE is not None) and ((cur_T - filter_state[cur_P_ind, cur_Y, cur_X]) <= THRESH_NEGATIVE):
                filter_state[cur_P_ind, cur_Y, cur_X] = cur_T
                index_state[cur_P_ind, cur_Y, cur_X] = num_ref.ctr
                num_ref.ctr += 1
                return False
            
            if (THRESH_POSITIVE is not None) and ((cur_T - filter_state[cur_P_ind, cur_Y, cur_X]) >= THRESH_POSITIVE):
                false_signal_indices[num_ref.false_signal_num] = index_state[cur_P_ind, cur_Y, cur_X]
                num_ref.false_signal_num += 1

            filter_state[cur_P_ind, cur_Y, cur_X] = cur_T
            index_state[cur_P_ind, cur_Y, cur_X] = num_ref.ctr
            
            num_ref.ctr += 1
            return True
        
        # row_tracker_wrapper = RowTracker()
        include_mask = np.apply_along_axis(lambda event: include_event(event, NumRef), axis=1, arr=events_reshaped)  # events is assumed to be a structured numpy array, as usual
        # print(row_tracker_wrapper.ctr)
        include_mask[false_signal_indices[:NumRef.false_signal_num]] = False

        return events[include_mask]
    
    def filter_events_tensor(self, events: torch.Tensor, order: List[str]=None) -> torch.Tensor:
        """
        Filter a torch.Tensor of events using the IE filter.

        :param events: The events to filter.
        :type events: torch.Tensor
        :param order: The order of the columns of the inputted events tensor. Defaults to ['b', 't', 'x', 'y', 'p'].
        :type order: List[str], optional
        :return: The filtered events.
        :rtype: torch.Tensor
        """
        if order is None:
            order = ['b', 't', 'x', 'y', 'p']
            print(f"Warning: No order provided. Assuming default order {order}.")
        temp_df = pd.DataFrame(events.detach().cpu().numpy(), columns=order)
        temp_df = self.filter_events_pd(temp_df, reset_indices=True)
        return torch.tensor(temp_df.to_numpy(), dtype=events.dtype, device=events.device)
    
    def filter_frames_tensor(self, frames: torch.Tensor, inplace: bool, device: torch.device=None, polarity_agnostic: bool=False) -> torch.Tensor:
        """
        Apply the IE filter to a 4D tensor of frames (frame volume of events).

        :param frames: A 4D torch.Tensor with shape (N, C, X, Y), where N is the number of frames, C is the channel (should be 1), X is the height, and Y is the width.
        :type frames: torch.Tensor
        :param inplace: Whether to modify the tensor in place or return a new filtered tensor.
        :type inplace: bool
        :param device: The device to run the filtering on. If not specified, defaults to 'cuda' if available.
        :type device: torch.device, optional
        :param polarity_agnostic: If True, the filter will treat both positive and negative polarities equally by applying the filtering operation on the absolute values of the input frames. 
                                   If False, the filter will handle positive and negative polarities independently, applying polarity-specific masks during filtering.
        :type polarity_agnostic: bool, optional
        :return: A torch.Tensor containing the filtered frames.
        :rtype: torch.Tensor
        """
        
        device = ("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        if not torch.is_tensor(frames): frames = torch.from_numpy(frames, dtype=torch.float)  # convert from numpy if not tensor input
        if len(frames.shape) == 2: frames = frames[None, None, :]  # if fed in a 2D image, expand dim to 4
        if len(frames.shape) == 3: frames = frames[:, None, :]  # interpreted to be (N, X, Y); inserting a channel dimension
        assert len(frames.shape) == 4, "Input stack must be of shape [Frames, Channels, X, Y]"
        assert frames.shape[1] == 1, "Channel dimension must be 1"
        frames = frames.to(device)  # ensure on correct device

        if polarity_agnostic:
            abs_frames = torch.abs(frames)
            mask = self._get_filter_mask(frames=abs_frames, device=device)
            if inplace:
                frames *= mask
                return frames
            return mask * frames
        else:
            abs_frames = torch.abs(frames.detach()) #! detaching from computational graph so that gradients don't propagate backwards on the masks themselves
            intermediate_mask = torch.ones_like(frames.detach())

            abs_frames[frames < 0] = 0
            mask1 = self._get_filter_mask(frames=abs_frames, device=device)
            intermediate_mask[frames < 0] = 0
            mask1 = torch.clamp(mask1 + intermediate_mask, min=0, max=1).to(device) # mask for negative events

            abs_frames[frames < 0] = 1
            abs_frames[frames > 0] = 0
            intermediate_mask[frames < 0] = 1
            intermediate_mask[frames > 0] = 0
            mask2 = self._get_filter_mask(frames=abs_frames, device=device)
            mask2 = torch.clamp(mask2 + intermediate_mask, min=0, max=1).to(device) # mask for positive events

            if inplace:
                frames *= mask1 * mask2
                return frames
            return mask1 * mask2 * frames

    def _get_filter_mask(self, frames: torch.Tensor, device: torch.device=None) -> torch.Tensor:
        device = ("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        if not torch.is_tensor(frames): frames = torch.from_numpy(frames, dtype=torch.float)  # convert from numpy if not tensor input
        if len(frames.shape) == 2: frames = frames[None, None, :]  # if fed in a 2D image, expand dim to 4
        if len(frames.shape) == 3: frames = frames[:, None, :]  # interpreted to be (N, X, Y); inserting a channel dimension
        assert len(frames.shape) == 4, "Input stack must be of shape [Frames, Channels, X, Y]"
        assert frames.shape[1] == 1, "Channel dimension must be 1"
        frames = frames.to(device)  # ensure on correct device

        t_frames = torch.abs(frames).to(device)  # insert channel dimension and return magnitude of all events
        N, C, X, Y = t_frames.shape  # determine the nominal shape of the input

        t_frames = t_frames.permute(1, 2, 3, 0)  # permute so time axis is first
        t_frames = t_frames.view(-1, N)[:, None, :]  # reshape such that [minibatches, channels, timepoints]

        # setup the kernels
        kern_n = torch.ones(self.thresh_negative)[None, None, :] # in real version, just replace frames
        kern_p = torch.ones(self.thresh_positive)[None, None, :]
        kern_n = kern_n.to(device)
        kern_p = kern_p.to(device)
        
        # Summing Past Frames
        # apply convolution and reshape to nominal shape
        past_time_acc_frames = torch.nn.functional.conv1d(F.pad(t_frames, (self.thresh_negative - 1, 0), "constant", 0), kern_n, stride=1).to(device)  # perform 1D conv
        past_time_acc_frames = past_time_acc_frames.squeeze(0).view(C, X, Y, N)  # reshape result
        past_time_acc_frames = past_time_acc_frames.permute(3, 0, 1, 2)  # This is a spatial map WEIGHTED by their temporal neighbors at each pixel.
        # print(past_time_acc_frames.shape)

        # Summing Future Frames
        t_frames = torch.flip(t_frames, dims=[-1]) # flipping the N timepoints as we want to compute a prefix/accumulate from the "future"
        future_time_acc_frames = torch.nn.functional.conv1d(F.pad(t_frames, (self.thresh_positive - 1, 0), "constant", 0), kern_p, stride=1).to(device)  # perform 1D conv
        future_time_acc_frames = torch.flip(future_time_acc_frames, dims=[-1]) # flipping the output back to be in ascending temporal order
        future_time_acc_frames = future_time_acc_frames.squeeze(0).view(C, X, Y, N)  # reshape result
        future_time_acc_frames = future_time_acc_frames.permute(3, 0, 1, 2)  # This is a spatial map WEIGHTED by their temporal neighbors at each pixel.
        # print(future_time_acc_frames.shape)

        # Completing creation of the filter/mask and applying it to the frames
        past_time_acc_frames.neg_() # in-place negation
        past_time_acc_frames += 2
        past_time_acc_frames = torch.clamp(past_time_acc_frames, min=0, max=1).to(device)
        # 0 -> 1, 1 -> 1, 2 -> 0, 3 -> 0, 4 -> 0, 5 -> 0, etc.

        future_time_acc_frames -= 1
        future_time_acc_frames = torch.clamp(future_time_acc_frames, min=0, max=1).to(device)

        #? if inplace:
        #?    frames *= past_time_acc_frames * future_time_acc_frames
        #?    return frames
        
        #? return (frames * past_time_acc_frames * future_time_acc_frames)

        return past_time_acc_frames * future_time_acc_frames
    
    def set_thresh_negative(self, thresh_negative: Union[float, int]):
        """
        Set the negative time threshold of the IE filter.

        :param height: The new negative threshold.
        :type height: Union[float, int]
        """
        self.thresh_negative = thresh_negative
    
    def set_thresh_positive(self, thresh_positive: Union[float, int]):
        """
        Set the postiive time threshold of the IE filter.

        :param width: The new positive threshold.
        :type width: Union[float, int]
        """
        self.thresh_positive = thresh_positive
    
    def set_height(self, height: int):
        """
        Set the height of the event camera frame.

        :param height: The new height.
        :type height: int
        """
        self.height = height
    
    def set_width(self, width: int):
        """
        Set the width of the event camera frame.

        :param width: The new width.
        :type width: int
        """
        self.width = width

class YNoiseFilter(Filter):
    """
    A class to filter event-based vision data using the YNoise filter (event density and hot pixels) based on temporal and spatial constraints.

    E-MLB: https://ieeexplore.ieee.org/document/10078400
    Based on the following paper (as referenced in E-MLB): https://doi.org/10.3390/app10062024
    Y. Feng et al., “Event density based denoising method for dynamic vision sensor,” Appl. Sci., vol. 10, no. 6, 2020, Art. no. 2024.
    
    Event density-based filtering followed by hot pixel filtering. The space window size should be odd.
    """
    def __init__(self, time_context_delta: Union[float, int], space_window_size: int, density_threshold: Union[int, float], height: int, width: int):
        assert space_window_size%2 == 1, "`space_window_size` must be odd"
        
        self.time_context_delta = time_context_delta
        self.space_window_size = space_window_size
        self.height = height
        self.width = width
        self.density_threshold = density_threshold

    def __call__(self, events: Union[pd.DataFrame, np.ndarray, torch.Tensor], order: List[str]=None, \
                 reset_indices: bool=True, hot_pixel_filter: bool=True) -> Union[pd.DataFrame, np.ndarray, torch.Tensor]:
        return self.filter_events(events=events, order=order, reset_indices=reset_indices, hot_pixel_filter=hot_pixel_filter)

    def filter_events(self, events: Union[pd.DataFrame, np.ndarray, torch.Tensor], order: List[str]=None, reset_indices: bool=True, hot_pixel_filter: bool=True) -> Union[pd.DataFrame, np.ndarray, torch.Tensor]:
        """
        Filter events using the YNoise filter. This includes event density filtering and optional hot pixel filtering.

        :param events: The events to filter, which can be in Pandas, NumPy, or torch.Tensor format.
        :type events: Union[pd.DataFrame, np.ndarray, torch.Tensor]
        :param order: The column order for the events tensor. Defaults to None (will assume ['b', 't', 'x', 'y', 'p'] for tensors).
        :type order: List[str], optional
        :param reset_indices: Whether to reset the indices of the returned Pandas DataFrame after filtering.
        :type reset_indices: bool, optional
        :param hot_pixel_filter: Whether to apply the hot pixel filter after the density filter.
        :type hot_pixel_filter: bool
        :return: The filtered events in the same format as the input.
        :rtype: Union[pd.DataFrame, np.ndarray, torch.Tensor]
        """
        
        if isinstance(events, torch.Tensor):
            return self.filter_events_tensor(events, order=order, hot_pixel_filter=hot_pixel_filter)
        elif isinstance(events, pd.DataFrame):
            return self.filter_events_pd(events, reset_indices=reset_indices, hot_pixel_filter=hot_pixel_filter)
        else:
            return self.filter_events_np(events, hot_pixel_filter=hot_pixel_filter)
    
    def filter_events_pd(self, events: pd.DataFrame, reset_indices: bool, hot_pixel_filter: bool=True) -> pd.DataFrame:
        """
        Filter a Pandas DataFrame of events using the YNoise filter. Assumes that row indices of the data frame have not been modified.

        :param events: The events to filter.
        :type events: pd.DataFrame
        :param reset_indices: Whether to reset the indices of the returned DataFrame after filtering.
        :type reset_indices: bool, optional
        :param hot_pixel_filter: Whether to apply the hot pixel filter to the result of the initial event density filter on the inputted events.
        :type hot_pixel_filter: bool
        :return: The filtered events.
        :rtype: pd.DataFrame
        """

        if hot_pixel_filter:
            pass_event_density = np.empty(len(events), dtype=bool)
            hot_pixel_density_state = np.zeros((self.height, self.width), dtype=np.int64)
            hot_pixel_mask = np.array([[1, 1, 1],
                                       [1, 0, 1],
                                       [1, 1, 1]])
        
        density_state = np.zeros((self.height, self.width), dtype=np.int64)

        class NumReference:
            last_event_ptr = 0
        # First approach
        def include_event(row, num_reference: NumReference):
            cur_X = int(row['x'])
            cur_Y = int(row['y'])
            cur_T = row['t']
            cur_row_ind = int(row.name)
            
            # should still technically be O(N); though, iterating twice; with second indirect iteration being explicit and "non-vectorized" (following loop)
            # issue with current approach: unnecessarily updating the time surface for spaces that are not near/not relevant to space context of current event
            # issue with second approach: too much memory usage? (H x W x len(events))
            density_state[cur_Y, cur_X] += 1

            if hot_pixel_filter:
                hot_pixel_density_state[cur_Y, cur_X] += 1
            
            for event_index in (x for x in range(num_reference.last_event_ptr, cur_row_ind)):
                event_iter = events.loc[event_index]
                if (cur_T - self.time_context_delta) <= event_iter['t']:
                    num_reference.last_event_ptr = event_index
                    break
                density_state[int(event_iter['y']), int(event_iter['x'])] -= 1
                if hot_pixel_filter:
                    if pass_event_density[event_index]:
                        hot_pixel_density_state[int(event_iter['y']), int(event_iter['x'])] -= 1
            # not count_nonzero
            cur_event_density = np.sum(
                density_state[
                    max((cur_Y - int(self.space_window_size / 2)), 0):(cur_Y + int(self.space_window_size / 2) + 1),
                    max((cur_X - int(self.space_window_size / 2)), 0):(cur_X + int(self.space_window_size / 2) + 1)
                ]
            ) # will handle out-of-bounds (built-in)

            cur_passed_density = cur_event_density >= self.density_threshold

            if hot_pixel_filter:
                pass_event_density[cur_row_ind] = cur_passed_density
                if cur_passed_density:
                    # equivalent to L1 loss since all matrix entries should be > 0
                    min_mask_Y = 1 if (cur_Y - 1) < 0 else 0
                    max_mask_Y = 3 if (cur_Y + 1) < self.height else 2
                    min_mask_X = 1 if (cur_X - 1) < 0 else 0
                    max_mask_X = 3 if (cur_X + 1) < self.width else 2

                    min_Y = max(0, cur_Y - 1)
                    max_Y = min(cur_Y + 2, self.height)
                    min_X = max(0, cur_X - 1)
                    max_X = min(cur_X + 2, self.width)

                    hot_pixel_density = np.sum(hot_pixel_density_state[min_Y:max_Y, min_X:max_X] * \
                                               hot_pixel_mask[min_mask_Y:max_mask_Y, min_mask_X:max_mask_X])
                    return hot_pixel_density != 0
                else:
                    hot_pixel_density_state[cur_Y, cur_X] -= 1
                    return False

            return cur_passed_density
        
        include_mask = events.apply(lambda event: include_event(event, NumReference), axis=1)

        if reset_indices:
            return events[include_mask].reset_index(drop=True)
        
        return events[include_mask]
    
    def filter_events_np(self, events: np.ndarray, hot_pixel_filter: bool=True) -> np.ndarray:
        """
        Filter a numpy structured array of events using the YNoise filter.

        :param events: The events to filter.
        :type events: np.ndarray
        :param hot_pixel_filter: Whether to apply the hot pixel filter to the result of the initial event density filter on the inputted events.
        :type hot_pixel_filter: bool
        :return: The filtered events.
        :rtype: np.ndarray
        """
        events_reshaped = events.reshape(-1, 1)

        if hot_pixel_filter:
            pass_event_density = np.empty(len(events_reshaped), dtype=bool)
            hot_pixel_density_state = np.zeros((self.height, self.width), dtype=np.int64)
            hot_pixel_mask = np.array([[1, 1, 1],
                                       [1, 0, 1],
                                       [1, 1, 1]])
        
        density_state = np.zeros((self.height, self.width), dtype=np.int64)

        class NumReference:
            last_event_ptr = 0
            cur_row_ind = 0

        def include_event(row, num_reference: NumReference):
            cur_T = row['t'][0]
            cur_X = int(row['x'][0])
            cur_Y = int(row['y'][0])
            
            density_state[cur_Y, cur_X] += 1

            if hot_pixel_filter:
                hot_pixel_density_state[cur_Y, cur_X] += 1
            
            for event_index in (x for x in range(num_reference.last_event_ptr, num_reference.cur_row_ind)):
                event_iter = events_reshaped[event_index][0]
                if (cur_T - self.time_context_delta) <= event_iter['t']:
                    num_reference.last_event_ptr = event_index
                    break
                density_state[int(event_iter['y']), int(event_iter['x'])] -= 1
                if hot_pixel_filter:
                    if pass_event_density[event_index]:
                        hot_pixel_density_state[int(event_iter['y']), int(event_iter['x'])] -= 1
            # not count_nonzero
            cur_event_density = np.sum(
                density_state[
                    max((cur_Y - int(self.space_window_size / 2)), 0):(cur_Y + int(self.space_window_size / 2) + 1),
                    max((cur_X - int(self.space_window_size / 2)), 0):(cur_X + int(self.space_window_size / 2) + 1)
                ]
            ) # will handle out-of-bounds (built-in)

            cur_passed_density = cur_event_density >= self.density_threshold

            if hot_pixel_filter:
                pass_event_density[num_reference.cur_row_ind] = cur_passed_density
                if cur_passed_density:
                    # equivalent to L1 loss since all matrix entries should be > 0
                    min_mask_Y = 1 if (cur_Y - 1) < 0 else 0
                    max_mask_Y = 3 if (cur_Y + 1) < self.height else 2
                    min_mask_X = 1 if (cur_X - 1) < 0 else 0
                    max_mask_X = 3 if (cur_X + 1) < self.width else 2

                    min_Y = max(0, cur_Y - 1)
                    max_Y = min(cur_Y + 2, self.height)
                    min_X = max(0, cur_X - 1)
                    max_X = min(cur_X + 2, self.width)

                    hot_pixel_density = np.sum(hot_pixel_density_state[min_Y:max_Y, min_X:max_X] * \
                                               hot_pixel_mask[min_mask_Y:max_mask_Y, min_mask_X:max_mask_X])
                    num_reference.cur_row_ind += 1
                    return hot_pixel_density != 0
                else:
                    hot_pixel_density_state[cur_Y, cur_X] -= 1
                    num_reference.cur_row_ind += 1
                    return False
            
            num_reference.cur_row_ind += 1
            return cur_passed_density
        
        include_mask = np.apply_along_axis(lambda event: include_event(event, NumReference), axis=1, arr=events_reshaped)
        return events[include_mask]
    
    def filter_events_tensor(self, events: torch.Tensor, hot_pixel_filter: bool=True, order: List[str]=None) -> torch.Tensor:
        """
        Filter a torch.Tensor of events using the YNoise filter.

        :param events: The events to filter.
        :type events: torch.Tensor
        :param hot_pixel_filter: Whether to apply the hot pixel filter to the result of the initial event density filter on the inputted events.
        :type hot_pixel_filter: bool
        :param order: The order of the columns of the inputted events tensor. Defaults to ['b', 't', 'x', 'y', 'p'].
        :type order: List[str], optional
        :return: The filtered events.
        :rtype: torch.Tensor
        """
        if order is None:
            order = ['b', 't', 'x', 'y', 'p']
            print(f"Warning: No order provided. Assuming default order {order}.")
        temp_df = pd.DataFrame(events.detach().cpu().numpy(), columns=order)
        temp_df = self.filter_events_pd(temp_df, reset_indices=True, hot_pixel_filter=hot_pixel_filter)
        return torch.tensor(temp_df.to_numpy(), dtype=events.dtype, device=events.device)

    def filter_frames_tensor(self, frames: torch.Tensor, inplace: bool, hot_pixel_filter: bool=True, device: torch.device=None) -> torch.Tensor:
        """
        Filter a torch.Tensor frame volume of events using the YNoiseFilter filter.
        
        :param frames: The input tensor of event frames to be filtered. Expected shape: (N, C, X, Y).
        :type frames: torch.Tensor
        :param inplace: Whether to apply the filter in-place (modifies the input tensor directly) or out-of-place (returns a new tensor).
        :type inplace: bool
        :param hot_pixel_filter: Whether to apply the second phase hot-pixel filter after the YNoise filter. Defaults to True.
        :type hot_pixel_filter: bool, optional
        :param device: The device on which to perform the computations. If not provided, defaults to "cuda" if available.
        :type device: torch.device, optional
        :return: The filtered event frames tensor.
        :rtype: torch.Tensor
        """
        device = ("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        if not torch.is_tensor(frames): frames = torch.from_numpy(frames, dtype=torch.float)  # convert from numpy if not tensor input
        if len(frames.shape) == 2: frames = frames[None, None, :]  # if fed in a 2D image, expand dim to 4
        if len(frames.shape) == 3: frames = frames[:, None, :]  # interpreted to be (N, X, Y); inserting a channel dimension
        assert len(frames.shape) == 4, "Input stack must be of shape [Frames, Channels, X, Y]"
        assert frames.shape[1] == 1, "Channel dimension must be 1"
        frames = frames.to(device)  # ensure on correct device

        t_frames_2 = torch.abs(frames).to(device)  # insert channel dimension and return magnitude of all events
        N, C, X, Y = t_frames_2.shape  # determine the nominal shape of the input

        t_frames_2 = t_frames_2.permute(1, 2, 3, 0)  # permute so time axis is first
        t_frames_2 = t_frames_2.view(-1, N)[:, None, :]  # reshape such that [minibatches, channels, timepoints]

        t_kern = torch.ones(self.time_context_delta)[None, None, :] # in real version, just replace frames
        t_kern = t_kern.to(device)

        st_acc_frames = torch.nn.functional.conv1d(F.pad(t_frames_2, (self.time_context_delta - 1, 0), "constant", 0), t_kern, stride=1).to(device)  # perform 1D conv
        st_acc_frames = st_acc_frames.squeeze(0).view(C, X, Y, N)  # reshape result
        st_acc_frames = st_acc_frames.permute(3, 0, 1, 2)  # This is a spatial map WEIGHTED by their temporal neighbors at each pixel.

        # Restoring t_frames_2 to original view
        t_frames_2 = t_frames_2.squeeze(0).view(C, X, Y, N)
        t_frames_2 = t_frames_2.permute(3, 0, 1, 2)

        # Second step: Determine weighted spatial support in a rolling window. This can be done by looking at the unfolded image along the XY dimension unwrapping 
        # by a space_window_size x space_window_size neighborhood
        # setup parameters for torch unfold
        unfold_params = dict(
            kernel_size = (self.space_window_size, self.space_window_size),  # kernel size in x,y
            dilation=1, # no dilation
            padding=(self.space_window_size // 2),  # zero padding will reduce errors along edge at the cost of some activity
            stride=1  #no stride
        )

        unfold = torch.nn.Unfold(**unfold_params)  # create an instance of the unfold class
        st_acc_frames = unfold(st_acc_frames).to(device)  # since have inverse operation, can overwrite original
        
        #? #! ADDED
        mask1 = torch.sum(st_acc_frames, dim=1, keepdim=True).to(device)
        mask1 = mask1.squeeze().view(N, C, X, Y).to(device)

        #? pad_size = (self.space_window_size**2 - 1) // 2
        #? mask1 = torch.nn.functional.pad(mask1, pad=(0, 0, pad_size, pad_size))
        #? st_acc_frames = mask1.to(device) * st_acc_frames.to(device)
        #? #! END ADDED

        # Spatial sum of temporally accumulated pixels (event density matrix)
        #? fold = torch.nn.Fold(output_size = (X, Y), **unfold_params) # sum of each kernel unit is stored (fold documentation)
        #? st_acc_frames = fold(st_acc_frames).to(device)  # fold adds each replica together so end result is pixel * win_size

        mask1 -= (self.density_threshold - 1) #! >= thresh --> signal
        mask1 = torch.clamp(mask1, min=0, max=1).to(device)
        
        if hot_pixel_filter:
            second_phase_filter = BAFilter(time_threshold=self.time_context_delta, height=self.height, width=self.width)
            if inplace:
                frames *= mask1
                return second_phase_filter.filter_frames_tensor(frames, inplace=True, polarity_agnostic=True)
            else:
                return second_phase_filter.filter_frames_tensor(frames * mask1, inplace=False, polarity_agnostic=True)
        
        if inplace:
            frames *= mask1
            return frames
        else:
            return (frames * mask1)

    def set_density_threshold(self, density_threshold: Union[float, int]):
        """
        Set the density threshold of the YNoise filter.

        :param density_threshold: The new density threshold.
        :type density_threshold: Union[float, int]
        """
        self.density_threshold = density_threshold
    
    def set_time_context_delta(self, time_context_delta: Union[float, int]):
        """
        Set the density time_context_delta of the YNoise filter.

        :param time_context_delta: The new time context delta.
        :type time_context_delta: Union[float, int]
        """
        self.time_context_delta = time_context_delta
    
    def set_space_window_size(self, space_window_size: int):
        """
        Set the density space_window_size of the YNoise filter.

        :param space_window_size: The new space window size.
        :type space_window_size: int
        """
        self.space_window_size = space_window_size

    def set_height(self, height: int):
        """
        Set the height of the event camera frame.

        :param height: The new height.
        :type height: int
        """
        self.height = height
    
    def set_width(self, width: int):
        """
        Set the width of the event camera frame.

        :param width: The new width.
        :type width: int
        """
        self.width = width