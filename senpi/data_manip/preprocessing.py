from typing import List
from typing import Dict
from typing import Union
from typing import Generator
import torch
import numpy as np
import pandas as pd
import numpy.lib.recfunctions as rfn
from senpi.data_io.basic_utils import get_np_dtype_from_torch_dtype
from senpi.data_io.basic_utils import get_torch_dtype_from_np_dtype
from tqdm import tqdm
from senpi.constants import TimeUnit
import math

# 1
def np_events_to_tensor(events: np.ndarray, batch_index: int=0, transform_polarity: bool=True, stack_ret_order: List[str]=['b', 't', 'x', 'y', 'p'], \
                        stack_tensors: bool=True, dtypes: Dict[str, torch.dtype]=None, device: torch.device="cpu") -> torch.Tensor:
    """
    Convert a structured NumPy array of events to a stacked tensor.

    This function accepts a NumPy array where columns represent event attributes ('t', 'x', 'y', and 'p') and converts it into a PyTorch tensor.
    The resulting tensor can be stacked in a specified order and includes options for transforming polarity values.

    :param events: The structured NumPy array of events.
    :type events: np.ndarray
    :param batch_index: Index to assign to the batch dimension.
    :type batch_index: int
    :param transform_polarity: Whether to transform the polarity values such that -1 represents OFF events and 1 represents ON events (from 0 and 1). Note: if the polarities are already represented by -1s and 1s, they will remain the same no matter what `transform_polarity` is set to.
    :type transform_polarity: bool
    :param stack_ret_order: The order of dimensions in the resulting tensor, e.g., ['b', 't', 'x', 'y', 'p'].
    :type stack_ret_order: List[str]
    :param stack_tensors: Whether to stack the tensors into a single tensor (true) or return them as a tuple of tensors (false).
    :type stack_tensors: bool
    :param dtypes: Optional dictionary to specify the data types for each tensor.
    :type dtypes: Dict[str, torch.dtype], optional
    :param device: The device to move the tensor(s) to.
    :type device: torch.device
    :return: The tensor representing the events, stacked in the specified order.
    :rtype: torch.Tensor
    """
    # Accepts a structured numpy array of events (columns are 't', 'x', 'y', and 'p') and converts it to a tensor stack
    tensor_list = []
    cur_device = "cpu" if stack_tensors else device
    # ASSUMING THAT THE USER HAS CHECKED IF TORCH.CUDA.IS_AVAILABLE() IS TRUE PRIOR TO CALLING THIS FUNCTION
    for c in stack_ret_order:
        if c == 'b':
            tens_to_append = (torch.zeros(len(events)).to(torch.int64) + batch_index).to(cur_device)
        else:
            tens_to_append = torch.tensor(events[c].copy()).to(cur_device) #.copy() is needed in order to maintain contiguous locations for numpy str. arr.
        if (dtypes is not None) and (c in dtypes):
            tens_to_append = tens_to_append.type(dtypes[c])
        if c == 'p' and transform_polarity:
            tens_to_append = (tens_to_append * (tens_to_append + 1)) - 1
        tensor_list.append(tens_to_append)
    if stack_tensors:
        return torch.stack(tensor_list, dim=1).to(device)
    return tuple(tensor_list)

# 2
def events_tensor_batch_to_img(events: torch.Tensor, batch_size: int, height: int, width: int, time_surface_mode: str="exponential_decay_sum", \
                               separate_polarity: bool=False, displacement_mode: bool=False, \
                               window_size: int=5, decay_factor: float=1e9, prev_dict=None, \
                               offset: float=0, frequency: float=60, tolerance: float=100, penalty: float=0, \
                               time_unit: float=TimeUnit.MICROSECONDS, stack_order: List[str]=['b', 't', 'x', 'y', 'p']) -> torch.Tensor:
    #! ADD A NORMALIZE PARAMETER?? (should be universal across all time surface modes)
    #! -> divide entire time surface by maximum value; nope, instead total number of cycles in original events tensor
    # kernel: str="gaussian",
    # ALREADY DONE -> CAN MAYBE BE OPTIMIZED BY ADDING A `running_surface` PARAMETER INSTEAD OF HAVING `start_times`` = [0, 0, 0, ...] AND `durations` = [d, 2*d, 3*d, ...]
    """
    Convert a batch of events to an image.

    This function converts a tensor batch of events into an image based on the specified time surface representation mode.
    It can handle both separate polarities (2 separate channels) and combined polarity (1 channel) modes.
    Additionally, it will return the time surface within a displacement window if `displacement_mode` is True.

    :param events: The tensor batch of events.
    :type events: torch.Tensor
    :param batch_size: The number of events in the batch.
    :type batch_size: int
    :param height: The height of the event camera and the resulting image.
    :type height: int
    :param width: The width of the event camera and the resulting image.
    :type width: int
    :param time_surface_mode: The mode for generating the time surface: one of "count", "average", "average_absolute", "exponential_decay_sum", "exponential_decay_max," "most_recent," or "frequency."
    :type time_surface_mode: str
    :param separate_polarity: Whether to separate positive and negative polarities into different images.
    :type separate_polarity: bool
    :param displacement_mode: Whether to return the time surface within a displacement window around the final event.
    :type displacement_mode: bool
    :param window_size: The size of the displacement window. Must be an odd number.
    :type window_size: int
    :param decay_factor: The decay factor for the exponential decay mode.
    :type decay_factor: float
    :param prev_dict: A dictionary containing previous cumulative surfaces and metadata, used for running cumulative surface computation.
    :type prev_dict: dict, optional
    :param offset: The offset for the frequency mode, in the same time units as the event timestamps.
    :type offset: float
    :param frequency: The frequency for the frequency mode, in cycles per second (Hz).
    :type frequency: float
    :param tolerance: The tolerance for determining the frequency kernel width.
    :type tolerance: float
    :param penalty: The penalty value for events outside the frequency kernel.
    :type penalty: float
    :param time_unit: The unit of time for the event timestamps.
    :type time_unit: float
    :param stack_order: The order of dimensions in the events tensor, e.g., ['b', 't', 'x', 'y', 'p'].
    :type stack_order: List[str]
    :return: A dictionary with the resulting image tensor(s) based on the time surface mode, and metadata for cumulative surfaces.
    :rtype: dict
    """
    
    # dictionary keys
    # p_final_t, n_final_t, p_num_events, n_num_events, running_p_num_events, running_n_num_events
    # final_t, num_events, running_num_events
    # ret_img_tensor
    # running_time_surface

    if window_size % 2 == 0:
        window_size += 1
    # img_tensor = torch.zeros(height, width).type(torch.float32).to(events.device) # H, W
    param_ind = {stack_order[ind] : ind for ind in range(len(stack_order))} # for default case, stack_order = {'b': 0, 't': 1, 'x': 2, 'y': 3, 'p': 4}
    events_p = events[:batch_size, param_ind['p']].squeeze().type(torch.float32)
    running_cumulative_surface = prev_dict is not None

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

    # gaussian, epanechnikov, uniform, triangular, custom?
            # assuming that the unit of frequency is cycles/second (Hz)
    kernel_func_dict = {"gaussian": (lambda x: torch.exp(-0.5 * (x ** 2))),
                        "uniform": (lambda weight: weight),
                        "epanechnikov": (lambda x: (1 - (x ** 2)))} # need a normalization parameter to ensure that integral is 1
    def gen_freq_time_surface_vals(events_t: torch.Tensor) -> torch.Tensor:
        # obviously, offset should be in the same time units as events_t timestamp values
        # frequency should be in Hz (cycles/second)
        cycles_frac_tens = torch.frac((events_t - offset) * time_constant * frequency)
        delta_cycle_frac = (frequency / tolerance) * 0.5

        within_left_kernel_mask_positive = (cycles_frac_tens >= 0) & (cycles_frac_tens <= delta_cycle_frac)
        within_right_kernel_mask_positive = (cycles_frac_tens >= 0) & (cycles_frac_tens > (1 - delta_cycle_frac))
        within_left_kernel_mask_negative = (cycles_frac_tens < 0) & (cycles_frac_tens <= (delta_cycle_frac - 1))
        within_right_kernel_mask_negative = (cycles_frac_tens < 0) & (cycles_frac_tens > (-delta_cycle_frac))
        
        # "cycle domain" and not "time domain" # div by 3 (99.7% z-score 3)
        events_t[within_left_kernel_mask_positive] = kernel_func_dict['gaussian'](cycles_frac_tens[within_left_kernel_mask_positive] / (delta_cycle_frac / 3))
        events_t[within_right_kernel_mask_positive] = kernel_func_dict['gaussian']((cycles_frac_tens[within_right_kernel_mask_positive] - 1) / (delta_cycle_frac / 3))
        # cycles_frac_tens - (-1) = cycles_frac_tens + 1
        events_t[within_left_kernel_mask_negative] = kernel_func_dict['gaussian']((cycles_frac_tens[within_left_kernel_mask_negative] + 1) / (delta_cycle_frac / 3))
        events_t[within_right_kernel_mask_negative] = kernel_func_dict['gaussian'](cycles_frac_tens[within_right_kernel_mask_negative] / (delta_cycle_frac / 3))
        # time_delta = 1 / tolerance
        events_t[~(within_left_kernel_mask_positive | within_right_kernel_mask_positive \
                    | within_left_kernel_mask_negative | within_right_kernel_mask_negative)] = penalty
        return events_t

    if separate_polarity:
        if running_cumulative_surface:
            img_tensor = prev_dict['running_time_surface']
        else:
            img_tensor = torch.zeros(2 , height, width).to(events.device)
        p_mask = events_p == 1
        p_indices = p_mask.nonzero().squeeze()
        n_indices = (p_mask == False).nonzero().squeeze()
        # positive_events_x = events[p_indices, param_ind['x']].squeeze().type(torch.int64)
        # positive_events_y = events[p_indices, param_ind['y']].squeeze().type(torch.int64)
        positive_linear_indices = (events[p_indices, param_ind['y']].squeeze().type(torch.int64) * width) + events[p_indices, param_ind['x']].squeeze().type(torch.int64)
        positive_events_t = events[p_indices, param_ind['t']].squeeze()
        negative_linear_indices = (events[n_indices, param_ind['y']].squeeze().type(torch.int64) * width) + events[n_indices, param_ind['x']].squeeze().type(torch.int64)
        negative_events_t = events[n_indices, param_ind['t']].squeeze()
        p_event_img_index = 1
        n_event_img_index = 0
        final_time_positive_bin = positive_events_t[-1]
        final_time_negative_bin = negative_events_t[-1]
        num_events_positive = p_indices.size(0)
        num_events_negative = n_indices.size(0)

        if time_surface_mode == "count": # no change needed for running_cumulative_surface
            img_tensor = img_tensor.type(torch.int64)
            img_tensor[p_event_img_index].view(-1).scatter_add_(dim=0, index=positive_linear_indices, src=torch.ones(num_events_positive, dtype=torch.int64, device=img_tensor.device))
            img_tensor[n_event_img_index].view(-1).scatter_add_(dim=0, index=negative_linear_indices, src=torch.ones(num_events_negative, dtype=torch.int64, device=img_tensor.device))
        elif time_surface_mode == "average" or time_surface_mode == "average_absolute":
            mult_factor = 1 if time_surface_mode == "average" else -1
            img_tensor = img_tensor.type(torch.float32)
            if running_cumulative_surface:
                img_tensor[p_event_img_index] *= prev_dict['running_p_num_events']
                img_tensor[n_event_img_index] *= prev_dict['running_n_num_events']
            img_tensor[p_event_img_index].view(-1).scatter_add_(dim=0, index=positive_linear_indices, src=torch.tensor([1], dtype=torch.float32).repeat(p_indices.size(0)).to(img_tensor.device))
            img_tensor[n_event_img_index].view(-1).scatter_add_(dim=0, index=negative_linear_indices, src=torch.tensor([-1 * mult_factor], dtype=torch.float32).repeat(n_indices.size(0)).to(img_tensor.device))
            if running_cumulative_surface:
                img_tensor[p_event_img_index] = img_tensor[p_event_img_index] / (num_events_positive + prev_dict['running_p_num_events'])
                img_tensor[n_event_img_index] = img_tensor[n_event_img_index] / (num_events_negative + prev_dict['running_n_num_events'])
            else:
                img_tensor[p_event_img_index] = img_tensor[p_event_img_index] / num_events_positive
                img_tensor[n_event_img_index] = img_tensor[n_event_img_index] / num_events_negative
        elif time_surface_mode == "exponential_decay_sum":
            img_tensor = img_tensor.type(torch.float32)

            if running_cumulative_surface:
                img_tensor[p_event_img_index] *= math.exp((prev_dict['p_final_t'] - final_time_positive_bin) / decay_factor)
                img_tensor[n_event_img_index] *= math.exp((prev_dict['n_final_t'] - final_time_negative_bin) / decay_factor)

            positive_time_surface_vals =  torch.exp((positive_events_t - final_time_positive_bin) / decay_factor)
            positive_time_surface_vals = positive_time_surface_vals.type(torch.float32).to(events.device)
            img_tensor[p_event_img_index].view(-1).scatter_add_(dim=0, index=positive_linear_indices, src=positive_time_surface_vals)

            negative_time_surface_vals =  torch.exp((negative_events_t - final_time_negative_bin) / decay_factor)
            negative_time_surface_vals = negative_time_surface_vals.type(torch.float32).to(events.device)
            img_tensor[n_event_img_index].view(-1).scatter_add_(dim=0, index=negative_linear_indices, src=negative_time_surface_vals)
        elif time_surface_mode == "most_recent": # no change needed for running_cumulative_surface
            img_tensor = img_tensor.type(events.dtype)
            img_tensor[p_event_img_index].view(-1).scatter_(dim=0, index=positive_linear_indices, src=positive_events_t)
            img_tensor[n_event_img_index].view(-1).scatter_(dim=0, index=negative_linear_indices, src=negative_events_t)
        elif time_surface_mode == "exponential_decay_max":
            if running_cumulative_surface:
                img_tensor = img_tensor.to(torch.float32)
                img_tensor[p_event_img_index] *= math.exp((prev_dict['p_final_t'] - final_time_positive_bin) / decay_factor)
                img_tensor[n_event_img_index] *= math.exp((prev_dict['n_final_t'] - final_time_negative_bin) / decay_factor)
                
                new_events_img_tensor = -torch.ones((2, height, width), dtype=events.dtype, device=img_tensor.device) #! all -1s initialization
                new_events_img_tensor[p_event_img_index].view(-1).scatter_(dim=0, index=positive_linear_indices, src=positive_events_t)
                new_events_img_tensor[n_event_img_index].view(-1).scatter_(dim=0, index=negative_linear_indices, src=negative_events_t)

                new_events_mask = new_events_img_tensor != -1
                new_events_img_tensor = new_events_img_tensor.to(torch.float32)

                new_events_img_tensor[p_event_img_index] = torch.exp((new_events_img_tensor[p_event_img_index] - final_time_positive_bin) / decay_factor).to(new_events_img_tensor.device)
                new_events_img_tensor[n_event_img_index] = torch.exp((new_events_img_tensor[n_event_img_index] - final_time_negative_bin) / decay_factor).to(new_events_img_tensor.device)

                img_tensor[new_events_mask] = new_events_img_tensor[new_events_mask]
            else:
                img_tensor = img_tensor.to(events.dtype)
                img_tensor[p_event_img_index].view(-1).scatter_(dim=0, index=positive_linear_indices, src=positive_events_t)
                img_tensor[n_event_img_index].view(-1).scatter_(dim=0, index=negative_linear_indices, src=negative_events_t)

                img_tensor = img_tensor.to(torch.float32)

                img_tensor[p_event_img_index] = torch.exp((img_tensor[p_event_img_index] - final_time_positive_bin) / decay_factor).to(img_tensor.device)
                img_tensor[n_event_img_index] = torch.exp((img_tensor[n_event_img_index] - final_time_negative_bin) / decay_factor).to(img_tensor.device)
        elif time_surface_mode == "frequency":
            # Scratchwork --> optimal to have it in cycle domain; otherwise gaussian becomes too stretched out and the values close to the
            # "lattice"/whole number cycle does not get enough emphasis (weight)
            # period = 1 / frequency
            # x = (cycles_frac_tens[within_left_kernel_mask] * (period)) / (delta_cycle_frac * period / 3)
            # sigma = (delta_cycle_frac * period / 3)

            positive_events_t = positive_events_t.to(torch.float32)
            positive_time_surface_vals =  gen_freq_time_surface_vals(positive_events_t)
            positive_time_surface_vals = positive_time_surface_vals.type(torch.float32).to(events.device)
            img_tensor[p_event_img_index].view(-1).scatter_add_(dim=0, index=positive_linear_indices, src=positive_time_surface_vals)

            negative_events_t = negative_events_t.to(torch.float32)
            negative_time_surface_vals = gen_freq_time_surface_vals(negative_events_t)
            negative_time_surface_vals = negative_time_surface_vals.type(torch.float32).to(events.device)
            img_tensor[n_event_img_index].view(-1).scatter_add_(dim=0, index=negative_linear_indices, src=negative_time_surface_vals)
        else:
            raise ValueError(f"Time surface mode \"{time_surface_mode}\" not supported.")
        # elif time_surface_mode == "HATS":
        #     pass
    else:
        # events_x = events[:batch_size, param_ind['x']].squeeze().type(torch.int64) # must be converted to torch.int64, otherwise there is observed integer overflow for scatter add (negative dim is out of bounds)
        # events_y = events[:batch_size, param_ind['y']].squeeze().type(torch.int64) # scatter_add expects `index` tensor to be in int64
        if running_cumulative_surface:
            img_tensor = prev_dict['running_time_surface']
        else:
            img_tensor = torch.zeros(1 , height, width).to(events.device)
        events_t = events[:batch_size, param_ind['t']].squeeze()
        num_events = events_t.size(0)
        final_time_in_bin = events_t[-1]
        linear_event_indices = (events[:batch_size, param_ind['y']].squeeze().type(torch.int64) * width) + events[:batch_size, param_ind['x']].squeeze().type(torch.int64)
        
        if time_surface_mode == "count": # no change needed for running_cumulative_surface
            img_tensor = img_tensor.type(torch.int64)
            img_tensor[0].view(-1).scatter_add_(dim=0, index=linear_event_indices, src=torch.ones(num_events, dtype=torch.int64, device=img_tensor.device))
        elif time_surface_mode == "average" or time_surface_mode == "average_absolute":
            img_tensor = img_tensor.type(torch.float32)
            if running_cumulative_surface:
                img_tensor[0] *= prev_dict['running_num_events']
            img_tensor[0].view(-1).scatter_add_(dim=0, index=linear_event_indices, src=events_p if time_surface_mode == "average" else (events_p * events_p))
            if running_cumulative_surface:
                img_tensor[0] = img_tensor[0] / (num_events + prev_dict['running_num_events'])
            else:
                img_tensor[0] = img_tensor[0] / num_events
        elif time_surface_mode == "exponential_decay_sum":
            img_tensor = img_tensor.type(torch.float32)

            if running_cumulative_surface:
                img_tensor[0] *= math.exp((prev_dict['final_t'] - final_time_in_bin) / decay_factor)

            #! time_surface_pixel_vals = events_p * torch.exp((events_t - final_time_in_bin) / decay_factor) #! multiplying by the polarities as well (signed exp sum)
            time_surface_pixel_vals = torch.exp((events_t - final_time_in_bin) / decay_factor)
            time_surface_pixel_vals = time_surface_pixel_vals.type(torch.float32).to(events.device)
            img_tensor[0].view(-1).scatter_add_(dim=0, index=linear_event_indices, src=time_surface_pixel_vals)
        elif time_surface_mode == "most_recent": # no change needed for running_cumulative_surface
            img_tensor = img_tensor.type(events_t.dtype)
            img_tensor[0].view(-1).scatter_(dim=0, index=linear_event_indices, src=events_t)
        elif time_surface_mode == "exponential_decay_max":
            if running_cumulative_surface:
                img_tensor = img_tensor.to(torch.float32)
                img_tensor[0] *= math.exp((prev_dict['final_t'] - final_time_in_bin) / decay_factor)
                
                new_events_img_tensor = -torch.ones((1 , height, width), dtype=events.dtype, device=img_tensor.device) #! all -1s initialization
                new_events_img_tensor[0].view(-1).scatter_(dim=0, index=linear_event_indices, src=events_t)

                new_events_mask = new_events_img_tensor != -1
                new_events_img_tensor = new_events_img_tensor.to(torch.float32)

                new_events_img_tensor[0] = torch.exp((new_events_img_tensor[0] - final_time_in_bin) / decay_factor).to(new_events_img_tensor.device)

                img_tensor[new_events_mask] = new_events_img_tensor[new_events_mask]
            else:
                img_tensor = img_tensor.to(events.dtype)
                img_tensor[0].view(-1).scatter_(dim=0, index=linear_event_indices, src=events_t)
                img_tensor = img_tensor.to(torch.float32)
                img_tensor[0] = torch.exp((img_tensor[0] - final_time_in_bin) / decay_factor).to(img_tensor.device)
        elif time_surface_mode == "frequency":
            events_t = events_t.to(torch.float32)
            time_surface_pixel_vals = gen_freq_time_surface_vals(events_t)
            time_surface_pixel_vals = time_surface_pixel_vals.type(torch.float32).to(events.device)
            img_tensor[0].view(-1).scatter_add_(dim=0, index=linear_event_indices, src=time_surface_pixel_vals)
        else:
            raise ValueError(f"Time surface mode \"{time_surface_mode}\" not supported.")
        # elif time_surface_mode == "HATS":
        #     pass

    if displacement_mode:

        def get_window(cur_X, cur_Y):
            window_radius = int(window_size / 2)

            origin_Y_ret_real = cur_Y - window_radius
            origin_X_ret_real = cur_X - window_radius

            min_Y_ret = max(0, origin_Y_ret_real) - origin_Y_ret_real
            min_X_ret = max(0, origin_X_ret_real) - origin_X_ret_real

            max_Y_ret = (min(height - 1, final_event_Y + window_radius) - origin_Y_ret_real) + 1
            max_X_ret = (min(width - 1, final_event_X + window_radius) - origin_X_ret_real) + 1

            return min_X_ret, min_Y_ret, max_X_ret, max_Y_ret

        if not separate_polarity:
            ret_tens = torch.zeros((1, window_size, window_size), dtype=img_tensor.dtype, device=img_tensor.device)

            final_event_X = int(events[-1, param_ind['x']])
            final_event_Y = int(events[-1, param_ind['y']])
            
            min_X_ret, min_Y_ret, max_X_ret, max_Y_ret = get_window(final_event_X, final_event_Y)
            
            min_Y_tens = max(0, final_event_Y - int(window_size / 2))
            min_X_tens = max(0, final_event_X - int(window_size / 2))
            max_Y_tens = final_event_Y + int(window_size / 2) + 1
            max_X_tens = final_event_X + int(window_size / 2) + 1

            ret_tens[:, min_Y_ret:max_Y_ret, min_X_ret:max_X_ret] += img_tensor[:, min_Y_tens:max_Y_tens, min_X_tens:max_X_tens]
            # return ret_tens
        else:
            ret_tens = torch.zeros((2, window_size, window_size), dtype=img_tensor.dtype, device=img_tensor.device)
            
            pol_indices = [p_indices, n_indices]
            img_indices = [p_event_img_index, n_event_img_index]

            for i in range(len(pol_indices)):
                final_event_X = int(events[int(pol_indices[i][-1]), param_ind['x']])
                final_event_Y = int(events[int(pol_indices[i][-1]), param_ind['y']])
                
                min_X_ret, min_Y_ret, max_X_ret, max_Y_ret = get_window(final_event_X, final_event_Y)
                
                min_Y_tens = max(0, final_event_Y - int(window_size / 2))
                min_X_tens = max(0, final_event_X - int(window_size / 2))
                max_Y_tens = final_event_Y + int(window_size / 2) + 1
                max_X_tens = final_event_X + int(window_size / 2) + 1

                ret_tens[img_indices[i], min_Y_ret:max_Y_ret, min_X_ret:max_X_ret] += img_tensor[img_indices[i], min_Y_tens:max_Y_tens, min_X_tens:max_X_tens]
            # return ret_tens
    else:
        ret_tens = img_tensor
    
    ret_tens = ret_tens.to(events.device)
    img_tensor = img_tensor.to(events.device)

    # dictionary keys
    # p_final_t, n_final_t, p_num_events, n_num_events, running_p_num_events, running_n_num_events
    # final_t, num_events, running_num_events
    # ret_img_tensor
    # running_time_surface
    
    ret_dict = {}
    ret_dict['ret_img_tensor'] = ret_tens
    ret_dict['running_time_surface'] = img_tensor
    if separate_polarity:
        ret_dict['p_final_t'] = final_time_positive_bin
        ret_dict['n_final_t'] = final_time_negative_bin
        ret_dict['p_num_events'] = num_events_positive
        ret_dict['n_num_events'] = num_events_negative
        ret_dict['running_p_num_events'] = num_events_positive + (prev_dict['running_p_num_events'] if running_cumulative_surface else 0)
        ret_dict['running_n_num_events'] = num_events_negative + (prev_dict['running_n_num_events'] if running_cumulative_surface else 0)
    else:
        ret_dict['final_t'] = final_time_in_bin
        ret_dict['num_events'] = num_events
        ret_dict['running_num_events'] = num_events + (prev_dict['running_num_events'] if running_cumulative_surface else 0)
    
    return ret_dict # C, H, W

# 3
def events_tensor_batch_to_vol(events: torch.Tensor, total_batch_size: int, height: int, width: int, start_times: Union[List, np.ndarray], durations: Union[List, np.ndarray], nbins: int, \
                               time_surface_mode: str="exponential_decay_sum", displacement_mode: bool=False, window_size: int=5, \
                               separate_polarity: bool=False, decay_factor: float=1e9, running_cumulative_surface: bool=False, \
                               freq_offset: Union[int, float]=0, frequency: float = 60, tolerance: float=100, penalty: float=0, time_unit: Union[str, TimeUnit] = TimeUnit.MICROSECONDS, 
                               stack_order: List[str]=['b', 't', 'x', 'y', 'p'], ret_generator: bool=False) -> Union[torch.Tensor, Generator[torch.Tensor, None, None]]:
    # ALREADY DONE -> CAN MAYBE BE OPTIMIZED BY ADDING A `running_surface` PARAMETER INSTEAD OF HAVING `start_times`` = [0, 0, 0, ...] AND `durations` = [d, 2*d, 3*d, ...]
    """
    Convert a batch of events to a volumetric representation.

    This function converts a tensor batch of events into a 3D volume based on the specified time surface representation mode.
    It can handle both separate polarities (2 separate channels) and combined polarity (1 channel) modes.
    Additionally, it will return the time surface within a displacement window if `displacement_mode` is True.

    There should be `nbins` values each in both `start_times` and `durations`.
    
    For each time bin specified by `start_times` and its corresponding duration in `durations`, this method invokes the 
    `events_tensor_batch_to_img` method on the subset events in the `events` tensor falling in the time bin (s_i <= t_i <= s_i + d_i)
    and appends the result (of shape (C, H, W) ) to a tensor list. The tensors in this tensor list are stacked at the end are stacked
    along the 0th dimension, making the returned tensor have shape (`nbins`, C, H, W), where B is the number of time bins, C is the 
    number of channels (2 if `separate_polarity` is `True`, 1 if not), H is the height, and W is the width.

    :param events: The tensor batch of events.
    :type events: torch.Tensor
    :param total_batch_size: The total number of events in the batch.
    :type total_batch_size: int
    :param height: The height of the event camera and the resulting volume.
    :type height: int
    :param width: The width of the event camera and the resulting volume.
    :type width: int
    :param start_times: The start times for each time bin.
    :type start_times: Union[List, np.ndarray]
    :param durations: The durations for each time bin.
    :type durations: Union[List, np.ndarray]
    :param nbins: The number of bins for the temporal axis of the volume.
    :type nbins: int
    :param time_surface_mode: The mode for generating the time surface: one of "count", "average", "average_absolute", "exponential_decay_sum", "exponential_decay_max," "most_recent," or "frequency."
    :type time_surface_mode: str
    :param displacement_mode: Whether to return the time surface within a displacement window around the final event.
    :type displacement_mode: bool
    :param window_size: The size of the displacement window. Must be an odd number.
    :type window_size: int
    :param separate_polarity: Whether to separate positive and negative polarities into different volumes.
    :type separate_polarity: bool
    :param decay_factor: The decay factor for the exponential decay mode.
    :type decay_factor: float
    :param running_cumulative_surface: Whether to use running cumulative surface computation.
    :type running_cumulative_surface: bool
    :param freq_offset: The offset for the frequency mode, in the same time units as the event timestamps.
    :type freq_offset: Union[int, float]
    :param frequency: The frequency for the frequency mode, in cycles per second (Hz).
    :type frequency: float
    :param tolerance: The tolerance for determining the frequency kernel width.
    :type tolerance: float
    :param penalty: The penalty value for events outside the frequency kernel.
    :type penalty: float
    :param time_unit: The unit of time for the event timestamps.
    :type time_unit: Union[str, TimeUnit]
    :param stack_order: The order of dimensions in the events tensor, e.g., ['b', 't', 'x', 'y', 'p'].
    :type stack_order: List[str]
    :param ret_generator: Whether to return a generator yielding the image time surface representations one by one.
    :type ret_generator: bool
    :return: A 4D tensor of shape (`nbins`, `C`, `H`, `W`) where each of the `nbins` 3D tensors is an image formed by the aggregation of events within the time bins delineated by `start_times` and `durations`, or a generator yielding these tensors one by one.
    :rtype: Union[torch.Tensor, Generator[torch.Tensor, None, None]]
    """
    # dictionary keys
    # p_final_t, n_final_t, p_num_events, n_num_events, running_p_num_events, running_n_num_events
    # final_t, num_events, running_num_events
    # ret_img_tensor
    # running_time_surface
    
    if window_size % 2 == 0:
        window_size += 1
    # time_surface_mode = one of {"average", "most_recent", "exponential_decay", "HATS"}
    param_ind = {stack_order[ind] : ind for ind in range(len(stack_order))} # for default case, stack_order = {'b': 0, 't': 1, 'x': 2, 'y': 3, 'p': 4}
    events_t = events[:total_batch_size, param_ind['t']].squeeze()
    tensor_list = []
    
    print(f"Total number of iterations: {nbins}.")
    cur_ret_dict = None

    def ret_generator_vol():
        for i in tqdm((ind for ind in range(nbins))):
            # img_tensor = torch.zeros(height, width).type(torch.float32).to(events.device) # H, W
            start_time = start_times[i]
            end_time = start_time + durations[i]
            
            mask = (events_t >= start_time) & (events_t <= end_time)
            cur_ret_dict = events_tensor_batch_to_img(events=events[mask], batch_size=mask.nonzero().squeeze().size(0), height=height, width=width, time_surface_mode=time_surface_mode, \
                                                    displacement_mode=displacement_mode, window_size=window_size, \
                                                    prev_dict=(cur_ret_dict if running_cumulative_surface else None), \
                                                    separate_polarity=separate_polarity, decay_factor=decay_factor, stack_order=stack_order, \
                                                    offset=freq_offset, frequency=frequency, tolerance=tolerance, penalty=penalty, time_unit=time_unit)
            yield cur_ret_dict['ret_img_tensor']

    if ret_generator:
        return ret_generator_vol()
    else:
        for i in tqdm((ind for ind in range(nbins))):
            # img_tensor = torch.zeros(height, width).type(torch.float32).to(events.device) # H, W
            start_time = start_times[i]
            end_time = start_time + durations[i]
            
            mask = (events_t >= start_time) & (events_t <= end_time)
            cur_ret_dict = events_tensor_batch_to_img(events=events[mask], batch_size=mask.nonzero().squeeze().size(0), height=height, width=width, time_surface_mode=time_surface_mode, \
                                                    displacement_mode=displacement_mode, window_size=window_size, \
                                                    prev_dict=(cur_ret_dict if running_cumulative_surface else None), \
                                                    separate_polarity=separate_polarity, decay_factor=decay_factor, stack_order=stack_order, \
                                                    offset=freq_offset, frequency=frequency, tolerance=tolerance, penalty=penalty, time_unit=time_unit)
            
            tensor_list.append(cur_ret_dict['ret_img_tensor'])

        return torch.stack(tensor_list, dim=0).to(events.device) # nbins, C, H, W

# 4
def list_np_events_to_tensor(list_events: List[np.ndarray], transform_polarity: bool=True, device: torch.device="cpu", \
                             stack_ret_order: List[str]=['b', 't', 'x', 'y', 'p'], ret_dtype: torch.dtype=None) -> torch.Tensor:
    """
    Convert a list of structured NumPy arrays of events (representing different batches of events) to a single tensor (with or without a batch column).

    This function takes a list where each element is a NumPy array of events and converts it into a single PyTorch tensor.
    It supports optional polarity transformation and data type conversion.

    :param list_events: List of structured NumPy arrays of events.
    :type list_events: List[np.ndarray]
    :param transform_polarity: Whether to transform the polarity values.
    :type transform_polarity: bool
    :param device: The device to move the resulting tensor to.
    :type device: torch.device
    :param stack_ret_order: The order of dimensions in the resulting tensor, e.g., ['b', 't', 'x', 'y', 'p'].
    :type stack_ret_order: List[str]
    :param ret_dtype: Optional data type for the resulting tensor.
    :type ret_dtype: torch.dtype, optional
    :return: The combined tensor representing all events.
    :rtype: torch.Tensor
    """
    batch_index = 0
    tensor_list = []
    for event_batch in list_events: # each element in the list is a numpy array of events
        tensor_event = np_events_to_tensor(event_batch, stack_ret_order=stack_ret_order, stack_tensors=True, transform_polarity=transform_polarity, batch_index=batch_index, device="cpu")
        tensor_event.unsqueeze(0)
        tensor_list.append(tensor_event)
        batch_index += 1
    # b t x y p
    # 0 1 3 5 0
    # 0 4 6 3 1
    # ...
    # n 9 2 5 0
    if ret_dtype is not None:
        return torch.cat(tensor_list, dim=0).to(device).to(ret_dtype)
    return torch.cat(tensor_list, dim=0).to(device)

# 5
# may be able to speed up (using same method as "Optimizing Generation of Parameters" in Official SENPI Testing Notebook.ipynb)
# stack_ret_order = the order that the tensors are stacked in and the order that the "columns"/"keys" of the structured numpy arrays will have
def events_tensor_to_np_list(events: torch.Tensor, batch_size: int, group_batch_inds: bool=True, stack_ret_order: List[str]=['b', 't', 'x', 'y', 'p'], \
                             dtypes: Dict[str, np.dtype]=None) -> List[np.ndarray]: # each np.ndarray in the list is of `batch_size` length
    """
    Convert a tensor of events to a list of NumPy arrays (each element of the returned list represents a different batch of events).

    This function converts a PyTorch tensor of events into a list of structured NumPy arrays, each representing a batch of events.
    The resulting arrays have essentially the same structure as the original events tensor (they do not have a batch index column).

    :param events: The tensor of events to convert.
    :type events: torch.Tensor
    :param batch_size: The size of each batch in the resulting list of NumPy arrays.
    :type batch_size: int
    :param group_batch_inds: Whether to group events by batch index.
    :type group_batch_inds: bool
    :param stack_ret_order: The order of dimensions in the events tensor, e.g., ['b', 't', 'x', 'y', 'p'].
    :type stack_ret_order: List[str]
    :param dtypes: Optional dictionary to specify the data types for each column in the NumPy arrays.
    :type dtypes: Dict[str, np.dtype], optional
    :return: A list of NumPy arrays representing batches of events.
    :rtype: List[np.ndarray]
    """
    param_ind = {stack_ret_order[ind] : ind for ind in range(len(stack_ret_order))}
    col_names = stack_ret_order.copy()
    try:
        col_names.remove('b')
    except ValueError:
        pass
    equiv_np_dtype = get_np_dtype_from_torch_dtype(events.dtype)
    if dtypes is None:
        dtypes = {
                    't': equiv_np_dtype,
                    'x': equiv_np_dtype,
                    'y': equiv_np_dtype,
                    'p': equiv_np_dtype
                }
    np_list_events = []
    events = events.cpu().detach()
    if group_batch_inds:
        events = events[events[:, 0].sort()[1]]
    while events.size(0) > 0:
        current_batch_size = min(events.shape[0], batch_size)
        np_arr_events_batch = np.empty((current_batch_size,), dtype=[(col, dtypes[col]) if (col in dtypes) else (col, equiv_np_dtype) for col in col_names])
        for col in col_names:
            col_data = events[:batch_size, param_ind[col]]
            np_arr_events_batch[col] = col_data
        np_list_events.append(np_arr_events_batch)
        events = events[batch_size:]
    return np_list_events

# 6
def add_batch_col(events: torch.Tensor, batch_size: int, modify_original: bool=False) -> torch.Tensor:
    """
    Add a batch index column to the events tensor.

    This function adds a new column to the events tensor that represents the batch index for each event.
    The original tensor can either be modified in place or a new tensor can be returned.

    :param events: The tensor of events to modify.
    :type events: torch.Tensor
    :param batch_size: The size of each batch.
    :type batch_size: int
    :param modify_original: Whether to modify the original tensor in place.
    :type modify_original: bool
    :return: The tensor with the added batch index column.
    :rtype: torch.Tensor
    """
    num_batches = - (-events.size(0) // batch_size) # ceiling
    batch_col = torch.arange(num_batches).repeat_interleave(batch_size)
    batch_col = batch_col[:events.size(0)].unsqueeze(1)
    if not modify_original:
        return torch.cat((batch_col.to(events.device), events), dim=1)
    events.data = torch.cat((batch_col.to(events.device), events), dim=1)
    return events

# 7
# can include option as to whether to include the last events.size(0) % batch_size events in the returned list?
def partition_np_list_batch(events: np.ndarray, batch_size: int) -> List[np.ndarray]:
    """
    Partition a NumPy array into a list of batches.

    This function divides a NumPy array into smaller arrays (batches) of a specified size.

    :param events: The NumPy array to partition.
    :type events: np.ndarray
    :param batch_size: The size of each batch.
    :type batch_size: int
    :return: A list of NumPy arrays, each representing a batch.
    :rtype: List[np.ndarray]
    """
    np_arr_list = []
    while len(events) > 0:
        np_arr_list.append(events[:batch_size])
        events = events[batch_size:]
    return np_arr_list