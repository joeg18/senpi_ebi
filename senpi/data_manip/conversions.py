import torch
import numpy as np
import pandas as pd

from typing import List, Union

# Add more general conversion functions (from TimeUnit A to TimeUnit B; another enum for exponent values)

def df_seconds_to_microseconds(df: pd.DataFrame, new_dtype: np.dtype=None):
    """
    Convert the time column of a DataFrame from seconds to microseconds.

    This function multiplies the time values in the 't' column of the DataFrame (in-place) by 1e6 to convert the time unit from seconds to microseconds. Optionally, it can also change the data type of the 't' column.

    :param df: The DataFrame containing the time data to convert.
    :type df: pd.DataFrame
    :param new_dtype: The new data type for the 't' column after conversion (default is None, which uses np.int64).
    :type new_dtype: np.dtype, optional
    :return: The DataFrame with the time column converted to microseconds.
    :rtype: pd.DataFrame
    """
    df['t'] *= 1e6
    if new_dtype is not None:
        df['t'] = df['t'].astype(new_dtype)
    else:
        df['t'] = df['t'].astype(np.int64)
    return df

def df_microseconds_to_seconds(df: pd.DataFrame, new_dtype: np.dtype=None):
    """
    Convert the time column of a DataFrame from microseconds to seconds.

    This function divides the time values in the 't' column of the DataFrame (in-place) by 1e6 to convert the time unit from microseconds to seconds. Optionally, it can also change the data type of the 't' column.

    :param df: The DataFrame containing the time data to convert.
    :type df: pd.DataFrame
    :param new_dtype: The new data type for the 't' column after conversion (default is None, which uses np.float64).
    :type new_dtype: np.dtype, optional
    :return: The DataFrame with the time column converted to seconds.
    :rtype: pd.DataFrame
    """
    if new_dtype is not None:
        df['t'] = df['t'].astype(new_dtype)
    else:
        df['t'] = df['t'].astype(np.float64)
    df['t'] /= 1e6
    return df

def event_stream_to_frame_vol(event_stream: torch.Tensor, num_frames: int, width: int, height: int, order: List[str]=None) -> torch.Tensor:
    """
    Convert a tensor containing a standard event stream ((N, 4) or (N, 5)) into a frame volume representation. 
    
    The output of this function will be a tensor with dimensions (N, C, H, W), where N is the number of frames (each frame corresponds to a single time step of events; if the event stream's
    timestamp measurements are in microseconds, then each frame corresponds to microsecond of events), C is the number of channels (set to 1), H is the frame height, and W is the
    frame width.

    :param event_stream: The PyTorch tensor containing the event stream to convert.
    :type event_stream: torch.Tensor
    :param num_frames: The number of frames in the returned volume (N).
    :type num_frames: int
    :param width: The width of each frame (W).
    :type width: int
    :param height: The height of each frame (H).
    :type height: int
    :param order: The order of the columns of the inputted events tensor. Defaults to ['t', 'x', 'y', 'p'].
    :type order: List[str], optional
    :return: A tensor containing the frame volume representation of the event stream.
    :rtype: torch.Tensor
    """

    if order is None:
        order = ['t', 'x', 'y', 'p']
        print(f"Warning: No order provided. Assuming default order {order}.")
    
    param_ind = {order[ind]: ind for ind in range(len(order))}

    frames = torch.zeros(num_frames, height, width)
    frames[event_stream[:, param_ind['t']].to(torch.int64), \
        event_stream[:, param_ind['y']].to(torch.int64), \
        event_stream[:, param_ind['x']].to(torch.int64)] = event_stream[:, param_ind['p']].to(torch.float32)
    
    frames = frames.unsqueeze(dim = 1) # Adding C = 1

    return frames

def frame_vol_to_event_stream(frames: torch.Tensor, ret_order: List[str]=['t', 'x', 'y', 'p']) -> torch.Tensor:
    """
    Convert a frame volume representation ((N, 1, H, W) or (N, H, W)) back into an event stream format (dimensions (N, 4)).

    The input to this function is a frame volume where each frame corresponds to a time step of events (e.g., each frame could represent a microsecond of events, depending on the 
    timestamp resolution in the original event stream). The function extracts the non-zero entries (events) from the volume and converts them back into an event stream representation.

    :param frames: The PyTorch tensor containing the frame volume to convert. Expected dimensions are (N, 1, H, W) or (N, H, W), where N is the number of frames, H is the frame height,
    and W is the frame width.
    :type frames: torch.Tensor
    :param ret_order: The desired order of the columns in the outputted event stream tensor. Defaults to ['t', 'x', 'y', 'p'].
    :type ret_order: List[str], optional
    :return: A tensor containing the event stream representation of the frame volume, with the columns ordered as specified by `ret_order`.
    :rtype: torch.Tensor
    """
    
    frames_squeezed_view = frames.squeeze()
    es_frames = frames_squeezed_view.nonzero()
    
    polarities_frames = frames_squeezed_view[es_frames[:, 0], es_frames[:, 1], es_frames[:, 2]]

    es_frames = torch.concat([es_frames, polarities_frames.view(-1, 1)], dim=1)
    
    current_order = {'t': 0, 'y': 1, 'x': 2, 'p': 3}
    
    ret_order_list = []
    for c in ret_order:
        ret_order_list.append(current_order[c])
    
    es_frames = es_frames[:, ret_order_list]

    return es_frames

# Conversion between all 1D stream data types
def to_1D_tensor(events, device = 'cuda' if torch.cuda.is_available else 'cpu'):
    """
    Convert event data to a 1D PyTorch tensor.

    Converts input event data from either pandas DataFrame or NumPy array format to a PyTorch tensor.
    If the input is already a tensor, it is returned unchanged.

    :param events: The event data to convert. Can be a pandas DataFrame, NumPy array, or PyTorch tensor.
    :type events: Union[pd.DataFrame, np.ndarray, torch.Tensor]
    :param device: The device to store the tensor on ('cuda' or 'cpu'). Defaults to 'cuda' if available, else 'cpu'.
    :type device: str
    :return: The event data as a PyTorch tensor on the specified device.
    :rtype: torch.Tensor
    """
    # check input type
    if isinstance(events, pd.DataFrame):
        if device == 'cuda':
            return torch.tensor(events.to_numpy(), device=device)
        else:
            return torch.tensor(events.to_numpy())
    if isinstance(events, np.ndarray):
        if device == 'cuda':
            return torch.from_numpy(events).to(device)  # mitigate issue on cpu only devices
        else:
            return torch.from_numpy(events)
    if isinstance(events, torch.Tensor):
        return events

def to_1D_dataframe(events, order = ['t', 'x', 'y', 'p']):
    """
    Convert event data to a pandas DataFrame.

    Converts input event data from either PyTorch tensor or NumPy array format to a pandas DataFrame.
    If the input is already a DataFrame, it is returned unchanged.

    :param events: The event data to convert. Can be a PyTorch tensor, NumPy array, or pandas DataFrame.
    :type events: Union[torch.Tensor, np.ndarray, pd.DataFrame]
    :param order: The column names for the DataFrame in order. Defaults to ['t', 'x', 'y', 'p'].
    :type order: List[str], optional
    :return: The event data as a pandas DataFrame with specified column names.
    :rtype: pd.DataFrame
    """
    if order is None:
                order = ['t', 'x', 'y', 'p']
    if isinstance(events, pd.DataFrame):
        return events
    if isinstance(events, np.ndarray):
        return pd.DataFrame(events, columns=order)
    if isinstance(events, torch.Tensor):
        return pd.DataFrame(events.detach().cpu().numpy(), columns=order)

def to_1D_numpy(events):
    """
    Convert event data to a NumPy array.

    Converts input event data from either PyTorch tensor or pandas DataFrame format to a NumPy array.
    If the input is already a NumPy array, it is returned unchanged.

    :param events: The event data to convert. Can be a PyTorch tensor, pandas DataFrame, or NumPy array.
    :type events: Union[torch.Tensor, pd.DataFrame, np.ndarray]
    :return: The event data as a NumPy array.
    :rtype: np.ndarray
    """
    if isinstance(events, pd.DataFrame):
        return events.to_numpy()
    if isinstance(events, np.ndarray):
        return events
    if isinstance(events, torch.Tensor):
        return events.detach().cpu().numpy()


    
