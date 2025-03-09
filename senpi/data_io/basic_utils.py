import numpy as np
import pandas as pd
import torch
import h5py
import struct
from typing import List
from typing import Dict
from typing import Union
from typing import Tuple
from pathlib import Path
from dv import AedatFile
from dv import LegacyAedatFile
from tqdm import tqdm

def load_to_df(f, delim: str=',', order: List[str]=None, ret_order: List[str]=None, dtypes: Dict[str, np.dtype]=None, \
               transform_polarity: bool=True, format_version: str=None, sub_first_timestamp: bool=True) -> pd.DataFrame:
    """
    Load event data from a CSV-like, AEDAT (DAVIS output), HDF5 (Prophesee output), or RAW (Prophesee output; either EVT 2.0, 2.1, or 3.0) file into a Pandas DataFrame.

    This function reads event data from a file and converts it into a Pandas DataFrame. It supports various file formats (.csv, .txt, .hdf5, .raw, etc.) and column orders.
    The function also has options for transforming polarity values and specifying data types for columns.

    :param f: The path to the data file.
    :type f: str
    :param delim: The delimiter used in the data file (default is a comma).
    :type delim: str
    :param order: The order of columns in the file (default is ['t', 'x', 'y', 'p']).
    :type order: List[str]
    :param ret_order: The desired order of columns in the resulting DataFrame (default is None, which keeps the original order).
    :type ret_order: List[str], optional
    :param dtypes: A dictionary specifying data types for columns.
    :type dtypes: Dict[str, np.dtype], optional
    :param transform_polarity: Whether to transform the polarity values from 0 for OFF events and 1 for ON events to -1 for OFF events and 1 for ON events (default is True). Note: if the polarities are already represented by -1s and 1s, they will remain the same no matter what `transform_polarity` is set to.
    :type transform_polarity: bool
    :param format_version: The EVT format version of the inputted RAW file (one of "2.0", "2.1" or "3.0"). If not set, it will be inferred from the header of the RAW file.
    :type format_version: str
    :param sub_first_timestamp: Whether to subtract the first event timestamp from all of the recorded event timestamps when writing to the data frame.
    :type sub_first_timestamp: bool
    :return: A Pandas DataFrame containing the loaded data.
    :rtype: pd.DataFrame
    """
    def load_csv_like():
        try:
            if order is None:
                order2 = ['t', 'x', 'y', 'p']
                print(f"Warning: No order provided. Assuming default order {order2}.")
            else:
                order2 = order
            if delim.isspace():
                data_df = pd.read_csv(f, names=order2, delim_whitespace=True, dtype=dtypes)
            else:
                data_df = pd.read_csv(f, names=order2, sep=delim, dtype=dtypes)
            return data_df
        except pd.errors.ParserError:
            print("Could not read CSV-like file into Pandas dataframe.")
            return None

    def load_hdf5():
        with h5py.File(f, "r") as file:
            if 'CD' not in file or 'events' not in file['CD']:
                print("HDF5 file does not contain expected datasets.")
                return None

            events_dataset = file['CD']['events']
            x = events_dataset['x']
            y = events_dataset['y']
            p = events_dataset['p']
            t = events_dataset['t']

            data = np.vstack((t, x, y, p)).T

            data_df = pd.DataFrame(data, columns=['t', 'x', 'y', 'p'])

            return data_df
    
    def load_aedat():
        try:
            print("Attempting to read file as an .aedat file of a recent file format version.")
            data_df = load_aedat_recent()
        except Exception:
            print("Defaulting to old, legacy .aedat file reading method.")
            data_df = load_aedat_old()
        return data_df
    
    def load_aedat_recent():
        with AedatFile(f) as recent_f:
            # list all the names of streams in the file
            # print(recent_f.names)
            events = []
            # loop through the "events" stream
            for e in tqdm(recent_f['events']):
                events.append([e.timestamp, e.x, e.y, e.polarity])
            # loop through the "frames" stream
            # for frame in f['frames']:
            #     print(frame)
            #     cv2.imshow('out', frame.image)
            #     cv2.waitKey(1)
            data_df = pd.DataFrame(events, columns=['t', 'x', 'y', 'p'])
            return data_df
    
    def load_aedat_old():
        with LegacyAedatFile(f) as old_f:
            events = []
            for e in tqdm(old_f):
                events.append([e.timestamp, e.x, e.y, e.polarity])
        data_df = pd.DataFrame(events, columns=['t', 'x', 'y', 'p'])
        return data_df # could add a time cutoff
    
    def load_raw():
        header = {}
        with open(f, 'rb') as file:
            while True:
                cur_pos = file.tell()
                line = file.readline()
                # print(line)
                if not line.startswith(b'%'):
                    file.seek(cur_pos)
                    break
                line = line[1:].strip().decode('utf-8')
                if line == 'end':
                    break
                key, value = line.split(' ', 1)
                header[key] = value

            if format_version is not None:
                if (format_version == '2.0'):
                    print("Using EVT 2.0")
                    data_df = load_evt2(file)
                elif (format_version == '2.1'):
                    print("Using EVT 2.1")
                    data_df = load_evt21(file)
                elif (format_version == '3.0'):
                    print("Using EVT 3.0")
                    data_df = load_evt3(file)
                else:
                    print("Unsupported RAW file format.")
                    return None
            else:
                format_spec = header['format'] if 'format' in header else ""
                version_num = header['evt'] if 'evt' in header else ""
                if ('format' not in header) and ('evt' not in header):
                    print("RAW file header does not contain format information.")
                    print(f"Defaulting to EVT 2.0.")
                    format_spec = "EVT2"
                    version_num = "2.0"
                if ('EVT2.1' in format_spec) or ('EVT21' in format_spec) or (version_num.startswith('2.1')):
                    print("Using EVT 2.1 (found in RAW file header)")
                    data_df = load_evt21(file)
                elif ('EVT2' in format_spec) or (version_num.startswith('2')):
                    print("Using EVT 2.0 (found in RAW file header)")
                    data_df = load_evt2(file)
                elif ('EVT3' in format_spec) or (version_num.startswith('3')):
                    print("Using EVT 3.0 (found in RAW file header)")
                    data_df = load_evt3(file)
                else:
                    print("Unsupported RAW file format.")
                    return None
        return data_df
    
    def load_evt2(file):
        events = []
        first_time_base_set = False
        timestamp_high = 0
        # first_time_offset = None
        while True:
            data = file.read(4)
            if not data:
                break
            word = struct.unpack('<I', data)[0]
            event_type = (word >> 28) & 0x0F
            if event_type == 0x0 or event_type == 0x1:  # EVT2.0 CD events
                if first_time_base_set:
                    x = (word >> 11) & 0x07FF
                    y = word & 0x07FF
                    # polarity = word & 0x1
                    timestamp = (timestamp_high << 6) + ((word >> 22) & 0x03F) # will make timestamp a 64-bit int?
                    # if first_time_offset is None:
                    #     first_time_offset = timestamp
                    # events.append([timestamp - first_time_offset, x, y, (0 if event_type == 0x0 else 1)])
                    events.append([timestamp, x, y, (0 if event_type == 0x0 else 1)])
            elif event_type == 0x8:  # EVT2.0 TIME_HIGH
                timestamp_high = word & 0x0FFFFFFF
                first_time_base_set = True
            else:
                continue
            # NO SUPPORT FOR CONTINUED EVENT TYPES
        
        df_ret = pd.DataFrame(events, columns=['t', 'x', 'y', 'p'])

        return df_ret
    
    def load_evt21(file):
        events = []
        first_time_base_set = False
        timestamp_high = 0
        while True:
            data = file.read(8)
            if not data:
                break
            word = struct.unpack('<Q', data)[0]
            event_type = (word >> 60) & 0x0F
            if event_type == 0x0 or event_type == 0x1: # EVT_NEG or EVT_POS
                if first_time_base_set:
                    x = (word >> 43) & 0x07FF
                    y = (word >> 32) & 0x07FF
                    valid_vectorized = (word & 0x0FFFFFFFF)
                    timestamp = (timestamp_high << 6) + ((word >> 54) & 0x03F)
                    for x_offset in range(32):
                        if (valid_vectorized >> x_offset) & 0x01:
                            events.append([timestamp, x + x_offset, y, (0 if event_type == 0x0 else 1)])
            elif event_type == 0x8: # EVT_TIME_HIGH
                timestamp_high = (word >> 32) & 0x0FFFFFFF
                first_time_base_set = True
            else:
                continue
        
        df_ret = pd.DataFrame(events, columns=['t', 'x', 'y', 'p'])
        
        return df_ret

    def load_evt3(file):
        base_x = 0
        base_polarity = 0
        base_y = 0
        current_time = 0
        events = []

        while True:
            data = file.read(2)
            if not data:
                break
            word = struct.unpack('<H', data)[0]
            event_type = (word >> 12) & 0x0F
            
            if event_type == 0b0000:  # EVT_ADDR_Y
                base_y = word & 0x7FF
            elif event_type == 0b0010:  # EVT_ADDR_X
                x = word & 0x7FF
                polarity = (word >> 11) & 0x01
                events.append([current_time, x, base_y, polarity])
            elif event_type == 0b0011:  # VECT_BASE_X
                base_x = word & 0x7FF
                base_polarity = (word >> 11) & 0x01
            elif event_type == 0b0100:  # VECT_12
                valid = word & 0xFFF
                for j in range(12):
                    if (valid >> j) & 0x01:
                        events.append([current_time, base_x + j, base_y, base_polarity])
                base_x += 12
            elif event_type == 0b0101:  # VECT_8
                valid = word & 0xFF
                for j in range(8):
                    if (valid >> j) & 0x1:
                        events.append([current_time, base_x + j, base_y, base_polarity])
                base_x += 8
            elif event_type == 0b0110:  # EVT_TIME_LOW
                current_time = (current_time & 0xFFF000) | (word & 0x0FFF)
            elif event_type == 0b1000:  # EVT_TIME_HIGH
                current_time = (current_time & 0x0FFF) | ((word & 0x0FFF) << 12)
            # NO SUPPORT FOR CONTINUED EVENT TYPES

        df_ret = pd.DataFrame(events, columns=['t', 'x', 'y', 'p'])

        return df_ret
    
    df_ret = None

    if f.endswith(".raw"):
        df_ret = load_raw()
    elif f.endswith(".hdf5"):
        df_ret = load_hdf5()
    elif f.endswith(".aedat"):
        df_ret = load_aedat()
    else:
        df_ret = load_csv_like()

    if df_ret is None:
        return None
    
    if dtypes is not None:
        for col in df_ret.columns:
            if col in dtypes:
                df_ret[col] = df_ret[col].astype(dtypes[col])

    if transform_polarity:
        df_ret['p'] = (df_ret['p'] * (df_ret['p'] + 1)) - 1

    if sub_first_timestamp:
        df_ret['t'] -= df_ret.iloc[0]['t']
    
    df_ret = df_ret[ret_order] if ret_order is not None else df_ret

    return df_ret

def load_to_np_arr(f, delim: str=',', order: List[str]=None, ret_order: List[str]=None, dtypes: Dict[str, np.dtype]=None, \
                   transform_polarity: bool=True, format_version: str=None, sub_first_timestamp: bool=True) -> np.ndarray:
    """
    Load event data from a CSV-like, AEDAT (DAVIS output), HDF5 (Prophesee output), or RAW (Prophesee output; either EVT 2.0, 2.1, or 3.0) file into a NumPy structured array.

    This function reads event data from a file and converts it into a NumPy structured array. It supports various file formats (.csv, .txt, etc.) and column orders.
    The function also has options for transforming polarity values and specifying data types for columns. Invokes `load_to_df`.

    :param f: The path to the data file.
    :type f: str
    :param delim: The delimiter used in the data file (default is a comma).
    :type delim: str
    :param order: The order of columns in the file (default is ['t', 'x', 'y', 'p']).
    :type order: List[str]
    :param ret_order: The desired order of columns in the resulting NumPy array (default is None, which keeps the original order).
    :type ret_order: List[str], optional
    :param dtypes: A dictionary specifying data types for columns.
    :type dtypes: Dict[str, np.dtype], optional
    :param transform_polarity: Whether to transform the polarity values from 0 for OFF events and 1 for ON events to -1 for OFF events and 1 for ON events (default is True). Note: if the polarities are already represented by -1s and 1s, they will remain the same no matter what `transform_polarity` is set to.
    :type transform_polarity: bool
    :param format_version: The EVT format version of the inputted RAW file (one of "2.0", "2.1", or "3.0"). If not set, it will be inferred from the header of the RAW file.
    :type format_version: str
    :param sub_first_timestamp: Whether to subtract the first event timestamp from all of the recorded event timestamps when writing to the NumPy array.
    :type sub_first_timestamp: bool
    :return: A NumPy structured array containing the loaded data.
    :rtype: np.ndarray
    """
    # can use to_records?
    # order: the order that the data file's columns are in
    # ret_order: the order that you would like the np structured array's dtypes / "keys" / "indices" to be in
    data_df = load_to_df(f=f, delim=delim, order=order, ret_order=ret_order, dtypes=dtypes, transform_polarity=transform_polarity, \
                            format_version=format_version, sub_first_timestamp=sub_first_timestamp)
    if data_df is None:
        return None
    np_arr = np.empty((len(data_df.index),), dtype=[(col, data_df[col].dtype) for col in data_df.columns])
    for col in data_df.columns:
        np_arr[col] = data_df[col]
    return np_arr

def load_to_tensor(f, delim: str=',', order: List[str]=None, ret_order: List[str]=None, dtypes: Dict[str, torch.dtype]=None, \
                   stack_tensors: bool=True, device: torch.device="cpu", transform_polarity: bool=True, \
                   format_version: str=None, sub_first_timestamp: bool=True) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
    """
    Load event data from a CSV-like, AEDAT (DAVIS output), HDF5 (Prophesee output), or RAW (Prophesee output; either EVT 2.0, 2.1, or 3.0) file into a PyTorch tensor.

    This function reads event data from a file and converts it into a PyTorch Tensor. It supports various file formats (.csv, .txt, etc.), column orders, and options for stacking tensors.
    The function also has options for transforming polarity values and specifying data types for columns. Invokes `load_to_df`.

    :param f: The path to the data file.
    :type f: str
    :param delim: The delimiter used in the data file (default is a comma).
    :type delim: str
    :param order: The order of columns in the file, if applicable. If not inputted, default value will be set to ['t', 'x', 'y', 'p']).
    :type order: List[str]
    :param ret_order: The desired order of columns in the resulting Tensor (default is None, which keeps the original order).
    :type ret_order: List[str], optional
    :param dtypes: A dictionary specifying data types for columns.
    :type dtypes: Dict[str, torch.dtype], optional
    :param stack_tensors: Whether to stack the tensors into a single Tensor (default is True).
    :type stack_tensors: bool
    :param device: The device to which the Tensor should be moved (default is "cpu").
    :type device: torch.device
    :param transform_polarity: Whether to transform the polarity values from 0 for OFF events and 1 for ON events to -1 for OFF events and 1 for ON events (default is True). Note: if the polarities are already represented by -1s and 1s, they will remain the same no matter what `transform_polarity` is set to.
    :type transform_polarity: bool
    :param format_version: The EVT format version of the inputted RAW file (one of "2.0", "2.1", or "3.0"). If not set, it will be inferred from the header of the RAW file.
    :type format_version: str
    :param sub_first_timestamp: Whether to subtract the first event timestamp from all of the recorded event timestamps when writing to the torch Tensor.
    :type sub_first_timestamp: bool
    :return: A PyTorch Tensor containing the loaded data.
    :rtype: torch.Tensor
    """
    # can add normalize (z-score) functionality
    data_df = load_to_df(f=f, delim=delim, order=order, ret_order=ret_order, dtypes=None, transform_polarity=transform_polarity, \
                            format_version=format_version, sub_first_timestamp=sub_first_timestamp)
    if data_df is None:
        return None

    if stack_tensors:
        if dtypes is not None:
            return torch.tensor(data_df.to_numpy(), dtype=get_dominant_torch_dtype(dtypes.values()), device=device)
        return torch.tensor(data_df.to_numpy(), device=device)

    tensor_list = []
    for c in data_df.columns: # CAN REIMPLEMENT THIS TO DO SOMETHING LIKE TORCH_TENS(PD.TO_NUMPY_ND_ARR)?
        if dtypes is None:
            tens_to_append = torch.tensor(data_df[c]).to(device)
        else:
            cur_dtype = dtypes[c] if c in dtypes else None
            if cur_dtype is not None:
                tens_to_append = torch.tensor(data_df[c], dtype=dtypes[c], device=device)
            else:
                tens_to_append = torch.tensor(data_df[c]).to(device)
        tensor_list.append(tens_to_append)
    
    return tuple(tensor_list)

def save_df_to_csv(df: pd.DataFrame, f: Union[Path, str]=None, delim: str=',', save_order: List[str]=None, revert_polarity: bool=True, with_index: bool=False, with_header: bool=False):
    """
    Save a Pandas DataFrame to a CSV file.

    This function saves a Pandas DataFrame to a CSV file with options for column ordering, polarity transformation, and whether to include the index and header.

    :param df: The Pandas DataFrame to save.
    :type df: pd.DataFrame
    :param f: The path where the CSV file will be saved (default is "saved_event_data.csv").
    :type f: Union[Path, str], optional
    :param delim: The delimiter to use in the CSV file (default is a comma).
    :type delim: str
    :param save_order: The order of columns in the resulting CSV file (default is None, which keeps the original order).
    :type save_order: List[str], optional
    :param revert_polarity: Whether to revert the polarity values from -1 for OFF events and 1 for ON events to 0 for OFF events and 1 for ON events (default is True). Note: if the polarities are already represented by 0s and 1s, they will remain the same no matter what `revert_polarity` is set to.
    :type revert_polarity: bool
    :param with_index: Whether to include the index in the CSV file (default is False).
    :type with_index: bool
    :param with_header: Whether to include the header in the CSV file (default is False).
    :type with_header: bool
    """
    if f is None:
        f = "saved_event_data.csv"
    if save_order is None:
        reordered_df = df
    else:
        reordered_df = df[save_order]
    if revert_polarity:
        reordered_df['p'] = (((reordered_df['p'] * 0.5) + 0.5) * reordered_df['p']).astype(reordered_df['p'].dtype) # modified so that it doesn't matter if polarity is already reverted
    reordered_df.to_csv(f, sep=delim, index=with_index, header=with_header)

def save_tensor_to_csv(events: torch.Tensor, f: Union[Path, str]=None, delim: str=',', order: List[str]=None, save_order: List[str]=None, revert_polarity: bool=True, with_index: bool=False, with_header: bool=False):
    """
    Save a PyTorch Tensor to a CSV file.

    This function converts a PyTorch Tensor to a Pandas DataFrame and saves it to a CSV file. It supports column ordering, polarity transformation, and options for including the index and header.

    :param events: The PyTorch Tensor to save.
    :type events: torch.Tensor
    :param f: The path where the CSV file will be saved (default is "saved_event_data.csv").
    :type f: Union[Path, str], optional
    :param delim: The delimiter to use in the CSV file (default is a comma).
    :type delim: str
    :param order: The order of columns in the resulting CSV file (default is None, which uses the order from the Tensor).
    :type order: List[str], optional
    :param save_order: The desired order of columns in the CSV file (default is None, which keeps the order from `order`).
    :type save_order: List[str], optional
    :param revert_polarity: Whether to revert the polarity values from -1 for OFF events and 1 for ON events to 0 for OFF events and 1 for ON events (default is True). Note: if the polarities are already represented by 0s and 1s, they will remain the same no matter what `revert_polarity` is set to.
    :type revert_polarity: bool
    :param with_index: Whether to include the index in the CSV file (default is False).
    :type with_index: bool
    :param with_header: Whether to include the header in the CSV file (default is False).
    :type with_header: bool
    """
    if f is None:
        f = "saved_event_data.csv"
    if order is None:
        order = ['t', 'x', 'y', 'p']
        print(f"Warning: No order provided. Assuming default order {order}.")
    data_df = pd.DataFrame(events.cpu().detach().numpy(), columns=order)
    save_df_to_csv(data_df, f, delim=delim, save_order=save_order, revert_polarity=revert_polarity, with_index=with_index, with_header=with_header)

def save_np_arr_to_csv(np_arr: pd.DataFrame, f: Union[Path, str]=None, delim: str=',', save_order: List[str]=None, revert_polarity: bool=True, with_index: bool=False, with_header: bool=False):
    """
    Save a NumPy structured array to a CSV file.

    This function converts a NumPy structured array to a Pandas DataFrame and saves it to a CSV file. It supports column ordering, polarity transformation, and options for including the index and header.

    :param np_arr: The NumPy structured array to save.
    :type np_arr: pd.DataFrame
    :param f: The path where the CSV file will be saved (default is "saved_event_data.csv").
    :type f: Union[Path, str], optional
    :param delim: The delimiter to use in the CSV file (default is a comma).
    :type delim: str
    :param save_order: The desired order of columns in the CSV file (default is None, which keeps the original order).
    :type save_order: List[str], optional
    :param revert_polarity: Whether to revert the polarity values from -1 for OFF events and 1 for ON events to 0 for OFF events and 1 for ON events (default is True). Note: if the polarities are already represented by 0s and 1s, they will remain the same no matter what `revert_polarity` is set to.
    :type revert_polarity: bool
    :param with_index: Whether to include the index in the CSV file (default is False).
    :type with_index: bool
    :param with_header: Whether to include the header in the CSV file (default is False).
    :type with_header: bool
    """
    if f is None:
        f = "saved_event_data.csv"
    data_df = pd.DataFrame(np_arr)
    save_df_to_csv(data_df, f, delim=delim, save_order=save_order, revert_polarity=revert_polarity, with_index=with_index, with_header=with_header)

def get_dominant_torch_dtype(dtypes_iter: List[torch.dtype]) -> torch.dtype:
    """
    Determine the dominant PyTorch data type from a list of data types.

    This function takes a list of PyTorch data types and determines the dominant data type based on the list.

    :param dtypes_iter: A list of PyTorch data types.
    :type dtypes_iter: List[torch.dtype]
    :return: The dominant PyTorch data type.
    :rtype: torch.dtype
    """
    temp_list = []
    for type_item in dtypes_iter:
        temp_list.append(torch.empty(1, dtype=type_item))
    return torch.stack(temp_list).dtype

def get_np_dtype_from_torch_dtype(torch_dtype: torch.dtype): # Add these to conversions.py?
    """
    Convert a PyTorch data type to a NumPy data type.

    This function converts a PyTorch data type to the corresponding NumPy data type.

    :param torch_dtype: The PyTorch data type to convert.
    :type torch_dtype: torch.dtype
    :return: The corresponding NumPy data type.
    :rtype: np.dtype
    """
    torch_temp = torch.empty(1, dtype=torch_dtype)
    return torch_temp.detach().cpu().numpy().dtype

def get_torch_dtype_from_np_dtype(np_dtype: np.dtype):
    """
    Convert a NumPy data type to a PyTorch data type.

    This function converts a NumPy data type to the corresponding PyTorch data type.

    :param np_dtype: The NumPy data type to convert.
    :type np_dtype: np.dtype
    :return: The corresponding PyTorch data type.
    :rtype: torch.dtype
    """
    np_temp = np.empty(1, dtype=np_dtype)
    return torch.tensor(np_temp).dtype