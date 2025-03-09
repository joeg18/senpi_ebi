# Synthetic Events Through Neural Processing and Integration (SENPI)

This repository contains code for SENPI, a PyTorch-based Python package that seamlessly enables the simulation of realistic event data, visualization, processing (algorithms, filters, denoising), and intensity frame reconstruction based on existing deep learning and computer vision literature.

Key features include:

- Converting photometric data to voltages to simulate various noises, and comparing log difference to determine events.
- Incorporation of common event-stream operations such as converting to time surface volume, statistical first-order filtering methods, and algorithmic reconstruction based on a continuous-time ODE.
- All operations, including event-stream conversions, statistical filtering, and reconstruction algorithms, are designed to preserve differentiability, supporting PyTorch’s autograd for seamless integration into neural network training and optimization.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Simulation](#simulation)
  - [Simulator](#simulator)
- [Data IO](#data-io)
  - [Basic Utils](#basic-utils-senpidata_iobasic_utils)
    - [`load_to_df`](#load_to_df)
    - [`load_to_np_arr`](#load_to_np_arr)
    - [`load_to_tensor`](#load_to_tensor)
    - [`save_df_to_csv`](#save_df_to_csv)
    - [`save_tensor_to_csv`](#save_tensor_to_csv)
    - [`save_np_arr_to_csv`](#save_np_arr_to_csv)
    - [`get_dominant_torch_dtype`](#get_dominant_torch_dtype)
    - [`get_np_dtype_from_torch_dtype`](#get_np_dtype_from_torch_dtype)
    - [`get_torch_dtype_from_np_dtype`](#get_torch_dtype_from_np_dtype)
- [Data Manipulation](#data-manipulation)
  - [Algorithms](#algorithms-senpidata_manipalgs)
    - [Alg Objects](#alg-objects)
    - [FlipXAlg Objects](#flipxalg-objects)
    - [FlipYAlg Objects](#flipyalg-objects)
    - [InvertPolarityAlg Objects](#invertpolarityalg-objects)
  - [Filters](#filters-senpidata_manipfilters)
    - [Filter Objects](#filter-objects)
    - [PolarityFilter Objects](#polarityfilter-objects)
    - [BAFilter Objects](#bafilter-objects)
    - [IEFilter Objects](#iefilter-objects)
    - [YNoiseFilter Objects](#ynoisefilter-objects)
  - [Conversions](#conversions-senpidata_manipconversions)
    - [`df_seconds_to_microseconds`](#df_seconds_to_microseconds)
    - [`df_microseconds_to_seconds`](#df_microseconds_to_seconds)
    - [`event_stream_to_frame_vol`](#event_stream_to_frame_vol)
    - [`frame_vol_to_event_stream`](#frame_vol_to_event_stream)
    - [`to_1D_tensor`](#to_1d_tensor)
    - [`to_1D_dataframe`](#to_1d_dataframe)
    - [`to_1D_numpy`](#to_1d_numpy)
  - [Preprocessing](#preprocessing-senpidata_manippreprocessing)
    - [`np_events_to_tensor`](#np_events_to_tensor)
    - [`events_tensor_batch_to_img`](#events_tensor_batch_to_img)
    - [`events_tensor_batch_to_vol`](#events_tensor_batch_to_vol)
    - [`list_np_events_to_tensor`](#list_np_events_to_tensor)
    - [`events_tensor_to_np_list`](#events_tensor_to_np_list)
    - [`add_batch_col`](#add_batch_col)
    - [`partition_np_list_batch`](#partition_np_list_batch)
  - [Computation](#computation-senpidata_manipcomputation)
    - [`has_parameter`](#has_parameter)
    - [SequentialCompute Objects](#sequentialcompute-objects)
- [Data Visualization](#data-visualization)
  - [Visualization](#visualization-senpidata_visvisualization)
    - [`plot_events`](#plot_events)
    - [`plot_time_surface`](#plot_time_surface)
- [Data Generation (Algorithmic Reconstruction)](#data-generation-algorithmic-reconstruction)
  - [Reconstruction](#reconstruction-senpidata_genreconstruction)
    - [ImgFramesFromEventsGenerator Objects](#imgframesfromeventsgenerator-objects)
      

## Installation

### Install with Anaconda

The installation requires [Anaconda3](https://www.anaconda.com/distribution/). You can create a new Anaconda environment with the required dependencies as follows (make sure to adapt the CUDA toolkit version according to your setup):

```bash
conda env create --name senpi --file=environment.yml
conda activate senpi
```

## Usage

Please refer to the "Official SENPI Event Processing Notebook.ipynb" and "Official SENPI Simulator Notebook.ipynb" files for examples of usage of all the functionalities provided by SENPI.

[comment]: <> (PROCESSING)
# SENPI Terminology and Data Format
## Terminology
**Binary Polarity Mode**: A dataset is said to be in *binary polarity mode* if the polarities take on values within the set {0, 1}. Otherwise, it is assumed that the polarities take on values within the set {-1, 1}.

**Order**: For functions and classes dealing with PyTorch tensors representing event streams, an `order` parameter, which is of type `List[str]` (list of strings), and is some permutation of `['t', 'x', 'y', 'p']` or `['b', 't', 'x', 'y', 'p']` must be passed in. The order of the elements in the `order` list should match the order of the columns in the tensor (`'t'` for timestamp, `'x'` for pixel x coordinate, `'y'` for pixel y coordinate, `'p'` for polarity, and `b` for batch number).

## Data Format
The expected data format for data structures representing an event stream is as follows. Supported data structures/containers include Pandas dataframes, NumPy structured arrays, and PyTorch tensors. The shape of these data structures is expected to be either (`N`, 4) or (`N`, 5), where `N` is the number of events in the event stream. The column names for the Pandas dataframes and the NumPy structured arrays should be some permutation of `['t', 'x', 'y', 'p']` or `['b', 't', 'x', 'y', 'p']` (`'t'` for timestamp, `'x'` for pixel x coordinate, `'y'` for pixel y coordinate, `'p'` for polarity, and `b` for batch number).

# Simulation

## Usage
The simulator takes in a tensor of photometric data with shape `[f, x, y]`, where `f` is the number of frames, and `x` and `y` are the XY coordinates respectively, then returns a tensor of generated events of shape `[n, 4]` where `n` is the number of generated events and the second dimension contains information about the events in `[t, x, y, p]` format. Optional parameter to return calculated event frames for ground truth. To generate events, first create a sim object, then pass the photometric data through the `forward()` function.
```python
es = senpi.sim.EventSimulator()
data = torch.load(path_to_data)
generated_events = sim.forward(data)
```

## Simulator

### `EventSimulator` class
Base class to generate synthetic event data from photometric frames. Takes input data as a torch Tensor of shape `[f,x,y]`, where `f` is the number of frames and `x,y` are the XY coorindates of the photometric image. Parameters and camera settings are changed in the `params.py` file. Returns a tensor of events in the shape of `[n, 4]`, where `n` is the number of events generated stored in `[t,x,y,p]` format.

**Parameters:**
- `refractory` (`int`): Number of microseconds between when a pixel fires off an event and when a pixel can fire off another event.
- `dt` (`int`): Time between each frame in microseconds
- `well_cap` (`int`): Maximum number of photons each pixel can detect before it becomes saturated
- `QE` (`float`): Quantum efficiency of all the pixels. Default `1`
- `pos_th` (`float`): Positive threshold to trigger a positive event
- `neg_th` (`float`): Negative threshold to trigger a negative event
- `sigma_pos` (`float`): Standard deviation of the positive threshold between pixels. Default is `.03`
- `sigma_neg` (`float`): Standard deviation of the negative threshold between pixels. Default is `.03`
- 'leak_rate' ('float'): Expected fraction of pixels that emit a false event during each time point due to leakage at the comparitor. Default is 0.1. Equal probability of positive or negative false event.
- 'sigma_leak' ('sigma_leak'): Standard deviation of the leakage rate between pixels. Default is 0.01.
- `hot_prob` (`float`): What fraction of pixels are "hot" and fire off erroneous events at a higher rate. Default 0.1.
- `hot_rate` (`float`): Expected fraction of hot pixels to produce a false event at each time point.
- `seed` (`int`): Seed for the random generators used in the simulator. Setting the same seed guarantees the same output with the same input frames
- 'v1' ('float'): Mean reference voltage at first time point. Option in forward method to replace this generated map with a user-defined map.
- 'sigma_v1' ('float'): Standard deviation in the generated v1 map.
- `device` (`str`): Device to run simulator on. Due to iterating over frames, gpu ('cuda') sees no particular speedup over cpu
- 'return_frame' (bool): Boolean flag to return calculated event frames (1) with event stream for GT inferencing or to return frames as an empty vector (0) to conserve memory.
- 'shot_noise' (bool): Boolean flag to apply effects of shot noise to the input photometric image (1) before determining the resulting events or not (0).
- 'sensor_noise' (bool): Boolean flag to add the effects of noise into the resulting event stream (1) or to solely return events caused by the photometric input (0)

### Class functions
#### **_init()**
```python
def __init__(self, 
          params: dict) -> None
```
Initializes an instance of the class and assigned internal attributes to track user-parameters, per-pixel variation, gain and internal clock.

**Inputs:**
- `params` (`Dict`): Dictionary containing parameters set in `params.py`

#### **_reset()**
```python
def _init(self,
          params: dict=[]) -> None
```
Resets all internal per-pixel variation maps to empty arrays. If a new params dictionary is passed, resets all internal properties.

**Parameters:**
- `params` (`Dict`): *Optional* Dictionary containing parameters set in `params.py`

#### **_generate_internal_grids()**
```python
def _generate_internal_grids(self,
          im_size: tensor) -> None
```
Uses the expected input 'x', 'y' image size to define new attributes containing the per pixel variation in key properties. These include:
- 'pos_map' ('tensor'): variational map in positive threshold.
- 'neg_map' ('tensor'): variational map in neg threshold.
- 'noise_map' ('tensor'): assigned false event rate at each pixel.
- 'hot_map' ('tensor'): binary map indicating location of assigned hot pixels in simulation.
- 'pos_map' ('tensor'): variational map in positive threshold.
- 'refractory_map' ('tensor'): map tracking local refractory time before pixel may fire another event.

**Parameters:**
- `im_size` (`tensor`): Expected (x,y) image size to generate correct size grids.

#### **_stack_to_events()**
```python
def _stack_to_events(self,
          stack: tensor) -> tensor
```
Uses an input stack of the form [f, x, y] to determine an output event stream. If return_frames is true, also returns the calculated event frames else returns an empty vector.

**Parameters:**
- `stack` (`tensor`): Photometric data in units of PHOTONS to be converted into events. Must be of shape [f, x, y].

**Returns:**
- 'events' ('tensor'): tensor of the size (n, 4) where n is the total number of calculated events. Each event contains 4D information of [t, x, y, p]. In raw output, the events are organized in n by time, location then polarity. (i.e. [0, 0, 0, 1] will be first if an event exists). 
- 'eframes' ('tensor'): frames containing the calculated events. Positive events are 1, negative events are -1 and all other pixels assume the value of 0. Of same shape as input.

#### **forward()**
```python
def _stack_to_events(self,
          stack: tensor,
          reset: bool = 1,
          v1: tensor = []) -> tensor
```
Uses an input stack of the form [f, x, y] to determine an output event stream. If return_frames is true, also returns the calculated event frames else returns an empty vector. Forward build on _stack_to_events by performing additional input size + type checks, checks internal properties for validity before running and gives the option to reset or perserve internal maps before running. It should be used as the primary functionality of EventSimulator.

**Parameters:**
- `stack` (`tensor`): Photometric data in units of PHOTONS to be converted into events. Must be of shape [f, x, y].
- `reset` (`Bool`): Choose to reset internal grids on run (True, default) or save currently generated grids on run (False). If set to False, forward checks if grids exist then generate if not.
- `v1` (`tensor` or 'empty'): If tensor: uses as first time point reference voltage. If empty: generates first time point reference voltage based on user params. 

**Returns:**
- 'events' ('tensor'): tensor of the size (n, 4) where n is the total number of calculated events. Each event contains 4D information of [t, x, y, p]. In raw output, the events are organized in n by time, location then polarity. (i.e. [0, 0, 0, 1] will be first if an event exists). 
- 'eframes' ('tensor'): frames containing the calculated events. Positive events are 1, negative events are -1 and all other pixels assume the value of 0. Of same shape as input.

### Simulator_utils functions

#### **display_total_events()**
Visualization function to show the log counts of the number of events a pixel has fired.

**Parameters:**
-`shape`: Shape of the video from which the events were generated. Should be larger than the greatest `x` and `y` in events
-`events` (`torch.Tensor`): List of events in `[t, x, y, p]` format.

#### **display_total_events()**
Visualization function to show the sum of the number of events a pixel has fired.

**Parameters:**
- `shape`: Shape of the video from which the events were generated. Should be larger than the greatest `x` and `y` in events
- `events` (`torch.Tensor`): List of events in `[t, x, y, p]` format.

<a id="Data_IO"></a>

# Data IO

<a id="senpi.data_io.basic_utils"></a>

## Basic Utils (senpi.data\_io.basic\_utils)

<a id="senpi.data_io.basic_utils.load_to_df"></a>

### load\_to\_df

```python
def load_to_df(f,
               delim: str = ',',
               order: List[str] = None,
               ret_order: List[str] = None,
               dtypes: Dict[str, np.dtype] = None,
               transform_polarity: bool = True,
               format_version: str = None,
               sub_first_timestamp: bool = True) -> pd.DataFrame
```

Load event data from a CSV-like, AEDAT (DAVIS output), HDF5 (Prophesee output), or RAW (Prophesee output; either EVT 2.0, 2.1, or 3.0) file into a Pandas DataFrame.

The shape of the returned dataframe will be `N` by 4, where `N` is the number of events in the event stream. The data columns and their order is determined by the parameter `ret_order`, which must be of type `List[str]` and some permutation of `['t', 'x', 'y', 'p']`. The data types of each of the columns can also be modified by passing an optional `dtypes` parameter, which should be of type `Dict[str, np.dtype]` (dictionary mapping `str` to `np.dtype`).

In the case where a CSV-like file is being read in, it is necessary to pass in an `order` parameter of type `List[str]` (some permutation of `['t', 'x', 'y', 'p']`) so that the order of the input data's columns is known. A delimeter parameter `delim` must also be passed when reading in a CSV-like file.

The `format_version` parameter is relevant when a RAW file is being read. This optional parameter denotes the [EVT format](https://docs.prophesee.ai/stable/data/encoding_formats/index.html) that the inputted RAW file is in and should be set to one of `'2.0'`, `'2.1'`, or `'3.0'`. Otherwise, it is inferred from the header of the RAW file.

The `transform_polarity` parameter, if set to `True` (which is its default value), will transform the polarity values from 0 for OFF events and 1 for ON events to -1 for OFF events and 1 for ON events. Note: if the polarities are already represented by -1s and 1s, they will remain the same no matter what `transform_polarity` is set to.

The `sub_first_timestamp` parameter, if set to `True` (which is its default value), will subtract the first event timestamp value from all of the recorded event timestamps when writing to the data frame.

**Arguments**:

- `f` (`str`): The path to the data file.
- `delim` (`str`): The delimiter used in the data file (default is a comma).
- `order` (`List[str]`): The order of columns in the file (default is ['t', 'x', 'y', 'p']).
- `ret_order` (`List[str], optional`): The desired order of columns in the resulting DataFrame (default is None, which keeps the original order).
- `dtypes` (`Dict[str, np.dtype], optional`): A dictionary specifying data types for columns.
- `transform_polarity` (`bool`): Whether to transform the polarity values from 0 for OFF events and 1 for ON events to -1 for OFF events and 1 for ON events (default is True). Note: if the polarities are already represented by -1s and 1s, they will remain the same no matter what `transform_polarity` is set to.
- `format_version` (`str`): The EVT format version of the inputted RAW file (one of "2.0", "2.1" or "3.0"). If not set, it will be inferred from the header of the RAW file.
- `sub_first_timestamp` (`bool`): Whether to subtract the first event timestamp from all of the recorded event timestamps when writing to the data frame.

**Returns**:

`pd.DataFrame`: A Pandas DataFrame containing the loaded data.

<a id="senpi.data_io.basic_utils.load_to_np_arr"></a>

### load\_to\_np\_arr

```python
def load_to_np_arr(f,
                   delim: str = ',',
                   order: List[str] = None,
                   ret_order: List[str] = None,
                   dtypes: Dict[str, np.dtype] = None,
                   transform_polarity: bool = True,
                   format_version: str = None,
                   sub_first_timestamp: bool = True) -> np.ndarray
```

Load event data from a CSV-like, AEDAT (DAVIS output), HDF5 (Prophesee output), or RAW (Prophesee output; either EVT 2.0, 2.1, or 3.0) file into a NumPy structured array.

The shape of the returned structured array will be `N` by 4, where `N` is the number of events in the event stream. The data columns and their order is determined by the parameter `ret_order`. All of the parameters for this function have the same behavior as specified in the description of [`load_to_df`](#load_to_df). Also, `load_to_np_arr` itself invokes `load_to_df` in its implementation.

**Arguments**:

- `f` (`str`): The path to the data file.
- `delim` (`str`): The delimiter used in the data file (default is a comma).
- `order` (`List[str]`): The order of columns in the file (default is ['t', 'x', 'y', 'p']).
- `ret_order` (`List[str], optional`): The desired order of columns in the resulting NumPy array (default is None, which keeps the original order).
- `dtypes` (`Dict[str, np.dtype], optional`): A dictionary specifying data types for columns.
- `transform_polarity` (`bool`): Whether to transform the polarity values from 0 for OFF events and 1 for ON events to -1 for OFF events and 1 for ON events (default is True). Note: if the polarities are already represented by -1s and 1s, they will remain the same no matter what `transform_polarity` is set to.
- `format_version` (`str`): The EVT format version of the inputted RAW file (one of "2.0", "2.1", or "3.0"). If not set, it will be inferred from the header of the RAW file.
- `sub_first_timestamp` (`bool`): Whether to subtract the first event timestamp from all of the recorded event timestamps when writing to the NumPy array.

**Returns**:

`np.ndarray`: A NumPy structured array containing the loaded data.

<a id="senpi.data_io.basic_utils.load_to_tensor"></a>

### load\_to\_tensor

```python
def load_to_tensor(
    f,
    delim: str = ',',
    order: List[str] = None,
    ret_order: List[str] = None,
    dtypes: Dict[str, torch.dtype] = None,
    stack_tensors: bool = True,
    device: torch.device = "cpu",
    transform_polarity: bool = True,
    format_version: str = None,
    sub_first_timestamp: bool = True
) -> Union[torch.Tensor, Tuple[torch.Tensor]]
```

Load event data from a CSV-like, AEDAT (DAVIS output), HDF5 (Prophesee output), or RAW (Prophesee output; either EVT 2.0, 2.1, or 3.0) file into a PyTorch tensor.

If the `stack_tensors` parameter is set to `True` (which is its default value), then a single 2D tensor with shape `N` by 4 will be returned, where `N` is the number of events in the event stream. If `stack_tensors` is set to `False`, then a tuple of 4 tensors (each of shape `N`) will be returned. In both cases, the data columns/tensors and their order is determined by the parameter `ret_order`.

The data types of each of the data tensors can be modified by passing an optional `dtypes` parameter, which should be of type `Dict[str, torch.dtype]` (dictionary mapping `str` to `torch.dtype`). If `stack_tensors` is set to `True`, then a single 2D tensor of the most dominant type out of the values of the `dtypes` dictionary will be returned.

All of the other parameters for this function have the same behavior as specified in the description of [`load_to_df`](#load_to_df). Also, `load_to_tensor` itself invokes `load_to_df` in its implementation.

**Arguments**:

- `f` (`str`): The path to the data file.
- `delim` (`str`): The delimiter used in the data file (default is a comma).
- `order` (`List[str]`): The order of columns in the file, if applicable. If not inputted, default value will be set to ['t', 'x', 'y', 'p'].
- `ret_order` (`List[str], optional`): The desired order of columns in the resulting Tensor (default is None, which keeps the original order).
- `dtypes` (`Dict[str, torch.dtype], optional`): A dictionary specifying data types for columns.
- `stack_tensors` (`bool`): Whether to stack the tensors into a single Tensor (default is True).
- `device` (`torch.device`): The device to which the Tensor should be moved (default is "cpu").
- `transform_polarity` (`bool`): Whether to transform the polarity values from 0 for OFF events and 1 for ON events to -1 for OFF events and 1 for ON events (default is True). Note: if the polarities are already represented by -1s and 1s, they will remain the same no matter what `transform_polarity` is set to.
- `format_version` (`str`): The EVT format version of the inputted RAW file (one of "2.0", "2.1", or "3.0"). If not set, it will be inferred from the header of the RAW file.
- `sub_first_timestamp` (`bool`): Whether to subtract the first event timestamp from all of the recorded event timestamps when writing to the torch Tensor.

**Returns**:

`torch.Tensor`: A PyTorch Tensor containing the loaded data.

<a id="senpi.data_io.basic_utils.save_df_to_csv"></a>

### save\_df\_to\_csv

```python
def save_df_to_csv(df: pd.DataFrame,
                   f: Union[Path, str] = None,
                   delim: str = ',',
                   save_order: List[str] = None,
                   revert_polarity: bool = True,
                   with_index: bool = False,
                   with_header: bool = False)
```

Save a Pandas DataFrame to a CSV file.

**Arguments**:

- `df` (`pd.DataFrame`): The Pandas DataFrame to save.
- `f` (`Union[Path, str], optional`): The path where the CSV file will be saved (default is "saved_event_data.csv").
- `delim` (`str`): The delimiter to use in the CSV file (default is a comma).
- `save_order` (`List[str], optional`): The order of columns in the resulting CSV file (default is None, which keeps the original order).
- `revert_polarity` (`bool`): Whether to revert the polarity values from -1 for OFF events and 1 for ON events to 0 for OFF events and 1 for ON events (default is True). Note: if the polarities are already represented by 0s and 1s, they will remain the same no matter what `revert_polarity` is set to.
- `with_index` (`bool`): Whether to include the index in the CSV file (default is False).
- `with_header` (`bool`): Whether to include the header in the CSV file (default is False).

<a id="senpi.data_io.basic_utils.save_tensor_to_csv"></a>

### save\_tensor\_to\_csv

```python
def save_tensor_to_csv(events: torch.Tensor,
                       f: Union[Path, str] = None,
                       delim: str = ',',
                       order: List[str] = None,
                       save_order: List[str] = None,
                       revert_polarity: bool = True,
                       with_index: bool = False,
                       with_header: bool = False)
```

Save a PyTorch Tensor to a CSV file.

**Arguments**:

- `events` (`torch.Tensor`): The PyTorch Tensor to save.
- `f` (`Union[Path, str], optional`): The path where the CSV file will be saved (default is "saved_event_data.csv").
- `delim` (`str`): The delimiter to use in the CSV file (default is a comma).
- `order` (`List[str], optional`): The order of columns in the resulting CSV file (default is None, which uses the order from the Tensor).
- `save_order` (`List[str], optional`): The desired order of columns in the CSV file (default is None, which keeps the order from `order`).
- `revert_polarity` (`bool`): Whether to revert the polarity values from -1 for OFF events and 1 for ON events to 0 for OFF events and 1 for ON events (default is True). Note: if the polarities are already represented by 0s and 1s, they will remain the same no matter what `revert_polarity` is set to.
- `with_index` (`bool`): Whether to include the index in the CSV file (default is False).
- `with_header` (`bool`): Whether to include the header in the CSV file (default is False).

<a id="senpi.data_io.basic_utils.save_np_arr_to_csv"></a>

### save\_np\_arr\_to\_csv

```python
def save_np_arr_to_csv(np_arr: pd.DataFrame,
                       f: Union[Path, str] = None,
                       delim: str = ',',
                       save_order: List[str] = None,
                       revert_polarity: bool = True,
                       with_index: bool = False,
                       with_header: bool = False)
```

Save a NumPy structured array to a CSV file.

**Arguments**:

- `np_arr` (`pd.DataFrame`): The NumPy structured array to save.
- `f` (`Union[Path, str], optional`): The path where the CSV file will be saved (default is "saved_event_data.csv").
- `delim` (`str`): The delimiter to use in the CSV file (default is a comma).
- `save_order` (`List[str], optional`): The desired order of columns in the CSV file (default is None, which keeps the original order).
- `revert_polarity` (`bool`): Whether to revert the polarity values from -1 for OFF events and 1 for ON events to 0 for OFF events and 1 for ON events (default is True). Note: if the polarities are already represented by 0s and 1s, they will remain the same no matter what `revert_polarity` is set to.
- `with_index` (`bool`): Whether to include the index in the CSV file (default is False).
- `with_header` (`bool`): Whether to include the header in the CSV file (default is False).

<a id="senpi.data_io.basic_utils.get_dominant_torch_dtype"></a>

### get\_dominant\_torch\_dtype

```python
def get_dominant_torch_dtype(dtypes_iter: List[torch.dtype]) -> torch.dtype
```

Determine the dominant PyTorch data type from a list of data types.

**Arguments**:

- `dtypes_iter` (`List[torch.dtype]`): A list of PyTorch data types.

**Returns**:

`torch.dtype`: The dominant PyTorch data type.

<a id="senpi.data_io.basic_utils.get_np_dtype_from_torch_dtype"></a>

### get\_np\_dtype\_from\_torch\_dtype

```python
def get_np_dtype_from_torch_dtype(torch_dtype: torch.dtype)
```

Convert a PyTorch data type to its corresponding NumPy data type.

**Arguments**:

- `torch_dtype` (`torch.dtype`): The PyTorch data type to convert.

**Returns**:

`np.dtype`: The corresponding NumPy data type.

<a id="senpi.data_io.basic_utils.get_torch_dtype_from_np_dtype"></a>

### get\_torch\_dtype\_from\_np\_dtype

```python
def get_torch_dtype_from_np_dtype(np_dtype: np.dtype)
```

Convert a NumPy data type to its corresponding PyTorch data type.

**Arguments**:

- `np_dtype` (`np.dtype`): The NumPy data type to convert.

**Returns**:

`torch.dtype`: The corresponding PyTorch data type.

<a id="Data_Manipulation"></a>

# Data Manipulation

<a id="senpi.data_manip.algs"></a>

## Algorithms (senpi.data\_manip.algs)

<a id="senpi.data_manip.algs.Alg"></a>

### Alg Objects

```python
class Alg()
```

A base class for algorithms that transform event data.

This class should be subclassed to implement specific transformation methods for event data.

<a id="senpi.data_manip.algs.Alg.__call__"></a>

#### \_\_call\_\_

```python
def __call__(events: Union[pd.DataFrame, np.ndarray, torch.Tensor],
             **kwargs) -> Union[pd.DataFrame, np.ndarray, torch.Tensor]
```

Apply the transformation algorithm to the event data.

**Arguments**:

- `events` (`Union[pd.DataFrame, np.ndarray, torch.Tensor]`): The event data to transform.
- `kwargs`: Additional parameters for the transform method.

**Returns**:

`Union[pd.DataFrame, np.ndarray, torch.Tensor]`: The transformed event data.

<a id="senpi.data_manip.algs.Alg.__str__"></a>

#### \_\_str\_\_

```python
def __str__()
```

Return a string representation of the Alg instance.

**Returns**:

`str`: A string representation of the instance's attributes.

<a id="senpi.data_manip.algs.Alg.transform"></a>

#### transform

```python
def transform(events: Union[pd.DataFrame, np.ndarray, torch.Tensor],
              **kwargs) -> Union[pd.DataFrame, np.ndarray, torch.Tensor]
```

Transform the event data.

This method should be implemented by subclasses to define the specific transformation behavior.

**Arguments**:

- `events` (`Union[pd.DataFrame, np.ndarray, torch.Tensor]`): The event data to transform.
- `kwargs`: Additional parameters for the transform method.

**Raises**:

- `NotImplementedError`: If the method is not implemented by a subclass.

**Returns**:

`Union[pd.DataFrame, np.ndarray, torch.Tensor]`: The transformed event data.

<a id="senpi.data_manip.algs.FlipXAlg"></a>

### FlipXAlg Objects

```python
class FlipXAlg(Alg)
```

A class for flipping events along the X-axis (across the Y-axis).

This class has one attribute `max_width_index`, which should be set to the maximum width index of the camera frame (WIDTH - 1).

After instantiation of a `FlipXAlg` object, it can be called on (or, more explicitly, its `transform` method can be called on) any of the core data structures containing event stream data, namely Pandas DataFrames, NumPy structured arrays, and PyTorch tensors, to transform their data accordingly. If the `modify_original` parameter is set to `True`, the data structure passed in will be modified in-place (if this parameter is set to `False` a modified copy of the inputted data structure will be returned instead). If the inputted data structure is a PyTorch tensor, then an `order` parameter must be passed in to this function as specified in the **Order** section in [Terminology](#terminology).

**Arguments**:

- `max_width_index` (`int`): The maximum width index (width minus one) used for the transformation.

<a id="senpi.data_manip.algs.FlipXAlg.__init__"></a>

#### \_\_init\_\_

```python
def __init__(max_width_index: int)
```

Initialize the FlipXAlg with the maximum width index.

**Arguments**:

- `max_width_index` (`int`): The maximum width index (width minus one) used for the transformation.

<a id="senpi.data_manip.algs.FlipXAlg.transform"></a>

#### transform

```python
def transform(
    events: Union[pd.DataFrame, np.ndarray, torch.Tensor],
    order: List[str] = None,
    modify_original: bool = True
) -> Union[pd.DataFrame, np.ndarray, torch.Tensor]
```

Transform the event data by flipping it along the X-axis.

This method determines the type of the input data and calls the appropriate transformation method for Pandas DataFrame, NumPy array, or PyTorch Tensor.

**Arguments**:

- `events` (`Union[pd.DataFrame, np.ndarray, torch.Tensor]`): The event data to transform.
- `order` (`List[str], optional`): The order of columns in the event data (default is None, which uses the default order).
- `modify_original` (`bool`): Whether to modify the original data or return a modified copy (default is True).

**Returns**:

`Union[pd.DataFrame, np.ndarray, torch.Tensor]`: The transformed event data.

<a id="senpi.data_manip.algs.FlipXAlg.transform_pd_np"></a>

#### transform\_pd\_np

```python
def transform_pd_np(
        events: Union[pd.DataFrame, np.ndarray],
        modify_original: bool = True) -> Union[pd.DataFrame, np.ndarray]
```

Transform the event data by flipping it along the X-axis for Pandas DataFrame or NumPy array.

**Arguments**:

- `events` (`Union[pd.DataFrame, np.ndarray]`): The event data to transform.
- `modify_original` (`bool`): Whether to modify the original data or return a modified copy (default is True).

**Returns**:

`Union[pd.DataFrame, np.ndarray]`: The transformed event data.

<a id="senpi.data_manip.algs.FlipXAlg.transform_tensor"></a>

#### transform\_tensor

```python
def transform_tensor(events: torch.Tensor,
                     order: List[str] = None,
                     modify_original: bool = True) -> torch.Tensor
```

Transform the event data by flipping it along the X-axis for PyTorch Tensor.

**Arguments**:

- `events` (`torch.Tensor`): The event data to transform.
- `order` (`List[str], optional`): The order of columns in the event data (default is None, which uses the default order).
- `modify_original` (`bool`): Whether to modify the original data or return a modified copy (default is True).

**Returns**:

`torch.Tensor`: The transformed event data.

<a id="senpi.data_manip.algs.FlipXAlg.set_max_width_index"></a>

#### set\_max\_width\_index

```python
def set_max_width_index(max_width_index: int)
```

Set the maximum width index for the transformation.

**Arguments**:

- `max_width_index` (`int`): The maximum width index (width minus one) to set.

<a id="senpi.data_manip.algs.FlipYAlg"></a>

### FlipYAlg Objects

```python
class FlipYAlg(Alg)
```

A class for flipping events along the Y-axis (across the X-axis).

This class provides methods to transform event data by flipping it along the Y-axis. It supports operations on data in the form of Pandas DataFrames, NumPy arrays, and PyTorch tensors.

**Arguments**:

- `max_height_index` (`int`): The maximum height index (height minus one) used for the transformation.

<a id="senpi.data_manip.algs.FlipYAlg.__init__"></a>

#### \_\_init\_\_

```python
def __init__(max_height_index: int)
```

Initialize the FlipYAlg with the maximum height index.

**Arguments**:

- `max_height_index` (`int`): The maximum height index (height minus one) used for the transformation.

<a id="senpi.data_manip.algs.FlipYAlg.transform"></a>

#### transform

```python
def transform(
    events: Union[pd.DataFrame, np.ndarray, torch.Tensor],
    order: List[str] = None,
    modify_original: bool = True
) -> Union[pd.DataFrame, np.ndarray, torch.Tensor]
```

Transform the event data by flipping it along the Y-axis.

This method determines the type of the input data and calls the appropriate transformation method for Pandas DataFrame, NumPy array, or PyTorch Tensor.

**Arguments**:

- `events` (`Union[pd.DataFrame, np.ndarray, torch.Tensor]`): The event data to transform.
- `order` (`List[str], optional`): The order of columns in the event data (default is None, which uses the default order).
- `modify_original` (`bool`): Whether to modify the original data or return a modified copy (default is True).

**Returns**:

`Union[pd.DataFrame, np.ndarray, torch.Tensor]`: The transformed event data.

<a id="senpi.data_manip.algs.FlipYAlg.transform_pd_np"></a>

#### transform\_pd\_np

```python
def transform_pd_np(
        events: Union[pd.DataFrame, np.ndarray],
        modify_original: bool = True) -> Union[pd.DataFrame, np.ndarray]
```

Transform the event data by flipping it along the Y-axis for Pandas DataFrame or NumPy array.

**Arguments**:

- `events` (`Union[pd.DataFrame, np.ndarray]`): The event data to transform.
- `modify_original` (`bool`): Whether to modify the original data or return a modified copy (default is True).

**Returns**:

`Union[pd.DataFrame, np.ndarray]`: The transformed event data.

<a id="senpi.data_manip.algs.FlipYAlg.transform_tensor"></a>

#### transform\_tensor

```python
def transform_tensor(events: torch.Tensor,
                     order: List[str] = None,
                     modify_original: bool = True) -> torch.Tensor
```

Transform the event data by flipping it along the Y-axis for PyTorch Tensor.

**Arguments**:

- `events` (`torch.Tensor`): The event data to transform.
- `order` (`List[str], optional`): The order of columns in the event data (default is None, which uses the default order).
- `modify_original` (`bool`): Whether to modify the original data or return a modified copy (default is True).

**Returns**:

`torch.Tensor`: The transformed event data.

<a id="senpi.data_manip.algs.FlipYAlg.set_max_height_index"></a>

#### set\_max\_height\_index

```python
def set_max_height_index(max_height_index: int)
```

Set the maximum height index for the transformation.

**Arguments**:

- `max_height_index` (`int`): The maximum height index (height minus one) to set.

<a id="senpi.data_manip.algs.InvertPolarityAlg"></a>

### InvertPolarityAlg Objects

```python
class InvertPolarityAlg(Alg)
```

A class for inverting the polarity of events.

This class provides methods to invert the polarity of event data. It supports operations on data in the form of Pandas DataFrames, NumPy arrays, and PyTorch tensors. It also supports binary polarity mode.

<a id="senpi.data_manip.algs.InvertPolarityAlg.transform"></a>

#### transform

```python
def transform(
    events: Union[pd.DataFrame, np.ndarray, torch.Tensor],
    order: List[str] = None,
    binary_polarity_mode: bool = False,
    modify_original: bool = True
) -> Union[pd.DataFrame, np.ndarray, torch.Tensor]
```

Transform the event data by inverting the polarity.

This method determines the type of the input data and calls the appropriate transformation method for Pandas DataFrame, NumPy array, or PyTorch Tensor.

**Arguments**:

- `events` (`Union[pd.DataFrame, np.ndarray, torch.Tensor]`): The event data to transform.
- `order` (`List[str], optional`): The order of columns in the event data (default is None, which uses the default order).
- `binary_polarity_mode` (`bool`): Whether the events are represented such that 0 denotes OFF events and 1 denotes ON events (default is False; assuming that OFF events are denoted by -1 and ON events are denoted by 1).
- `modify_original` (`bool`): Whether to modify the original data or return a modified copy (default is True).

**Returns**:

`Union[pd.DataFrame, np.ndarray, torch.Tensor]`: The transformed event data.

<a id="senpi.data_manip.algs.InvertPolarityAlg.transform_pd_np"></a>

#### transform\_pd\_np

```python
def transform_pd_np(
        events: Union[pd.DataFrame, np.ndarray],
        binary_polarity_mode: bool = False,
        modify_original: bool = True) -> Union[pd.DataFrame, np.ndarray]
```

Transform the event data by inverting the polarity for Pandas DataFrame or NumPy array.

**Arguments**:

- `events` (`Union[pd.DataFrame, np.ndarray]`): The event data to transform.
- `binary_polarity_mode` (`bool`): Whether the events are represented such that 0 denotes OFF events and 1 denotes ON events (default is False; assuming that OFF events are denoted by -1 and ON events are denoted by 1).
- `modify_original` (`bool`): Whether to modify the original data or return a modified copy (default is True).

**Returns**:

`Union[pd.DataFrame, np.ndarray]`: The transformed event data.

<a id="senpi.data_manip.algs.InvertPolarityAlg.transform_tensor"></a>

#### transform\_tensor

```python
def transform_tensor(events: torch.Tensor,
                     order: List[str] = None,
                     binary_polarity_mode: bool = False,
                     modify_original: bool = True) -> torch.Tensor
```

Transform the event data by inverting the polarity for PyTorch Tensor.

**Arguments**:

- `events` (`torch.Tensor`): The event data to transform.
- `order` (`List[str], optional`): The order of columns in the event data (default is None, which uses the default order).
- `binary_polarity_mode` (`bool`): Whether the events are represented such that 0 denotes OFF events and 1 denotes ON events (default is False; assuming that OFF events are denoted by -1 and ON events are denoted by 1).
- `modify_original` (`bool`): Whether to modify the original data or return a modified copy (default is True).

**Returns**:

`torch.Tensor`: The transformed event data.

<a id="senpi.data_manip.filters"></a>

## Filters (senpi.data\_manip.filters)

<a id="senpi.data_manip.filters.Filter"></a>

### Filter Objects

```python
class Filter()
```

A base class for filtering event data.

This class should be subclassed to implement specific filtering methods for event data.

<a id="senpi.data_manip.filters.Filter.__call__"></a>

#### \_\_call\_\_

```python
def __call__(events: Union[pd.DataFrame, np.ndarray, torch.Tensor],
             **kwargs) -> Union[pd.DataFrame, np.ndarray, torch.Tensor]
```

Apply the filter to the event data.

**Arguments**:

- `events` (`Union[pd.DataFrame, np.ndarray, torch.Tensor]`): The event data to filter.
- `kwargs`: Additional parameters for the filter method.

**Returns**:

`Union[pd.DataFrame, np.ndarray, torch.Tensor]`: The filtered event data.

<a id="senpi.data_manip.filters.Filter.__str__"></a>

#### \_\_str\_\_

```python
def __str__()
```

Return a string representation of the Filter instance.

**Returns**:

`str`: A string representation of the instance's attributes.

<a id="senpi.data_manip.filters.Filter.filter_events"></a>

#### filter

```python
def filter_events(events: Union[pd.DataFrame, np.ndarray, torch.Tensor],
           **kwargs) -> Union[pd.DataFrame, np.ndarray, torch.Tensor]
```

Filter the event data.

This method should be implemented by subclasses to define the specific filtering behavior.

**Arguments**:

- `events` (`Union[pd.DataFrame, np.ndarray, torch.Tensor]`): The event data to filter.
- `kwargs`: Additional parameters for the filter method.

**Raises**:

- `NotImplementedError`: If the method is not implemented by a subclass.

**Returns**:

`Union[pd.DataFrame, np.ndarray, torch.Tensor]`: The filtered event data.

<a id="senpi.data_manip.filters.Filter.filter_frames_tensor"></a>

#### filter\_frames\_tensor

```python
def filter_frames_tensor(frames: torch.Tensor, **kwargs) -> torch.Tensor
```

Filter the frames data.

This method should be implemented by subclasses to define the specific filtering behavior for frame volumes.

**Arguments**:

- `frames` (`torch.Tensor`): The frame data to filter.
- `kwargs`: Additional parameters for the filter method.

**Raises**:

- `NotImplementedError`: If the method is not implemented by a subclass.

**Returns**:

`torch.Tensor`: The filtered frame volume data.

<a id="senpi.data_manip.filters.PolarityFilter"></a>

### PolarityFilter Objects

```python
class PolarityFilter(Filter)
```

A class to filter event-based vision data by event polarity.

This class has one attribute `filter_type`, which determines the type of events to filter for (`True` for positive events, `False` for negative events).

After instantiation of a `PolarityFilter` object, it can be called on (or, more explicitly, its `filter` method can be called on) any of the core data structures containing event stream data, namely Pandas DataFrames, NumPy structured arrays, and PyTorch tensors, to filter their data accordingly. If the inputted data structure is a PyTorch tensor, then an `order` parameter must be passed in to this function as specified in the **Order** section in [Terminology](#terminology).

Specifically for the `PolarityFilter` filter class's `filter` method, a `binary_polarity_mode` parameter must be passed in as specified in the **Binary Polarity Mode** section in [Terminology](#terminology).

Also, unless `reset_indices` is set to `True` in the case where a Pandas DataFrame is passed in, the `filter` method for `PolarityFilter` will return the rows of the original data structure (Pandas DataFrame, NumPy structured array, or PyTorch tensor) matching the class's polarity type **in-place**.

**Arguments**:

- `filter_type` (`bool`): The type of events to filter (True for positive events, False for negative events).

<a id="senpi.data_manip.filters.PolarityFilter.__init__"></a>

#### \_\_init\_\_

```python
def __init__(filter_type: bool)
```

Initialize the polarity filter.

**Arguments**:

- `filter_type` (`bool`): The type of events to filter (True for positive events, False for negative events).

<a id="senpi.data_manip.filters.PolarityFilter.filter_events"></a>

#### filter\_events

```python
def filter_events(
    events: Union[pd.DataFrame, np.ndarray, torch.Tensor],
    order: List[str] = None,
    binary_polarity_mode: bool = False,
    reset_indices: bool = True
) -> Union[pd.DataFrame, np.ndarray, torch.Tensor]
```

Filter the inputted events by the polarity given by `self.filter_type`.

**Arguments**:

- `events` (`Union[pd.DataFrame, np.ndarray, torch.Tensor]`): The events to filter.
- `order` (`List[str], optional`): The order of the columns of an inputted events torch.Tensor. By default, this is set to None and further handled by helper methods.
- `binary_polarity_mode` (`bool, optional`): Whether the polarities are represented by 0s and 1s (True) or -1s and 1s (False).
- `reset_indices` (`bool, optional`): Whether to reset the indices of the returned Pandas dataframe (if a Pandas dataframe is inputted) after filtering it.

**Returns**:

`Union[pd.DataFrame, np.ndarray, torch.Tensor]`: The filtered events.

<a id="senpi.data_manip.filters.PolarityFilter.filter_events_pd"></a>

#### filter\_events\_pd

```python
def filter_events_pd(events: pd.DataFrame,
              binary_polarity_mode: bool = False,
              reset_indices: bool = True) -> pd.DataFrame
```

Filter a Pandas DataFrame of events by polarity.

**Arguments**:

- `events` (`pd.DataFrame`): The events to filter.
- `binary_polarity_mode` (`bool, optional`): Whether the polarities are represented by 0s and 1s (True) or -1s and 1s (False).
- `reset_indices` (`bool, optional`): Whether to reset the indices of the returned DataFrame after filtering.

**Returns**:

`pd.DataFrame`: The filtered events.

<a id="senpi.data_manip.filters.PolarityFilter.filter_np"></a>

#### filter\_events\_np

```python
def filter_events_np(events: np.ndarray,
              binary_polarity_mode: bool = False) -> np.ndarray
```

Filter a numpy array of events by polarity.

**Arguments**:

- `events` (`np.ndarray`): The events to filter.
- `binary_polarity_mode` (`bool, optional`): Whether the polarities are represented by 0s and 1s (True) or -1s and 1s (False).

**Returns**:

`np.ndarray`: The filtered events.

<a id="senpi.data_manip.filters.PolarityFilter.filter_tensor"></a>

#### filter\_events\_tensor

```python
def filter_events_tensor(events: torch.Tensor,
                  order: List[str] = None,
                  binary_polarity_mode: bool = False) -> torch.Tensor
```

Filter a torch.Tensor of events by polarity.

**Arguments**:

- `events` (`torch.Tensor`): The events to filter.
- `order` (`List[str], optional`): The order of the columns of the inputted events tensor. Defaults to ['b', 't', 'x', 'y', 'p'].
- `binary_polarity_mode` (`bool, optional`): Whether the polarities are represented by 0s and 1s (True) or -1s and 1s (False).

**Returns**:

`torch.Tensor`: The filtered events.

<a id="senpi.data_manip.filters.PolarityFilter.filter_frames_tensor"></a>

#### filter\_frames\_tensor

```python
def filter_frames_tensor(frames: torch.Tensor, inplace: bool) -> torch.Tensor
```

Filter a torch.Tensor frame volume of events by polarity.

**Arguments**:

- `frames` (`torch.Tensor`): The frames to filter. Expected format: (N, C, X, Y), where N is the number of frames (each frame displays an accumulation of events for every delta t window) and C is the number of channels (expected to be 1).
- `inplace` (`bool`): Whether to apply the filter in place or out of place.

**Returns**:

`torch.Tensor`: The filtered events.

<a id="senpi.data_manip.filters.PolarityFilter.set_filter_type"></a>

#### set\_filter\_type

```python
def set_filter_type(filter_type: bool)
```

Set the type of polarity to filter.

**Arguments**:

- `filter_type` (`bool`): The type of events to filter (True for positive events, False for negative events).

<a id="senpi.data_manip.filters.BAFilter"></a>

### BAFilter Objects

```python
class BAFilter(Filter)
```

A class to filter event-based vision data using a background activity (BA) filter based on temporal and spatial constraints.

**E-MLB:** https://ieeexplore.ieee.org/document/10078400 <br/>

**Based on the following paper(s) (as referenced in E-MLB):** <br/>

**Paper 1:** <br/>
T. Delbruck, “Frame-free dynamic digital vision,” in Proc. Intl. Symp. Secure-Life Electron., Adv. Electron. Qual. Life Soc., 2008, vol. 1, pp. 21-26. <br/>

**Paper 1 Page:** <br/>
https://www.scirp.org/reference/referencespapers?referenceid=1176812 <br/>

**Paper 1 PDF:** https://web.archive.org/web/20170809001804/http://www.ini.uzh.ch/~tobi/wiki/lib/exe/fetch.php?media=delbruckeventbasedvision_gcoe_symp_2008.pdf <br/>

**Paper 2:** <br/>
D. Czech and G. Orchard, "Evaluating noise filtering for event-based asynchronous change detection image sensors,"
2016 6th IEEE International Conference on Biomedical Robotics and Biomechatronics (BioRob), Singapore, 2016, pp. 19-24, doi: 10.1109/BIOROB.2016.7523452.

**Filter Functionality**<br/>
The filter has a single time threshold parameter, namely $T$. Then, the entire event stream is iterated over sequentially. The steps for each event are the following:
- Store the event’s timestamp in only its 8 neighboring pixels in the timestamp map, overwriting the previous values.
- Check if the timestamp of the current event is within $T$ of the previous value written to the timestamp map at this event’s pixel. If so, keep the event in the event stream (classify it as signal), otherwise discard it.

**Sidenote**<br/>
Unless a PyTorch tensor is passed in or `reset_indices` is set to `True` in the case where a Pandas DataFrame is passed in, the `filter` method for `BAFilter` will return the rows of the original data structure (Pandas DataFrame or NumPy structured array) corresponding to signal events **in-place**.

**Arguments**:

- `time_threshold` (`Union[int, float]`): The time threshold for filtering events.
- `height` (`int`): The height of the event frame.
- `width` (`int`): The width of the event frame.

<a id="senpi.data_manip.filters.BAFilter.__init__"></a>

#### \_\_init\_\_

```python
def __init__(time_threshold: Union[int, float], height: int, width: int)
```

Initialize the background activity filter (BAF).

**Arguments**:

- `time_threshold` (`Union[int, float]`): The time threshold for filtering events.
- `height` (`int`): The height of the event frame.
- `width` (`int`): The width of the event frame.

<a id="senpi.data_manip.filters.BAFilter.filter_events"></a>

#### filter\_events

```python
def filter_events(
    events: Union[pd.DataFrame, np.ndarray, torch.Tensor],
    order: List[str] = None,
    reset_indices: bool = True
) -> Union[pd.DataFrame, np.ndarray, torch.Tensor]
```

Filter the inputted events using the BA filter.

**Arguments**:

- `events` (`Union[pd.DataFrame, np.ndarray, torch.Tensor]`): The events to filter.
- `order` (`List[str], optional`): The order of the columns of an inputted events torch.Tensor. By default, this is set to None and further handled by helper methods.
- `reset_indices` (`bool, optional`): Whether to reset the indices of the returned Pandas dataframe (if a Pandas dataframe is inputted) after filtering it.

**Returns**:

`Union[pd.DataFrame, np.ndarray, torch.Tensor]`: The filtered events.

<a id="senpi.data_manip.filters.BAFilter.filter_events_pd"></a>

#### filter\_events\_pd

```python
def filter_events_pd(events: pd.DataFrame,
                     reset_indices: bool) -> pd.DataFrame
```

Filter a Pandas DataFrame of events using the BA filter.

**Arguments**:

- `events` (`pd.DataFrame`): The events to filter.
- `reset_indices` (`bool, optional`): Whether to reset the indices of the returned DataFrame after filtering.

**Returns**:

`pd.DataFrame`: The filtered events.

<a id="senpi.data_manip.filters.BAFilter.filter_events_np"></a>

#### filter\_events\_np

```python
def filter_events_np(events: np.ndarray) -> np.ndarray
```

Filter a numpy array of events using the BA filter.

**Arguments**:

- `events` (`np.ndarray`): The events to filter.

**Returns**:

`np.ndarray`: The filtered events.

<a id="senpi.data_manip.filters.BAFilter.filter_events_tensor"></a>

#### filter\_events\_tensor

```python
def filter_events_tensor(events: torch.Tensor,
                         order: List[str] = None) -> torch.Tensor
```

Filter a torch.Tensor of events using the BA filter. This method converts the inputted PyTorch tensor into an intermediate Pandas DataFrame and invokes `filter_events_pd` on it. The filtered dataframe is then converted back to a torch.Tensor and returned.

**Arguments**:

- `events` (`torch.Tensor`): The events to filter.
- `order` (`List[str], optional`): The order of the columns of the inputted events tensor. Defaults to ['b', 't', 'x', 'y', 'p'].

**Returns**:

`torch.Tensor`: The filtered events.

<a id="senpi.data_manip.filters.BAFilter.filter_frames_tensor"></a>

#### filter\_frames\_tensor

```python
def filter_frames_tensor(frames: torch.Tensor,
                         inplace: bool,
                         device: torch.device = None,
                         polarity_agnostic: bool = False) -> torch.Tensor
```

Apply the BA filter to a 4D tensor of frames (frame volume of events).

**Arguments**:

- `frames` (`torch.Tensor`): A 4D torch.Tensor of frames to filter. Expected format: (N, C, X, Y), where N is the number of frames, C is the number of channels (should be 1), and (X, Y) are the frame dimensions.
- `inplace` (`bool`): Whether to apply the filter in place or out of place.
- `device` (`torch.device, optional`): The device to run the filter on (e.g., CPU or GPU). If not provided, defaults to "cuda" if available.
- `polarity_agnostic` (`bool, optional`): If True, the filter will treat both positive and negative polarities equally by applying the filtering operation on the absolute values of the input frames. If False, the filter will handle positive and negative polarities independently, applying polarity-specific masks during filtering.

**Returns**:

`torch.Tensor`: The filtered frames as a torch.Tensor.

<a id="senpi.data_manip.filters.BAFilter.set_time_threshold"></a>

#### set\_time\_threshold

```python
def set_time_threshold(time_threshold: Union[int, float])
```

Set the time threshold for the BA filter.

**Arguments**:

- `time_threshold` (`Union[int, float]`): The new time threshold.

<a id="senpi.data_manip.filters.BAFilter.set_height"></a>

#### set\_height

```python
def set_height(height: int)
```

Set the height of the event camera frame.

**Arguments**:

- `height` (`int`): The new height.

<a id="senpi.data_manip.filters.BAFilter.set_width"></a>

#### set\_width

```python
def set_width(width: int)
```

Set the width of the event camera frame.

**Arguments**:

- `width` (`int`): The new width.

<a id="senpi.data_manip.filters.IEFilter"></a>

### IEFilter Objects

```python
class IEFilter(Filter)
```
A class to filter event-based vision data using an inceptive event (IE) filter based on temporal and spatial constraints.

**E-MLB:** https://ieeexplore.ieee.org/document/10078400 <br/>

**Based on the following paper (as referenced in E-MLB):** <br/>
R. Baldwin, M. Almatrafi, J. R. Kaufman, V. Asari, and K. Hirakawa, “Inceptive event time-surfaces for object classification using neuromorphic
cameras,” in Proc. Int. Conf. Image Anal. Recognit., 2019, pp. 395-403. <br/>

**Paper link:** <br/>
https://arxiv.org/abs/2002.11656 <br/>

**Filter Functionality**<br/>
The IE filter will only classify events $e_i$ as signal if $t_i - t_{i - 1} > \tau^{-}$ and $t_{i + 1} - t_i < \tau^{+}$ where $t_i \in T(x, y, p)$, the set of all timestamps occuring at a given $(x, y)$ pixel location and a polarity $p$.

Either threshold can be set to `None` if the restriction corresponding to that threshold is not desired to be imposed on the data.

**Sidenote**<br/>
Unless a PyTorch tensor is passed in or `reset_indices` is set to `True` in the case where a Pandas DataFrame is passed in, the `filter` method for `IEFilter` will return the rows of the original data structure (Pandas DataFrame or NumPy structured array) corresponding to signal events **in-place**.

**Arguments**:

- `width` (`int`): The width of the event frame.
- `height` (`int`): The height of the event frame.
- `thresh_negative` (`Union[int, float]`): The negative time threshold for filtering events ($\tau^{-}$).
- `thresh_positive` (`Union[int, float]`): The positive time threshold for filtering events ($\tau^{+}$).

<a id="senpi.data_manip.filters.IEFilter.__init__"></a>

#### \_\_init\_\_

```python
def __init__(thresh_negative: Union[float, int], thresh_positive: Union[float, int], height: int, width: int)
```

Initialize the inceptive events filter (IE Filter).

**Arguments**:

- `width` (`int`): The width of the event frame.
- `height` (`int`): The height of the event frame.
- `thresh_negative` (`Union[int, float]`): The negative time threshold for filtering events ($\tau^{-}$).
- `thresh_positive` (`Union[int, float]`): The positive time threshold for filtering events ($\tau^{+}$).

<a id="senpi.data_manip.filters.IEFilter.filter_events"></a>

#### filter\_events

```python
def filter_events(
    events: Union[pd.DataFrame, np.ndarray, torch.Tensor],
    order: List[str] = None,
    reset_indices: bool = True
) -> Union[pd.DataFrame, np.ndarray, torch.Tensor]
```

Filter the input events using the IE filter based on temporal constraints defined by `thresh_negative` and `thresh_positive`.

**Arguments**:

- `events` (`Union[pd.DataFrame, np.ndarray, torch.Tensor]`): The events to filter.
- `order` (`List[str], optional`): The order of the columns of an inputted events torch.Tensor. By default, this is set to None and further handled by helper methods.
- `reset_indices` (`bool, optional`): Whether to reset the indices of the returned Pandas dataframe (if a Pandas dataframe is inputted) after filtering it.

**Returns**:

`Union[pd.DataFrame, np.ndarray, torch.Tensor]`: The filtered events in the same format as input (Pandas DataFrame, NumPy array, or PyTorch Tensor).

<a id="senpi.data_manip.filters.IEFilter.filter_events_pd"></a>

#### filter\_events\_pd

```python
def filter_events_pd(events: pd.DataFrame,
                     reset_indices: bool) -> pd.DataFrame
```

Filter a Pandas DataFrame of events using the IE filter. Assumes that row indices of the data frame have not been modified.

**Arguments**:

- `events` (`pd.DataFrame`): The events to filter.
- `reset_indices` (`bool, optional`): Whether to reset the indices of the returned DataFrame after filtering.

**Returns**:

`pd.DataFrame`: The filtered events.

<a id="senpi.data_manip.filters.IEFilter.filter_events_np"></a>

#### filter\_events\_np

```python
def filter_events_np(events: np.ndarray) -> np.ndarray
```

Filter a numpy array of events using the IE filter.

**Arguments**:

- `events` (`np.ndarray`): The events to filter.

**Returns**:

`np.ndarray`: The filtered events.

<a id="senpi.data_manip.filters.IEFilter.filter_events_tensor"></a>

#### filter\_events\_tensor

```python
def filter_events_tensor(events: torch.Tensor,
                         order: List[str] = None) -> torch.Tensor
```

Filter a torch.Tensor of events using the IE filter. This method converts the inputted PyTorch tensor into an intermediate Pandas DataFrame and invokes `filter_events_pd` on it. The filtered dataframe is then converted back to a torch.Tensor and returned.

**Arguments**:

- `events` (`torch.Tensor`): The events to filter.
- `order` (`List[str], optional`): The order of the columns of the inputted events tensor. Defaults to ['b', 't', 'x', 'y', 'p'].

**Returns**:

`torch.Tensor`: The filtered events.

<a id="senpi.data_manip.filters.IEFilter.filter_frames_tensor"></a>

#### filter\_frames\_tensor

```python
def filter_frames_tensor(frames: torch.Tensor,
                         inplace: bool,
                         device: torch.device = None,
                         polarity_agnostic: bool = False) -> torch.Tensor
```

Apply the IE filter to a 4D tensor of frames (frame volume of events).

**Arguments**:

- `frames` (`torch.Tensor`): A 4D torch.Tensor with shape (N, C, Y, X), where N is the number of frames, C is the channel (should be 1), Y is the height, and X is the width.
- `inplace` (`bool`): Whether to modify the tensor in place or return a new filtered tensor.
- `device` (`torch.device, optional`): The device to run the filtering on. If not specified, defaults to 'cuda' if available.
- `polarity_agnostic` (`bool, optional`): If True, the filter will treat both positive and negative polarities equally by applying the filtering operation on the absolute values of the input frames. If False, the filter will handle positive and negative polarities independently, applying polarity-specific masks during filtering.

**Returns**:

`torch.Tensor`: A torch.Tensor containing the filtered frames.

<a id="senpi.data_manip.filters.IEFilter.set_thresh_negative"></a>

#### set\_thresh\_negative

```python
def set_thresh_negative(thresh_negative: Union[float, int])
```

Set the negative time threshold of the IE filter.

**Arguments**:

- `height` (`Union[float, int]`): The new negative threshold.

<a id="senpi.data_manip.filters.IEFilter.set_thresh_positive"></a>

#### set\_thresh\_positive

```python
def set_thresh_positive(thresh_positive: Union[float, int])
```

Set the postiive time threshold of the IE filter.

**Arguments**:

- `width` (`Union[float, int]`): The new positive threshold.

<a id="senpi.data_manip.filters.IEFilter.set_height"></a>

#### set\_height

```python
def set_height(height: int)
```

Set the height of the event camera frame.

**Arguments**:

- `height` (`int`): The new height.

<a id="senpi.data_manip.filters.IEFilter.set_width"></a>

#### set\_width

```python
def set_width(width: int)
```

Set the width of the event camera frame.

**Arguments**:

- `width` (`int`): The new width.

<a id="senpi.data_manip.filters.YNoiseFilter"></a>

### YNoiseFilter Objects

```python
class YNoiseFilter(Filter)
```
A class to filter event-based vision data using the YNoise filter (event density and hot pixels) based on temporal and spatial constraints. The space window size should be odd.

**E-MLB:** https://ieeexplore.ieee.org/document/10078400 <br/>

**Based on the following paper (as referenced in E-MLB):** <br/>
Y. Feng et al., “Event density based denoising method for dynamic vision sensor,” Appl. Sci., vol. 10, no. 6, 2020, Art. no. 2024. <br/>

**Paper link:** <br/>
https://doi.org/10.3390/app10062024 <br/>

**Sidenote**<br/>
Unless a PyTorch tensor is passed in or `reset_indices` is set to `True` in the case where a Pandas DataFrame is passed in, the `filter` method for `YNoiseFilter` will return the rows of the original data structure (Pandas DataFrame or NumPy structured array) corresponding to signal events **in-place**.

**Arguments**:

- `time_context_delta` (`Union[int, float]`): The time context delta ($\Delta t$) for the spatiotemporal neighborhood around each event ($\Omega^{L}_{\Delta t}$ or $\Omega^{3}_{\Delta t}$) on which to apply the event density filter and hot pixel filter.
- `space_window_size` (`int`): The space window size ($L$) for the spatiotemporal neighborhood around each event ($\Omega^{L}_{\Delta t}$) on which to apply the event density filter. This should be odd; if it isn't, then it will be set equal to the next highest odd number.
- `height` (`int`): The height of the event frame.
- `width` (`int`): The width of the event frame.
- `density_threshold` (`Union[int, float]`): The event density threshold for filtering events ($\Psi$).

<a id="senpi.data_manip.filters.YNoiseFilter.__init__"></a>

#### \_\_init\_\_

```python
def __init__(time_context_delta: Union[float, int], space_window_size: int, density_threshold: Union[int, float], height: int, width: int)
```

Initialize the YNoise filter.

**Arguments**:

- `time_context_delta` (`Union[int, float]`): The time context delta ($\Delta t$) for the spatiotemporal neighborhood around each event ($\Omega^{L}_{\Delta t}$ or $\Omega^{3}_{\Delta t}$) on which to apply the event density filter and hot pixel filter.
- `space_window_size` (`int`): The space window size ($L$) for the spatiotemporal neighborhood around each event ($\Omega^{L}_{\Delta t}$) on which to apply the event density filter. This should be odd; if it isn't, then it will be set equal to the next highest odd number.
- `height` (`int`): The height of the event frame.
- `width` (`int`): The width of the event frame.
- `density_threshold` (`Union[int, float]`): The event density threshold for filtering events ($\Psi$).

<a id="senpi.data_manip.filters.YNoiseFilter.filter_events"></a>

#### filter\_events

```python
def filter_events(
    events: Union[pd.DataFrame, np.ndarray, torch.Tensor],
    order: List[str] = None,
    reset_indices: bool = True,
    hot_pixel_filter: bool = True
) -> Union[pd.DataFrame, np.ndarray, torch.Tensor]
```

Filter events using the YNoise filter. This includes event density filtering and optional hot pixel filtering.

**Arguments**:

- `events` (`Union[pd.DataFrame, np.ndarray, torch.Tensor]`): The events to filter.
- `order` (`List[str], optional`): The order of the columns of an inputted events torch.Tensor. By default, this is set to None and further handled by helper methods.
- `reset_indices` (`bool, optional`): Whether to reset the indices of the returned Pandas dataframe (if a Pandas dataframe is inputted) after filtering it.
- `hot_pixel_filter` (`bool`): Whether to apply the hot pixel filter to the result of the initial event density filter on the inputted events.

**Returns**:

`Union[pd.DataFrame, np.ndarray, torch.Tensor]`: The filtered events in the same format as the input.

<a id="senpi.data_manip.filters.YNoiseFilter.filter_events_pd"></a>

#### filter\_events\_pd

```python
def filter_events_pd(events: pd.DataFrame,
                     reset_indices: bool,
                     hot_pixel_filter: bool = True) -> pd.DataFrame
```

Filter a Pandas DataFrame of events using the YNoise filter. Assumes that row indices of the data frame have not been modified.

**Arguments**:

- `events` (`pd.DataFrame`): The events to filter.
- `reset_indices` (`bool, optional`): Whether to reset the indices of the returned DataFrame after filtering.
- `hot_pixel_filter` (`bool`): Whether to apply the hot pixel filter to the result of the initial event density filter on the inputted events.

**Returns**:

`pd.DataFrame`: The filtered events.

<a id="senpi.data_manip.filters.YNoiseFilter.filter_events_np"></a>

#### filter\_events\_np

```python
def filter_events_np(events: np.ndarray,
                     hot_pixel_filter: bool = True) -> np.ndarray
```

Filter a numpy structured array of events using the YNoise filter.

**Arguments**:

- `events` (`np.ndarray`): The events to filter.
- `hot_pixel_filter` (`bool`): Whether to apply the hot pixel filter to the result of the initial event density filter on the inputted events.

**Returns**:

`np.ndarray`: The filtered events.

<a id="senpi.data_manip.filters.YNoiseFilter.filter_events_tensor"></a>

#### filter\_events\_tensor

```python
def filter_events_tensor(events: torch.Tensor,
                         hot_pixel_filter: bool = True,
                         order: List[str] = None) -> torch.Tensor
```

Filter a torch.Tensor of events using the YNoise filter. This method converts the inputted PyTorch tensor into an intermediate Pandas DataFrame and invokes `filter_events_pd` on it. The filtered dataframe is then converted back to a torch.Tensor and returned.

**Arguments**:

- `events` (`torch.Tensor`): The events to filter.
- `hot_pixel_filter` (`bool`): Whether to apply the hot pixel filter to the result of the initial event density filter on the inputted events.
- `order` (`List[str], optional`): The order of the columns of the inputted events tensor. Defaults to ['b', 't', 'x', 'y', 'p'].

**Returns**:

`torch.Tensor`: The filtered events.

<a id="senpi.data_manip.filters.YNoiseFilter.filter_frames_tensor"></a>

#### filter\_frames\_tensor

```python
def filter_frames_tensor(frames: torch.Tensor,
                         inplace: bool,
                         hot_pixel_filter: bool = True,
                         device: torch.device = None) -> torch.Tensor
```

Filter a torch.Tensor frame volume of events using the YNoiseFilter filter.

**Arguments**:

- `frames` (`torch.Tensor`): The input tensor of event frames to be filtered. Expected shape: (N, C, X, Y).
- `inplace` (`bool`): Whether to apply the filter in-place (modifies the input tensor directly) or out-of-place (returns a new tensor).
- `hot_pixel_filter` (`bool, optional`): Whether to apply the second phase hot-pixel filter after the YNoise filter. Defaults to True.
- `device` (`torch.device, optional`): The device on which to perform the computations. If not provided, defaults to "cuda" if available.

**Returns**:

`torch.Tensor`: The filtered event frames tensor.

<a id="senpi.data_manip.filters.YNoiseFilter.set_density_threshold"></a>

#### set\_density\_threshold

```python
def set_density_threshold(density_threshold: Union[float, int])
```

Set the density threshold of the YNoise filter.

**Arguments**:

- `density_threshold` (`Union[float, int]`): The new density threshold.

<a id="senpi.data_manip.filters.YNoiseFilter.set_time_context_delta"></a>

#### set\_time\_context\_delta

```python
def set_time_context_delta(time_context_delta: Union[float, int])
```

Set the density time_context_delta of the YNoise filter.

**Arguments**:

- `time_context_delta` (`Union[float, int]`): The new time context delta.

<a id="senpi.data_manip.filters.YNoiseFilter.set_space_window_size"></a>

#### set\_space\_window\_size

```python
def set_space_window_size(space_window_size: int)
```

Set the density space_window_size of the YNoise filter.

**Arguments**:

- `space_window_size` (`int`): The new space window size.

<a id="senpi.data_manip.filters.YNoiseFilter.set_height"></a>

#### set\_height

```python
def set_height(height: int)
```

Set the height of the event camera frame.

**Arguments**:

- `height` (`int`): The new height.

<a id="senpi.data_manip.filters.YNoiseFilter.set_width"></a>

#### set\_width

```python
def set_width(width: int)
```

Set the width of the event camera frame.

**Arguments**:

- `width` (`int`): The new width.

<a id="senpi.data_manip.conversions"></a>

## Conversions (senpi.data\_manip.conversions)

<a id="senpi.data_manip.conversions.df_seconds_to_microseconds"></a>

### df\_seconds\_to\_microseconds

```python
def df_seconds_to_microseconds(df: pd.DataFrame, new_dtype: np.dtype = None)
```

Convert the time column of a DataFrame from seconds to microseconds.

This function multiplies the time values in the 't' column of the DataFrame (in-place) by 1e6 to convert the time unit from seconds to microseconds. Optionally, it can also change the data type of the 't' column.

**Arguments**:

- `df` (`pd.DataFrame`): The DataFrame containing the time data to convert.
- `new_dtype` (`np.dtype, optional`): The new data type for the 't' column after conversion (default is None, which uses np.int64).

**Returns**:

`pd.DataFrame`: The DataFrame with the time column converted to microseconds.

<a id="senpi.data_manip.conversions.df_microseconds_to_seconds"></a>

### df\_microseconds\_to\_seconds

```python
def df_microseconds_to_seconds(df: pd.DataFrame, new_dtype: np.dtype = None)
```

Convert the time column of a DataFrame from microseconds to seconds.

This function divides the time values in the 't' column of the DataFrame (in-place) by 1e6 to convert the time unit from microseconds to seconds. Optionally, it can also change the data type of the 't' column.

**Arguments**:

- `df` (`pd.DataFrame`): The DataFrame containing the time data to convert.
- `new_dtype` (`np.dtype, optional`): The new data type for the 't' column after conversion (default is None, which uses np.float64).

**Returns**:

`pd.DataFrame`: The DataFrame with the time column converted to seconds.

<a id="senpi.data_manip.conversions.event_stream_to_frame_vol"></a>

### event\_stream\_to\_frame\_vol

```python
def event_stream_to_frame_vol(event_stream: torch.Tensor,
                              num_frames: int,
                              width: int,
                              height: int,
                              order: List[str] = None) -> torch.Tensor
```

Convert a tensor containing a standard event stream ((N, 4) or (N, 5)) into a frame volume representation. 

The output of this function will be a tensor with dimensions (N, C, H, W), where N is the number of frames (each frame corresponds to a single time step of events; if the event stream's
timestamp measurements are in microseconds, then each frame corresponds to microsecond of events), C is the number of channels (set to 1), H is the frame height, and W is the
frame width.

**Arguments**:

- `event_stream` (`torch.Tensor`): The PyTorch tensor containing the event stream to convert.
- `num_frames` (`int`): The number of frames in the returned volume (N).
- `width` (`int`): The width of each frame (W).
- `height` (`int`): The height of each frame (H).
- `order` (`List[str], optional`): The order of the columns of the inputted events tensor. Defaults to ['t', 'x', 'y', 'p'].

**Returns**:

`torch.Tensor`: A tensor containing the frame volume representation of the event stream.

<a id="senpi.data_manip.conversions.frame_vol_to_event_stream"></a>

### frame\_vol\_to\_event\_stream

```python
def frame_vol_to_event_stream(
        frames: torch.Tensor,
        ret_order: List[str] = ['t', 'x', 'y', 'p']) -> torch.Tensor
```

Convert a frame volume representation ((N, 1, H, W) or (N, H, W)) back into an event stream format (dimensions (N, 4)).

The input to this function is a frame volume where each frame corresponds to a time step of events (e.g., each frame could represent a microsecond of events, depending on the 
timestamp resolution in the original event stream). The function extracts the non-zero entries (events) from the volume and converts them back into an event stream representation.

**Arguments**:

- `frames` (`torch.Tensor`): The PyTorch tensor containing the frame volume to convert. Expected dimensions are (N, 1, H, W) or (N, H, W), where N is the number of frames, H is the frame height,
and W is the frame width.
- `ret_order` (`List[str], optional`): The desired order of the columns in the outputted event stream tensor. Defaults to ['t', 'x', 'y', 'p'].

**Returns**:

`torch.Tensor`: A tensor containing the event stream representation of the frame volume, with the columns ordered as specified by `ret_order`.

<a id="senpi.data_manip.conversions.to_1D_tensor"></a>

### to\_1D\_tensor

```python
def to_1D_tensor(events, device='cuda' if torch.cuda.is_available else 'cpu')
```

Convert event data to a 1D PyTorch tensor.

Converts input event data from either pandas DataFrame or NumPy array format to a PyTorch tensor.
If the input is already a tensor, it is returned unchanged.

**Arguments**:

- `events` (`Union[pd.DataFrame, np.ndarray, torch.Tensor]`): The event data to convert. Can be a pandas DataFrame, NumPy array, or PyTorch tensor.
- `device` (`str`): The device to store the tensor on ('cuda' or 'cpu'). Defaults to 'cuda' if available, else 'cpu'.

**Returns**:

`torch.Tensor`: The event data as a PyTorch tensor on the specified device.

<a id="senpi.data_manip.conversions.to_1D_dataframe"></a>

### to\_1D\_dataframe

```python
def to_1D_dataframe(events, order=['t', 'x', 'y', 'p'])
```

Convert event data to a pandas DataFrame.

Converts input event data from either PyTorch tensor or NumPy array format to a pandas DataFrame.
If the input is already a DataFrame, it is returned unchanged.

**Arguments**:

- `events` (`Union[torch.Tensor, np.ndarray, pd.DataFrame]`): The event data to convert. Can be a PyTorch tensor, NumPy array, or pandas DataFrame.
- `order` (`List[str], optional`): The column names for the DataFrame in order. Defaults to ['t', 'x', 'y', 'p'].

**Returns**:

`pd.DataFrame`: The event data as a pandas DataFrame with specified column names.

<a id="senpi.data_manip.conversions.to_1D_numpy"></a>

### to\_1D\_numpy

```python
def to_1D_numpy(events)
```

Convert event data to a NumPy array.

Converts input event data from either PyTorch tensor or pandas DataFrame format to a NumPy array.
If the input is already a NumPy array, it is returned unchanged.

**Arguments**:

- `events` (`Union[torch.Tensor, pd.DataFrame, np.ndarray]`): The event data to convert. Can be a PyTorch tensor, pandas DataFrame, or NumPy array.

**Returns**:

`np.ndarray`: The event data as a NumPy array.

<a id="senpi.data_manip.preprocessing"></a>

## Preprocessing (senpi.data\_manip.preprocessing)

<a id="senpi.data_manip.preprocessing.np_events_to_tensor"></a>

#### np\_events\_to\_tensor

```python
def np_events_to_tensor(events: np.ndarray,
                        batch_index: int = 0,
                        transform_polarity: bool = True,
                        stack_ret_order: List[str] = ['b', 't', 'x', 'y', 'p'],
                        stack_tensors: bool = True,
                        dtypes: Dict[str, torch.dtype] = None,
                        device: torch.device = "cpu") -> torch.Tensor
```

Convert a structured NumPy array of events to a stacked tensor.

This function accepts a NumPy array where columns represent event attributes ('t', 'x', 'y', and 'p') and converts it into a PyTorch tensor.
The resulting tensor can be stacked in a specified order and includes options for transforming polarity values.

**Arguments**:

- `events` (`np.ndarray`): The structured NumPy array of events.
- `batch_index` (`int`): Index to assign to the batch dimension.
- `transform_polarity` (`bool`): Whether to transform the polarity values such that -1 represents OFF events and 1 represents ON events (from 0 and 1). Note: if the polarities are already represented by -1s and 1s, they will remain the same no matter what `transform_polarity` is set to.
- `stack_ret_order` (`List[str]`): The order of dimensions in the resulting tensor, e.g., ['b', 't', 'x', 'y', 'p'].
- `stack_tensors` (`bool`): Whether to stack the tensors into a single tensor (true) or return them as a tuple of tensors (false).
- `dtypes` (`Dict[str, torch.dtype], optional`): Optional dictionary to specify the data types for each tensor.
- `device` (`torch.device`): The device to move the tensor(s) to.

**Returns**:

`torch.Tensor`: The tensor representing the events, stacked in the specified order.

<a id="senpi.data_manip.preprocessing.events_tensor_batch_to_img"></a>

#### events\_tensor\_batch\_to\_img

```python
def events_tensor_batch_to_img(
        events: torch.Tensor,
        batch_size: int,
        height: int,
        width: int,
        time_surface_mode: str = "exponential_decay_sum",
        separate_polarity: bool = False,
        displacement_mode: bool = False,
        window_size: int = 5,
        decay_factor: float = 1e9,
        prev_dict=None,
        offset: float = 0,
        frequency: float = 60,
        tolerance: float = 100,
        penalty: float = 0,
        time_unit: float = TimeUnit.MICROSECONDS,
        stack_order: List[str] = ['b', 't', 'x', 'y', 'p']) -> torch.Tensor
```

Convert a batch of events to an image (time surface representation).

This function converts a tensor batch of events into an image based on the specified time surface representation mode.
It can handle both separate polarities (2 separate channels) and combined polarity (1 channel) modes.
Additionally, it will return the time surface within a displacement window if `displacement_mode` is True.

**Return Dictionary Keys and Values Description:**

- **`ret_img_tensor`**
  - **Type**: `torch.Tensor`
  - **Description**: The resulting image tensor(s) generated from the event data based on the specified time surface mode. The shape and contents depend on the parameters such as `separate_polarity` and the selected `time_surface_mode`.

- **`p_final_t`**
  - **Type**: `torch.Tensor`
  - **Description**: The timestamp of the last positive event in the batch. Only present if `separate_polarity` is `True`.

- **`n_final_t`**
  - **Type**: `torch.Tensor`
  - **Description**: The timestamp of the last negative event in the batch. Only present if `separate_polarity` is `True`.

- **`p_num_events`**
  - **Type**: `int`
  - **Description**: The total number of positive events in the batch. Only present if `separate_polarity` is `True`.

- **`n_num_events`**
  - **Type**: `int`
  - **Description**: The total number of negative events in the batch. Only present if `separate_polarity` is `True`.

- **`running_p_num_events`**
  - **Type**: `int`
  - **Description**: The cumulative count of positive events if a running cumulative surface is being maintained. Only present if `separate_polarity` is `True` and `prev_dict` is provided.

- **`running_n_num_events`**
  - **Type**: `int`
  - **Description**: The cumulative count of negative events if a running cumulative surface is being maintained. Only present if `separate_polarity` is `True` and `prev_dict` is provided.

- **`final_t`**
  - **Type**: `torch.Tensor`
  - **Description**: The timestamp of the last event in the batch. Only present if `separate_polarity` is `False`.

- **`num_events`**
  - **Type**: `int`
  - **Description**: The total number of events in the batch. Only present if `separate_polarity` is `False`.

- **`running_num_events`**
  - **Type**: `int`
  - **Description**: The cumulative count of events if a running cumulative surface is being maintained. Only present if `separate_polarity` is `False` and `prev_dict` is provided.

- **`running_time_surface`**
  - **Type**: `torch.Tensor`
  - **Description**: The cumulative time surface image tensor(s). Present only if a running cumulative surface is being maintained and `prev_dict` is provided.

**Arguments**:

- `events` (`torch.Tensor`): The tensor batch of events.
- `batch_size` (`int`): The number of events in the batch.
- `height` (`int`): The height of the event camera and the resulting image.
- `width` (`int`): The width of the event camera and the resulting image.
- `time_surface_mode` (`str`): The mode for generating the time surface: one of "count", "average", "average_absolute", "exponential_decay_sum", "exponential_decay_max," "most_recent," or "frequency."
- `separate_polarity` (`bool`): Whether to separate positive and negative polarities into different images.
- `displacement_mode` (`bool`): Whether to return the time surface within a displacement window around the final event.
- `window_size` (`int`): The size of the displacement window. Must be an odd number.
- `decay_factor` (`float`): The decay factor for the exponential decay mode.
- `prev_dict` (`dict, optional`): A dictionary containing previous cumulative surfaces and metadata, used for running cumulative surface computation.
- `offset` (`float`): The offset for the frequency mode, in the same time units as the event timestamps.
- `frequency` (`float`): The frequency for the frequency mode, in cycles per second (Hz).
- `tolerance` (`float`): The tolerance for determining the frequency kernel width.
- `penalty` (`float`): The penalty value for events outside the frequency kernel.
- `time_unit` (`float`): The unit of time for the event timestamps.
- `stack_order` (`List[str]`): The order of dimensions in the events tensor, e.g., ['b', 't', 'x', 'y', 'p'].

**Returns**:

`dict`: A dictionary with the resulting image tensor(s) based on the time surface mode, and metadata for cumulative surfaces.

<a id="senpi.data_manip.preprocessing.events_tensor_batch_to_vol"></a>

#### events\_tensor\_batch\_to\_vol

```python
def events_tensor_batch_to_vol(
    events: torch.Tensor,
    total_batch_size: int,
    height: int,
    width: int,
    start_times: Union[List, np.ndarray],
    durations: Union[List, np.ndarray],
    nbins: int,
    time_surface_mode: str = "exponential_decay_sum",
    displacement_mode: bool = False,
    window_size: int = 5,
    separate_polarity: bool = False,
    decay_factor: float = 1e9,
    running_cumulative_surface: bool = False,
    freq_offset: Union[int, float] = 0,
    frequency: float = 60,
    tolerance: float = 100,
    penalty: float = 0,
    time_unit: Union[str, TimeUnit] = TimeUnit.MICROSECONDS,
    stack_order: List[str] = ['b', 't', 'x', 'y', 'p'],
    ret_generator: bool = False
) -> Union[torch.Tensor, Generator[torch.Tensor, None, None]]
```

Convert a batch of events to a volumetric representation.

This function converts a tensor batch of events into a 3D volume based on the specified time surface representation mode.
It can handle both separate polarities (2 separate channels) and combined polarity (1 channel) modes.
Additionally, it will return the time surface within a displacement window if `displacement_mode` is True.

There should be `nbins` values each in both `start_times` and `durations`.

For each time bin specified by `start_times` and its corresponding duration in `durations`, this method invokes the 
`events_tensor_batch_to_img` method on the subset events in the `events` tensor falling in the time bin ($s_i \leq t_i \leq s_i + d_i$)
and appends the result (of shape $(C, H, W)$) to a tensor list. The tensors in this tensor list are stacked at the end are stacked
along the 0th dimension, making the returned tensor have shape $($`nbins`$, C, H, W)$, where `nbins` is the number of time bins, $C$ is the number of channels (2 if `separate_polarity` is `True`, 1 if not), $H$ is the height, and $W$ is the width.

**Arguments**:

- `events` (`torch.Tensor`): The tensor batch of events.
- `total_batch_size` (`int`): The total number of events in the batch.
- `height` (`int`): The height of the event camera and the resulting volume.
- `width` (`int`): The width of the event camera and the resulting volume.
- `start_times` (`Union[List, np.ndarray]`): The start times for each time bin.
- `durations` (`Union[List, np.ndarray]`): The durations for each time bin.
- `nbins` (`int`): The number of bins for the temporal axis of the volume.
- `time_surface_mode` (`str`): The mode for generating the time surface: one of "count", "average", "average_absolute", "exponential_decay_sum", "exponential_decay_max," "most_recent," or "frequency."
- `displacement_mode` (`bool`): Whether to return the time surface within a displacement window around the final event.
- `window_size` (`int`): The size of the displacement window. Must be an odd number.
- `separate_polarity` (`bool`): Whether to separate positive and negative polarities into different volumes.
- `decay_factor` (`float`): The decay factor for the exponential decay mode.
- `running_cumulative_surface` (`bool`): Whether to use running cumulative surface computation.
- `freq_offset` (`Union[int, float]`): The offset for the frequency mode, in the same time units as the event timestamps.
- `frequency` (`float`): The frequency for the frequency mode, in cycles per second (Hz).
- `tolerance` (`float`): The tolerance for determining the frequency kernel width.
- `penalty` (`float`): The penalty value for events outside the frequency kernel.
- `time_unit` (`Union[str, TimeUnit]`): The unit of time for the event timestamps.
- `stack_order` (`List[str]`): The order of dimensions in the events tensor, e.g., ['b', 't', 'x', 'y', 'p'].
- `ret_generator` (`bool`): Whether to return a generator yielding the image time surface representations one by one.

**Returns**:

`Union[torch.Tensor, Generator[torch.Tensor, None, None]]`: A 4D tensor of shape (`nbins`, `C`, `H`, `W`) where each of the `nbins` 3D tensors is an image formed by the aggregation of events within the time bins delineated by `start_times` and `durations`, or a generator yielding these tensors one by one.

<a id="senpi.data_manip.preprocessing.list_np_events_to_tensor"></a>

#### list\_np\_events\_to\_tensor

```python
def list_np_events_to_tensor(list_events: List[np.ndarray],
                             transform_polarity: bool = True,
                             device: torch.device = "cpu",
                             stack_ret_order: List[str] = [
                                 'b', 't', 'x', 'y', 'p'
                             ],
                             ret_dtype: torch.dtype = None) -> torch.Tensor
```

Convert a list of structured NumPy arrays of events (representing different batches of events) to a single tensor (with or without a batch column).

This function takes a list where each element is a NumPy array of events and converts it into a single PyTorch tensor.
It supports optional polarity transformation and data type conversion.

**Arguments**:

- `list_events` (`List[np.ndarray]`): List of structured NumPy arrays of events.
- `transform_polarity` (`bool`): Whether to transform the polarity values.
- `device` (`torch.device`): The device to move the resulting tensor to.
- `stack_ret_order` (`List[str]`): The order of dimensions in the resulting tensor, e.g., ['b', 't', 'x', 'y', 'p'].
- `ret_dtype` (`torch.dtype, optional`): Optional data type for the resulting tensor.

**Returns**:

`torch.Tensor`: The combined tensor representing all events.

<a id="senpi.data_manip.preprocessing.events_tensor_to_np_list"></a>

#### events\_tensor\_to\_np\_list

```python
def events_tensor_to_np_list(
        events: torch.Tensor,
        batch_size: int,
        group_batch_inds: bool = True,
        stack_ret_order: List[str] = ['b', 't', 'x', 'y', 'p'],
        dtypes: Dict[str, np.dtype] = None) -> List[np.ndarray]
```

Convert a tensor of events to a list of NumPy arrays (each element of the returned list represents a different batch of events).

This function converts a PyTorch tensor of events into a list of structured NumPy arrays, each representing a batch of events.
The resulting arrays have essentially the same structure as the original events tensor (they do not have a batch index column).

**Arguments**:

- `events` (`torch.Tensor`): The tensor of events to convert.
- `batch_size` (`int`): The size of each batch in the resulting list of NumPy arrays.
- `group_batch_inds` (`bool`): Whether to group events by batch index.
- `stack_ret_order` (`List[str]`): The order of dimensions in the events tensor, e.g., ['b', 't', 'x', 'y', 'p'].
- `dtypes` (`Dict[str, np.dtype], optional`): Optional dictionary to specify the data types for each column in the NumPy arrays.

**Returns**:

`List[np.ndarray]`: A list of NumPy arrays representing batches of events.

<a id="senpi.data_manip.preprocessing.add_batch_col"></a>

#### add\_batch\_col

```python
def add_batch_col(events: torch.Tensor,
                  batch_size: int,
                  modify_original: bool = False) -> torch.Tensor
```

Add a batch index column to the events tensor.

This function adds a new column to the events tensor that represents the batch index for each event.
The original tensor can either be modified in place or a new tensor can be returned.

**Arguments**:

- `events` (`torch.Tensor`): The tensor of events to modify.
- `batch_size` (`int`): The size of each batch.
- `modify_original` (`bool`): Whether to modify the original tensor in place.

**Returns**:

`torch.Tensor`: The tensor with the added batch index column.

<a id="senpi.data_manip.preprocessing.partition_np_list_batch"></a>

#### partition\_np\_list\_batch

```python
def partition_np_list_batch(events: np.ndarray,
                            batch_size: int) -> List[np.ndarray]
```

Partition a NumPy array into a list of batches.

This function divides a NumPy array into smaller arrays (batches) of a specified size.

**Arguments**:

- `events` (`np.ndarray`): The NumPy array to partition.
- `batch_size` (`int`): The size of each batch.

**Returns**:

`List[np.ndarray]`: A list of NumPy arrays, each representing a batch.

## Computation (senpi.data\_manip.computation)
<a id="senpi.data_manip.computation.has_parameter"></a>

#### has\_parameter

```python
def has_parameter(func, param_name)
```

Check if the function has a parameter with the given name.

**Arguments**:

- `func` (`function`): The function to check.
- `param_name` (`str`): The name of the parameter to look for.

**Returns**:

`bool`: True if the parameter exists, False otherwise.

<a id="senpi.data_manip.computation.SequentialCompute"></a>

### SequentialCompute Objects

```python
class SequentialCompute()
```

A class to sequentially apply a list of computations (algorithms or filters) on event data. Note that in the case that a computation is a filter, the class will by default assume that the `filter_events` method should be invoked on the data. The `filter_frames_tensor` method can instead by invoked by setting `filter_frame_mode` to `True` in the `param_dict` dictionary of the `SequentialCompute` object.

**Attributes**:

- `computation_list` _list_ - A list of computations to apply.
- `param_dict` _dict_ - A dictionary of parameters to pass to each computation.
  

**Methods**:

- `set_param_dict(param_dict)` - Sets the parameter dictionary.
- `__call__(events)` - Applies the list of computations to the event data.
- `compute(events)` - Applies the list of computations to the event data.

<a id="senpi.data_manip.computation.SequentialCompute.__init__"></a>

#### \_\_init\_\_

```python
def __init__(*computation_list)
```

Initializes the SequentialCompute class with a list of computations.

**Arguments**:

- `computation_list` (`list`): A list of computations (algorithms or filters) to apply.

<a id="senpi.data_manip.computation.SequentialCompute.set_param_dict"></a>

#### set\_param\_dict

```python
def set_param_dict(param_dict)
```

Sets the parameter dictionary to be used for each computation.

**Arguments**:

- `param_dict` (`dict`): A dictionary of parameters to pass to each computation.

<a id="senpi.data_manip.computation.SequentialCompute.__call__"></a>

#### \_\_call\_\_

```python
def __call__(
    events: Union[torch.Tensor, np.ndarray, pd.DataFrame]
) -> Union[torch.Tensor, np.ndarray, pd.DataFrame]
```

Applies the list of computations to the event data by calling the compute method.

**Arguments**:

- `events` (`Union[torch.Tensor, np.ndarray, pd.DataFrame]`): The event data to process.

**Returns**:

`Union[torch.Tensor, np.ndarray, pd.DataFrame]`: The processed event data.

<a id="senpi.data_manip.computation.SequentialCompute.compute"></a>

#### compute

```python
def compute(
    events: Union[torch.Tensor, np.ndarray, pd.DataFrame]
) -> Union[torch.Tensor, np.ndarray, pd.DataFrame]
```

Applies the list of computations to the event data.

**Arguments**:

- `events` (`Union[torch.Tensor, np.ndarray, pd.DataFrame]`): The event data to process.

**Returns**:

`Union[torch.Tensor, np.ndarray, pd.DataFrame]`: The processed event data.

<a id="Data_Visualization"></a>

# Data Visualization

<a id="senpi.data_vis.visualization"></a>

## Visualization (senpi.data\_vis.visualization)

<a id="senpi.data_vis.visualization.plot_events"></a>

#### plot\_events

```python
def plot_events(events: Union[torch.Tensor, pd.DataFrame, np.ndarray],
                order: List[str] = None,
                separate_polarity: bool = True)
```

Plot event data in a 3D scatter plot (axes: pixel x position, pixel y position, and time).

**Arguments**:

- `events` (`Union[torch.Tensor, pd.DataFrame, np.ndarray]`): The event data to plot. Can be a torch.Tensor, pd.DataFrame, or np.ndarray.
- `order` (`List[str], optional`): The order of the parameters in the event data. Defaults to ['b', 't', 'x', 'y', 'p'] if None.
- `separate_polarity` (`bool, optional`): Whether to distinguish events of different polarities from each other when plotting. Defaults to True.

<a id="senpi.data_vis.visualization.plot_time_surface"></a>

#### plot\_time\_surface

```python
def plot_time_surface(time_surface: torch.Tensor, plot_type='3d')
```

Plot the time surface in either 2D or 3D.

This function visualizes the given time surface tensor. The visualization can be either a 2D heatmap or a 3D surface plot.

**Arguments**:

- `time_surface` (`torch.Tensor`): The time surface tensor to plot.
- `plot_type` (`str`): The type of plot to generate. Options are '2d' or '3d'.

**Raises**:

- `ValueError`: If an unsupported plot type is provided.

<a id="Data_Generation"></a>

# Data Generation (Algorithmic Reconstruction)

<a id="senpi.data_gen.reconstruction"></a>

## Reconstruction (senpi.data\_gen.reconstruction)

<a id="senpi.data_gen.reconstruction.ImgFramesFromEventsGenerator"></a>

### ImgFramesFromEventsGenerator Objects

```python
class ImgFramesFromEventsGenerator()
```

A class to generate grayscale image frames from only event data using the method described in the following paper:
Cedric Scheerlinck, Nick Barnes, Robert Mahony, "Continuous-time Intensity Estimation Using Event Cameras", Asian Conference on Computer Vision (ACCV), Perth, 2018.

**Paper link:** <br/>
https://arxiv.org/abs/1811.00386 <br/>

**Corresponding GitHub:**<br/>
https://github.com/cedric-scheerlinck/dvs_image_reconstruction <br/>

**Attributes**:

- `width` _int_ - Width of the event camera and generated image/video frame(s).
- `height` _int_ - Height of the event camera and generated image/video frame(s).
- `cutoff_frequency` _Union[int, float]_ - The cutoff frequency for the intensity estimation (the gain/alpha as described in the paper).
- `on_contrast_threshold` _float_ - Contrast threshold for ON events.
- `off_contrast_threshold` _float_ - Contrast threshold for OFF events.
- `most_recent_time_surface` _torch.Tensor_ - Tensor to store the most recent event time for each pixel.
- `cur_frame` _torch.Tensor_ - Tensor to store the current frame.
- `frame_list` _List[torch.Tensor]_ - List of generated frames.
- `timestamp_list` _List[float]_ - List of timestamps for the generated frames.
- `min_intensity` _float_ - Minimum intensity for the output frames.
- `max_intensity` _float_ - Maximum intensity for the output frames.

<a id="senpi.data_gen.reconstruction.ImgFramesFromEventsGenerator.__init__"></a>

#### \_\_init\_\_

```python
def __init__(width: int,
             height: int,
             on_contrast_threshold: float = 0.2,
             off_contrast_threshold: float = -0.2,
             cutoff_frequency: Union[int, float] = 5,
             min_intensity: float = -0.5,
             max_intensity: float = 1.5)
```

Initialize the `ImgFramesFromEventsGenerator` with given parameters.

**Arguments**:

- `width` (`int`): Width of the event camera and generated image/video frame(s).
- `height` (`int`): Height of the event camera and generated image/video frame(s).
- `on_contrast_threshold` (`float`): ON (positive) event contrast threshold (based on event camera specifications).
- `off_contrast_threshold` (`float`): OFF (negative) event contrast threshold (based on event camera specifications).
- `cutoff_frequency` (`Union[int, float]`): Cutoff frequency for the intensity estimation.
- `min_intensity` (`float`): Minimum intensity for normalization.
- `max_intensity` (`float`): Maximum intensity for normalization.

<a id="senpi.data_gen.reconstruction.ImgFramesFromEventsGenerator.reset"></a>

#### reset

```python
def reset()
```

Reset the internal state of the generator, which includes the time surface, current frame tensor, the frame list, and the timestamp list.

<a id="senpi.data_gen.reconstruction.ImgFramesFromEventsGenerator.generate_frames"></a>

#### generate\_frames

```python
def generate_frames(events: torch.Tensor,
                    gen_frames_mode: str = "events",
                    events_per_frame: int = 1500,
                    order: List[str] = None,
                    time_unit: Union[TimeUnit, str] = TimeUnit.MICROSECONDS,
                    enable_gpu: bool = False,
                    accumulate_frame_list: bool = False,
                    frames_tensor_list_device: torch.device = "cpu",
                    spatial_smoothing_method: str = "",
                    kernel_size: int = 5,
                    spatial_filter_sigma: float = -1.0,
                    save_images: bool = False,
                    generate_video: bool = False,
                    fps: int = 200,
                    output_parent_dir: Union[Path, str] = "",
                    video_file_name: str = None,
                    first_event_start_timestamp: bool = False)
```

Generate frames from event data.

**Arguments**:

- `events` (`torch.Tensor`): The input events.
- `gen_frames_mode` (`str, optional`): Mode of frame generation, either "events" or "fps", defaults to "events". Events mode will aggregate `events_per_frame` events per frame and FPS mode will aggregate events within time steps given by 1 / `fps` for each frame.
- `events_per_frame` (`int, optional`): Number of events per frame, defaults to 1500.
- `order` (`List[str], optional`): Order of columns in the events tensor, defaults to ['b', 't', 'x', 'y', 'p'].
- `time_unit` (`Union[TimeUnit, str], optional`): Unit of time for events, defaults to TimeUnit.MICROSECONDS.
- `enable_gpu` (`bool, optional`): Whether to use GPU (if available), defaults to False. If True, then `events`, `self.most_recent_time_surface`, and `self.cur_frame` will be moved to the GPU for the duration of the frame generation process and its associated operations. Note that, in some instances setting this parameter to True might actually cause the program to run slower if transferring the data between CPU and GPU imposes a significant enough overhead.
- `accumulate_frame_list` (`bool, optional`): Whether to accumulate generated frames in a list (list of torch.Tensor), defaults to False.
- `frames_tensor_list_device` (`torch.device, optional`): Device that each frame in `self.frame_list` should be stored on, defaults to "cpu".
- `spatial_smoothing_method` (`str, optional`): Method for spatial smoothing, defaults to "". Available smoothing methods are "gaussian" and "bilateral."
- `kernel_size` (`int, optional`): Kernel size for spatial smoothing, defaults to 5.
- `spatial_filter_sigma` (`float, optional`): Sigma for spatial smoothing, defaults to -1.0, which will prompt its calculation based on `kernel_size`.
- `save_images` (`bool, optional`): Whether to save generated images, defaults to False.
- `generate_video` (`bool, optional`): Whether to generate video from frames, defaults to False.
- `fps` (`int, optional`): Frames per second for the generated video, defaults to 200.
- `output_parent_dir` (`Union[Path, str], optional`): Directory to save the output, defaults to "".
- `video_file_name` (`str, optional`): Name of the output video file, defaults to None.
- `first_event_start_timestamp` (`bool, optional`): Whether the first event's timestamp should be the beginning of the video, defaults to False.

<a id="senpi.data_gen.reconstruction.ImgFramesFromEventsGenerator.convert_log_frame_to_intensity_img"></a>

#### convert\_log\_frame\_to\_intensity\_img

```python
def convert_log_frame_to_intensity_img(
        log_frame: torch.Tensor,
        spatial_smoothing_method: str = "",
        kernel_size: int = 5,
        spatial_filter_sigma: float = -1.0) -> torch.Tensor
```

Convert a log intensity frame to an intensity image.

This function applies exponential transformation to the log intensity frame,
scales it to the intensity range, and optionally applies spatial smoothing.

**Arguments**:

- `log_frame` (`torch.Tensor`): The log intensity frame.
- `spatial_smoothing_method` (`str`): Method for spatial smoothing, either "gaussian" or "bilateral".
- `kernel_size` (`int`): Size of the kernel for spatial smoothing.
- `spatial_filter_sigma` (`float`): Sigma value for spatial smoothing. If -1, it will be calculated based on the kernel size.

**Returns**:

`torch.Tensor`: The intensity image.

<a id="senpi.data_gen.reconstruction.ImgFramesFromEventsGenerator.create_video_from_frame_list"></a>

#### create\_video\_from\_frame\_list

```python
def create_video_from_frame_list(data_dir: Union[Path, str],
                                 fps: int,
                                 video_file_name: str = None)
```

Create a video from the accumulated frames in the frame list.

This function saves the frames as a video file in the specified directory with the given frame rate.

**Arguments**:

- `data_dir` (`Union[Path, str]`): Directory to save the video file.
- `fps` (`int`): Frames per second for the video.
- `video_file_name` (`str, optional`): Name of the generated video file. If None, defaults to "video.mp4".

[comment]: <> (DEEP LEARNING)

<!-- # Deep Learning Reconstruction

## Data Processing

### EventDataset Class

```python
class EventDataset(Dataset):
  def __init__(self, 
               event_files, 
               num_bins, 
               width, 
               height, 
               nrows, 
               skip_rows, 
               num_events):
```

The `EventDataset` Class initializes the directory to the event data, the number of bins, the width and height of the event camera, and many other parameters regarding the intake of event data. It utilizes `E2VID`'s `FixedSizeEventReader` class to read the events from the .txt file and packages the events into non-overlapping windows, each containing a fixed number of events. Finally, the class utilizes `E2VID`'s `events_to_voxel_grid_pytorch` function to build a pytorch which is 3D representation of the events in the time domain.

**Inputs**:
- `event_files` (`List[str]`): List of paths to event files.
- `num_bins` (`int`): Number of bins in the temporal axis of the voxel grid.
- `width` (`int`): Width of the voxel grid.
- `height` (`int`): Height of the voxel grid.
- `nrows` (`int`): Number of rows to read from each event file.
- `skip_rows` (`int`): Number of rows to skip at the beginning of each event file.
- `num_events` (`int`): Number of events per window.

    #### `__len__` function
    ```python
    def __len__(self):
    ```
    - Returns the number of event windows in the dataset

      ***Outputs***:
      - `int`: Number of event windows.

    #### `__getitem__` function
    ```python
    def __getitem__(self, idx=0):
    ```
    - Retuns the voxel grid for the event window at the specified index.

      ***Inputs***:
      - `idx` (`int`, optional): Index of the event window. Default is 0.
    
      ***Outputs***:
      - `torch.Tensor`: Voxel grid for the event window.

## Network 

**Paper:** https://rpg.ifi.uzh.ch/docs/TPAMI19_Rebecq.pdf

### Overview
The network used is adapted from the `E2VID` network, which is designed for event-based image reconstruction. It includes two main architectures: `Reconstruct` and `ReconstructRecurrent`, adaptations of `E2VID`'s work. Both architectures are based on a U-Net structure, with the recurrent version incorporation `ConvLSTM` or `ConvGRU` layers.

### `BaseModel` Class
```python
class BaseModel(nn.Module):
  def __init__(self, config):
```
The `BaseModel` class is the base class for all models. It initializes common configurations as well as logging options.

**Inputs**
- `config` (`dict`): Configuration dictionary for the model.


### `BaseUNet` Class
```python
class BaseUNet(nn.Module):
    def __init__(self, 
                 num_input_channels, 
                 num_output_channels=1, 
                 skip_type='sum', 
                 activation='sigmoid',
                 num_encoders=4, 
                 base_num_channels=32, 
                 num_residual_blocks=2, 
                 norm=None, 
                 use_upsample_conv=True):
```
The `BaseUNet` class provides the structure for the U-Net architecture. It initializes common configurations such as the number of input and output channels, skip connecion type, activation function, number of encoders, base number of channels, number of residual blocks, normalization type, and whether to use upsample convolution. It also includes methods to build residual blocks, decoders, and the prediction layer.

**Inputs**:
- `num_input_channels` (`int`): Number of input channels.
- `num_output_channels` (`int`, optional): Number of output channels. Default is 1.
- `skip_type` (`str`, optional): Type of skip connection ('sum' or 'concat'). Default is 'sum'.
- `activation` (`str`, optional): Activation function to use. Default is 'sigmoid'.
- `num_encoders` (`int`, optional): Number of encoder layers. Default is 4.
- `base_num_channels` (`int`, optional): Number of channels in the base layer. Default is 32.
- `num_residual_blocks` (`int`, optional): Number of residual blocks. Default is 2.
- `norm` (`str`, optional): Normalization type ('BN' for BatchNorm, 'IN' for InstanceNorm). Default is None.
- `use_upsample_conv` (`bool`, optional): Whether to use UpsampleConvLayer or TransposedConvLayer. Default is True.

  #### `build_resblocks` function 
  ```python
  def build_resblocks(self):
  ```
  - Creates a list of residual blocks and stores them in self.resblocks.

  #### `build_decoders` function
  ```python
  def build_decoders(self):
  ```
  - Creates a list of decoder layers and stores them in self.decoders.

  #### `build_prediction_layer` function
  ```python
  def build_prediction_layer(self):
  ```
  - Creates the final prediction layer that outputs the reconstructed image and stores it in self.pred.

### `UNet` Class
```python
class UNet(BaseUNet):
    def __init__(self, 
                 num_input_channels, 
                 num_output_channels=1, 
                 skip_type='sum', 
                 activation='sigmoid',
                 num_encoders=4, 
                 base_num_channels=32, 
                 num_residual_blocks=2, 
                 norm=None, 
                 use_upsample_conv=True):
```

The `Unet` class extends `BaseUnet` and defines the architecture of a standard U-Net, which is commonly used for image segmentation tasks. It consists of encoders, a series of residual blocks, and decoders.

**Inputs**:
- `num_input_channels` (`int`): Number of input channels.
- `num_output_channels` (`int`, optional): Number of output channels. Default is 1.
- `skip_type` (`str`, optional): Type of skip connection (`sum` or `concat`). Default is `sum`.
- `activation` (`str`, optional): Activation function to use. Default is `sigmoid`.
- `num_encoders` (`int`, optional): Number of encoder layers. Default is 4.
- `base_num_channels` (`int`, optional): Number of channels in the base layer. Default is 32.
- `num_residual_blocks` (`int`, optional): Number of residual blocks. Default is 2.
- `norm` (`str`, optional): Normalization type (`BN` for BatchNorm, `IN` for InstanceNorm). Default is None.
- `use_upsample_conv` (`bool`, optional): Whether to use `UpsampleConvLayer` or `TransposedConvLayer`. Default is True.

    #### `forward` function
    ```python
    def forward(self, x):
    ```
    - Forward pass of the `UNet` model.

      ***Inputs***
      - `x` (`torch.Tensor`): Input tensor of shape (N, num_input_channels, H, W).

      ***Outputs***
      - `torch.Tensor`: Output tensor of shape (N, num_output_channels, H, W).


### `UNetRecurrent` Class
```python
class UNetRecurrent(BaseUNet):
  def __init__(self, 
               num_input_channels, 
               num_output_channels=1, 
               skip_type='sum',
               recurrent_block_type='convlstm', 
               activation='sigmoid', 
               num_encoders=4, 
               base_num_channels=32, 
               num_residual_blocks=2, 
               norm=None, 
               use_upsample_conv=True):
```
The `UNetRecurrent` class extends `BaseUnet` and defines the architecture of a recurrent U-Net. Each encoder is followed by a recurrent convolutional block, such as `ConvLSTM` or `ConvGRU`.

**Inputs**
- `num_input_channels` (`int`): Number of input channels.
- `num_output_channels` (`int`, optional): Number of output channels. Default is 1.
- `skip_type` (`str`, optional): Type of skip connection (`sum` or `concat`). Default is `sum`.
- `recurrent_block_type` (`str`, optional): Type of recurrent block (`convlstm` or `convgru`). Default is `convlstm`.
- `activation` (`str`, optional): Activation function to use. Default is `sigmoid`.
- `num_encoders` (`int`, optional): Number of encoder layers. Default is 4.
- `base_num_channels` (`int`, optional): Number of channels in the base layer. Default is 32.
- `num_residual_blocks` (`int`, optional): Number of residual blocks. Default is 2.
- `norm` (`str`, optional): Normalization type (`BN` for BatchNorm, `IN` for InstanceNorm). Default is None.
- `use_upsample_conv` (`bool`, optional): Whether to use `UpsampleConvLayer` or `TransposedConvLayer`. Default is True.

    #### `forward` function
    ```python
    def forward(self, 
                x, 
                prev_states):
    ```
    - Forward pass of the `UNetRecurrent` model.

      ***Inputs***:
        - `x` (`torch.Tensor`): Input tensor of shape (N, num_input_channels, H, W).
        - `prev_states` (`list`, optional): Previous LSTM states for every encoder layer. Default is None.

      ***Outputs***:
        - `torch.Tensor`: Output tensor of shape (N, num_output_channels, H, W).
        - `list`: List of current states for every encoder layer.

### `BaseReconstruct` Class
```python
class BaseReconstruct(BaseModel):
  def __init__(self, 
               config):
```

The `BaseReconstruct` class is adapted from `E2VID`'s `BaseE2VID` class. It initializes common configurations for the network, such as the number of bins, skip type, number of encoders, base number of channels, number of residual blocks, normalization type, and whether to use upsample convolution.

**Inputs**
- `config` (`dict`): Configuration dictionary containing model parameters.
  - `num_bins` (`int`): Number of bins in the voxel grid event tensor.
  - `skip_type` (`str`, optional): Type of skip connection (`sum` or `concat`). Default is `sum`.
  - `num_encoders` (`int`, optional): Number of encoder layers. Default is 4.
  - `base_num_channels` (`int`, optional): Number of channels in the base layer. Default is 32.
  - `num_residual_blocks` (int, optional): Number of residual blocks. Default is 2.
  - `norm` (`str`, optional): Normalization type (`BN` for BatchNorm, `IN` for InstanceNorm). Default is None.
  - `use_upsample_conv` (`bool`, optional): Whether to use `UpsampleConvLayer` or `TransposedConvLayer`. Default is True.

### `Reconstruct` Class
```python
class Reconstruct(BaseReconstruct):
    def __init__(self, 
                 config):
```
The `Reconstruct` class extends `BaseReconstruct` and uses a standard U-Net architecture for image reconstruction.

**Inputs**:
- `config` (`dict`): Configuration dictionary containing model parameters.
  - `num_bins` (`int`): Number of bins in the voxel grid event tensor.
  - `skip_type` (`str`, optional): Type of skip connection (`sum` or `concat`). Default is `sum`.
  - `num_encoders` (`int`, optional): Number of encoder layers. Default is 4.
  - `base_num_channels` (`int`, optional): Number of channels in the base layer. Default is 32.
  - `num_residual_blocks` (int, optional): Number of residual blocks. Default is 2.
  - `norm` (`str`, optional): Normalization type (`BN` for BatchNorm, `IN` for InstanceNorm). Default is None.
  - `use_upsample_conv` (`bool`, optional): Whether to use `UpsampleConvLayer` or `TransposedConvLayer`. Default is `True`.

    #### `forward` function
    ```python
    def forward(self, 
                event_tensor, 
                prev_states=None):
    ```
    - Forward pass of the `Reconstruct` model

      ***Inputs***:
        - `event_tensor` (`torch.Tensor`): Input tensor of shape (N, num_bins, H, W).
        - `prev_states` (`list`, optional): Previous states for recurrent layers. Default is None.

      ***Outputs***:
        - `torch.Tensor`: A predicted image of size (N, 1, H, W), taking values in [0, 1].
        - `None`: Placeholder for compatibility with recurrent models.

### `ReconstructRecurrent` Class
```python
class ReconstructRecurrent(BaseReconstruct):
    def __init__(self, 
                 config):
```
The `ReconstructRecurrent` class extends `BaseReconstruct` and uses a recurrent U-Net architecture, where each encoder is followed by a `ConvLSTM` or `ConvGRU` layer.

**Inputs**:
- `config` (`dict`): Configuration dictionary containing model parameters.
  - `num_bins` (`int`): Number of bins in the voxel grid event tensor.
  - `skip_type` (`str`, optional): Type of skip connection (`sum` or `concat`). Default is `sum`.
  - `num_encoders` (`int`, optional): Number of encoder layers. Default is 4.
  - `base_num_channels` (`int`, optional): Number of channels in the base layer. Default is 32.
  - `num_residual_blocks` (int, optional): Number of residual blocks. Default is 2.
  - `norm` (`str`, optional): Normalization type (`BN` for BatchNorm, `IN` for InstanceNorm). Default is None.
  - `use_upsample_conv` (`bool`, optional): Whether to use `UpsampleConvLayer` or `TransposedConvLayer`. Default is `True`.
  - `recurrent_block_type` (`str`, optional): Type of recurrent block (`convlstm` or `convgru`). Default is `convlstm`.

#### `forward` function
```python
def forward(self, 
            event_tensor, 
            prev_states):
```
- Forward pass of the `ReconstructRecurrent` model.

  ***Inputs***:
    - `event_tensor` (`torch.Tensor`): Input tensor of shape $(N,$ `num_bins`$, H, W)$.
    - `prev_states` (`list`): Previous `ConvLSTM` or `ConvGRU` states for each encoder module.
  
  ***Outputs***:
  - `torch.Tensor`: A reconstructed image of size $(N, 1, H, W)$, taking values in $[0, 1]$.
  - `list`: Updated states for each encoder module.

### `submodules.py`
The `submodules.py` file contains contains various classes for the different layers of the models. These include `ConvLayer` (convolutional layer), variations of `ConvLayer` such as `RecurrentConvLayer`, `ResidualBlock` which includes two conv layers with residual connections, `ConvLSTM` and `ConvGRU`, and other layers.


## Training and Evaluation

### Overview
Training and Evaluation is conducted on the recurrent version of the network. The training is faciliated by converting the .txt event data into windows and then into 3D representations (voxel grids). The dataloaders then are iterated over in the form of event tensors, with which the model is able to learn and make predictions. The batch size indicates the amount of event windows processed at a time, and the number of bins indicates how many temporal bins each voxel grid will have. Additionally, `MSE` and `LPIPS` losses are combined for our loss function, and `ADAM` optimizer is used with a learning rate of 0.0001.  

### `set_random_seeds` function
```python
def set_random_seeds(seed=42):
```
Sets the random seeds for reproducibility in PyTorch.

### `detach_states` function
```python
def detach_states(states):
```
Detaches the state for the computational graph to prevent it from growing indefinitely. This can help mitigate issues with back propogation.

**Inputs**:
- `states` (`Union[None, torch.Tensor, Tuple, List]`): The states to be detached. This can be:
  - `None`: If there are no states.
  - `torch.Tensor`: A single state tensor.
  - `Tuple`: A tuple of state tensors.
  - `List`: A list of state tensors.

**Outputs**:
- `Union[None, torch.Tensor, Tuple, List]`: The detached states in the same structure as the input.

### `combined_loss` function
```python
def combined_loss(output, 
                  target, 
                  mse_loss, 
                  lpips_loss, 
                  device, 
                  alpha=0.3):
```
Computes a weighted sum of `MSE` (fidelity) and `LPIPS` (feature recovery) losses.

**Inputs**:
- `output` (`torch.Tensor`): The output tensor from the model.
- `target` (`torch.Tensor`): The ground truth tensor.
- `lpips_loss` (callable): The `LPIPS` loss function.
- `mse_loss` (callable): The `MSE` loss function.
- `alpha` (`float`, optional): The weighting factor for the `MSE` loss. The `LPIPS` loss is weighted by (1 - `alpha`).

**Outputs**:
- `torch.Tensor`: The combined loss value.

### `normalize_and_equalize` function
```python
def normalize_and_equalize(image):
```
Normalizes the ground truth frames to the range [0, 1] and applies local histogram equalization.

**Inputs**:
- `image` (`np.ndarray`): The input image, expected to be in the range [0, 255].

**Outputs**:
- `np.ndarray`: The processed image, normalized to [0, 1] and with local histogram equalization applied.

### `calculate_accuracy` function
```python
def calculate_accuracy(outputs, 
                       targets):
```
Calculates the accuracy of the model's predictions.

**Inputs**
- `outputs` (`torch.Tensor`): The output tensor from the model.
- `targets` (`torch.Tensor`): The ground truth tensor.

**Outputs**
- `float`: The accuracy of the predictions.

### `save_model` function
```python
def save_model(model, 
               optimizer, 
               MODEL_SAVE_PATH):
```
Saves a trained model and its optimizer state to the specified path.

**Inputs**:
- `model` (`torch.nn.Module`): The trained neural network model to be saved.
- `optimizer` (`torch.optim.Optimizer`): The optimizer used during training.
- `MODEL_PATH` (`str` or `Path`): The directory where the model will be saved.
- `MODEL_NAME` (`str`): The name of the file to save the model as.

### `load_model` function
```python
def load_model(model, 
               optimizer, 
               MODEL_SAVE_PATH, 
               device):
```
Loads a trained model and its optimizer state from the specified path.

**Inputs**:
- `model` (`torch.nn.Module`): The neural network model to be loaded.
- `optimizer` (`torch.optim.Optimizer`): The optimizer to be loaded.
- `MODEL_PATH` (`str` or `Path`): The directory where the model is saved.
- `MODEL_NAME` (`str`): The name of the file to load the model from.
- `device` (`torch.device`): The device to move the model to (`cpu` or `cuda`).

**Outputs**:
- `tuple`: A tuple containing the loaded model and optimizer.

### `compare_model_parameters` function
```python
def compare_model_parameters(model1, 
                             model2):
```
Compares the parameters of two models, or the trained model and the loaded model, and checks if they are identical

**Inputs**:
- `model1` (`torch.nn.Module`): The first model to compare.
- `model2` (`torch.nn.Module`): The second model to compare.

**Outputs**:
- `bool`: `True` if all parameters match, `False` otherwise.

### `plot_losses` function
```python
def plot_losses(train_loss, 
                test_loss, 
                num_epochs):
```
Plots the training and evaluation losses over epochs. Creates a linear subplot and a semilog plot.

**Inputs**:
- `model_results` (`dict`): A dictionary containing the training and validation losses. Expected keys are `train_loss` and `test_loss`, each mapping to a list of loss values.
- `num_epochs` (`int`): The number of epochs over which the model was trained.

### `training_step` function
```python
def training_step(model, 
                  dataloader, 
                  optimizer, 
                  event_preprocessor, 
                  options, 
                  mse_loss, 
                  lpips_loss, 
                  curr_epoch, 
                  device):
```
Performs a single training stop for the recurrent model. The function iterates over the dataloader, processes the event tensors, computes the loss using the `combined_loss` function, and updates the model paramaters using backpropagation.

**Inputs**:
- `model` (`torch.nn.Module`): The neural network model to be trained.
- `dataloader` (`torch.utils.data.DataLoader`): DataLoader providing the training data.
- `optimizer` (`torch.optim.Optimizer`): Optimizer for updating the model parameters.
- `device` (`torch.device`): Device on which to perform computations (`cpu` or `cuda`).
- `event_preprocessor` (`EventPreprocessor`): Preprocessor for event tensors.
- `options` (`argparse.Namespace`): Configuration options for training.
- `mse_loss` (callable): Mean Squared Error loss function.
- `lpips_loss` (callable): `LPIPS` loss function.
- `curr_epoch` (`int`): Current epoch number.

**Outputs**:
- `float`: The average loss over all batches.

### `eval_model` function
```python
def eval_model(model, 
               dataloader, 
               event_preprocessor, 
               options, 
               mse_loss, 
               lpips_loss, 
               curr_epoch, 
               device):
```
Evalutes the recurrent model on the provided dataset. It iterates over the dataloader, processes the event tensors, computes the loss using the `combined_loss` function, and returns a dictionary containing the evaluation results

**Inputs**:
- `model` (`torch.nn.Module`): The neural network model to be evaluated.
- `dataloader` (`torch.utils.data.DataLoader`): DataLoader providing the evaluation data.
- `device` (`torch.device`): Device on which to perform computations (`cpu` or `cuda`).
- `event_preprocessor` (`EventPreprocessor`): Preprocessor for event tensors.
- `mse_loss` (callable): Mean Squared Error loss function.
- `lpips_loss` (callable): `LPIPS` loss function.
- `curr_epoch` (`int`): Current epoch number.

**Outputs**
- `dict`: A dictionary containing the model name and the average loss over all batches.

### `train` function 
```python
def train(params: Dict):
```
Trains the model on the provided dataset and evaluates it on the test dataset. This function performs the training and evaluation loop for a specified number of epochs. It updates the model parametes using the training data and evaluates the model on the test data. This function also initializes the training and testing datasets and dataloaders. Additionally, the loss functions, optimizer, and the model are also defined here.

**Inputs**:
- `params` (`Dict`): Dictionary containing the following keys:
  - `device` (`torch.device`): Device on which to perform computations (`cpu` or `cuda`).
  - `batch_size` (`int`): Number of event windows processed at a time.
  - `num_epochs` (`int`): Number of epochs to train the model.
  - `learning_rate` (`float`): Learning rate for the optimizer.
  - `model_save_path` (`str`): Path to save the trained model.
  - `event_files` (`List[str]`): List of paths to event files.
  - `train_split` (`float`): Fraction of data to use for training.
  - `events_per_window` (`int`): Number of events per window.

**Outputs**:
- `dict`: A dictionary containing the training and test losses for each epoch.

    #### `calculate_num_rows` function
    - Calculate the number of rows in the event data file.

      ***Outputs***:
      - `int`: The total number of rows in the event data file.


## Reconstruction 

**Paper:** https://rpg.ifi.uzh.ch/docs/TPAMI19_Rebecq.pdf

### `ImageReconstructor` Class
```python
class ImageReconstructor:
  def __init__(self, 
               model, 
               height, 
               width, 
               num_bins, 
               options):

```
The `ImageReconstructor` class is used in `run_reconstruction.py`. This class applies different filters as well as rescalers from the `utils` folder in order to reconstruct the intensity frames.

**Inputs**:
- `model` (`torch.nn.Module`): The neural network model used for reconstruction.
- `height` (`int`): Height of the input images.
- `width` (`int`): Width of the input images.
- `num_bins` (`int`): Number of bins in the temporal axis of the voxel grid.
- `options`: Additional options for reconstruction.

    #### `initialize` function
    ```python
    def initialize(self, 
                   height, 
                   width, 
                   options):
    ```
    - Initialize the image reconstructor with the specified options.

      ***Inputs***:
      - `height` (`int`): Height of the input images.
      - `width` (`int`): Width of the input images.
      - `options`: Additional options for reconstruction.
    
    #### `update_reconstruction` function
    ```python
    def update_reconstruction(self, 
                              event_tensor, 
                              event_tensor_id, 
                              stamp=None):
    ```
    - Update the image reconstruction with a new event tensor.  

      ***Inputs***:
      - `event_tensor` (`torch.Tensor`): Tensor containing the event data.
      - `event_tensor_id` (`int`): Identifier for the event tensor.
      - `stamp` (`float`, optional): Timestamp for the event tensor. Default is None.

### `reconstruction` function
```python
def reconstruction(params: Dict):
```
This function initializes paramters  reconstruction using a `params` dictionary and `argparse`. The function uses these arguments in order to access the `path_to_events` to be reconstructed. A model is loaded, and an instance of `ImageReconstructor` is created. The passed in dataset is then processed as the events are converted into voxel grids, and the instance of `ImageReconstructor` is updated. This creates a side by side comparison of the stream of events vs the stream of intensity frames

**Inputs**:
- `params` (`Dict`): Dictionary containing the following keys:
  - `reconstruction_model_path` (`str`): Path to the trained model weights.
  - `data_file_for_reconstruction` (`str`): Path to the event data file.
  - `device` (`str`): Device to use for computations (`cpu` or `cuda`).
  - `learning_rate` (`float`): Learning rate for the optimizer.
  - `events_per_window` (`int`): Number of events per window for reconstruction.

    #### `calculate_num_rows` function
    - Calculate the number of rows in the event data file.

      ***Outputs***:
      - `int`: The total number of rows in the event data file.

### `saved_models`
This folder contains pre-trained models which can be utilized for reconstruction. The `dynamic_Recurrent` model is the best performing pre-trained model.

## Utilities 
The `utils` folder contains various helper functions which normalize events, convert txt to events, and more.

**Paper:** https://rpg.ifi.uzh.ch/docs/TPAMI19_Rebecq.pdf

### `config.py`
This file initalizes common configurations such as `num_bins` and `num_epochs`.

**Paper:** https://openaccess.thecvf.com/content_CVPR_2020/papers/I._Learning_to_Super_Resolve_Intensity_Images_From_Events_CVPR_2020_paper.pdf


### `event_readers.py`
This file contains the `FixedSizeEventReader` class which reads events from .txt or .zip and packages the events into non-overlapping event windows, each containing a a fixed number of events. The file also contains the `FixedDurationEventReader` class which also reads events, but the windows have fixed durations.

### `inference_utils.py`
This file contains the `EventPreprocessor` class which normalizes event tensors and flips them. This file also contains helper functions such as `events_to_voxel_grid_pytorch` which builds a 3D representation of the events in order to be fed into the network.

### `timers.py`
This file contains `CudaTimer` which times the GPU, and `Timer` which tracks the time taken to process datasets and training.

### `util.py`
This file contains various helper functions which help with normalization.

### `inference_options.py`
This file initializes both required and optional arguments for `run_reconstruction.py`.
 -->
