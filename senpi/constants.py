import torch
import numpy as np
import pandas as pd
from enum import Enum

class TimeUnit(Enum):
    """
    Enumeration for different time units used in event-based vision data.

    Attributes:
        SECONDS (str): Represents seconds.
        MILLISECONDS (str): Represents milliseconds.
        MICROSECONDS (str): Represents microseconds.
        NANOSECONDS (str): Represents nanoseconds.
    """
    SECONDS = 's'
    MILLISECONDS = 'ms'
    MICROSECONDS = 'us'
    NANOSECONDS = 'ns'

# DAVIS 240C: seconds, x, y, polarity
# Prophesee: x, y, polarity, microseconds

# Prophesee Constants

HEIGHT_PROPHESEE = 720
"""
The height (in pixels) of the Prophesee event camera's image sensor.

Used for setting up or understanding the resolution of Prophesee event camera data.
"""

WIDTH_PROPHESEE = 1280
"""
The width (in pixels) of the Prophesee event camera's image sensor.

Used for setting up or understanding the resolution of Prophesee event camera data.
"""

PROPHESEE_DATA_ORDER = ['x', 'y', 'p', 't']
"""
The order of data fields in Prophesee event camera data.

The data fields are:
- 'x': Horizontal pixel coordinate.
- 'y': Vertical pixel coordinate.
- 'p': Polarity of the event (1 or -1).
- 't': Timestamp of the event in microseconds.
"""

PROPHESEE_PD_NP_DTYPES = {
    't': np.int64,
    'x': np.short,
    'y': np.short,
    'p': np.short,
    'b': np.int64
}
"""
Data types for the Prophesee event camera data fields when using Pandas and NumPy.

Dictionary mapping:
- 't': Timestamp (microseconds), dtype `np.int64`
- 'x': Horizontal pixel coordinate, dtype `np.short`
- 'y': Vertical pixel coordinate, dtype `np.short`
- 'p': Polarity of the event, dtype `np.short`
- 'b': Batch index or additional field, dtype `np.int64`
"""

PROPHESEE_TORCH_DTYPES = {
    't': torch.int64,
    'x': torch.short,
    'y': torch.short,
    'p': torch.short,
    'b': torch.int64
}
"""
Data types for the Prophesee event camera data fields when using PyTorch.

Dictionary mapping:
- 't': Timestamp (microseconds), dtype `torch.int64`
- 'x': Horizontal pixel coordinate, dtype `torch.short`
- 'y': Vertical pixel coordinate, dtype `torch.short`
- 'p': Polarity of the event, dtype `torch.short`
- 'b': Batch index or additional field, dtype `torch.int64`
"""

# DAVIS Constants

WIDTH_DAVIS = 240
"""
The width (in pixels) of the DAVIS 240C event camera's image sensor.

Used for setting up or understanding the resolution of DAVIS 240C event camera data.
"""

HEIGHT_DAVIS = 180
"""
The height (in pixels) of the DAVIS 240C event camera's image sensor.

Used for setting up or understanding the resolution of DAVIS 240C event camera data.
"""

DAVIS_DATA_ORDER = ['t', 'x', 'y', 'p']
"""
The order of data fields in DAVIS 240C event camera data.

The data fields are:
- 't': Timestamp of the event in seconds.
- 'x': Horizontal pixel coordinate.
- 'y': Vertical pixel coordinate.
- 'p': Polarity of the event (1 or -1).
"""

DAVIS_PD_NP_DTYPES = {
    't': np.float64,
    'x': np.short,
    'y': np.short,
    'p': np.short,
    'b': np.int64
}
"""
Data types for the DAVIS 240C event camera data fields when using Pandas and NumPy.

Dictionary mapping:
- 't': Timestamp (seconds), dtype `np.float64`
- 'x': Horizontal pixel coordinate, dtype `np.short`
- 'y': Vertical pixel coordinate, dtype `np.short`
- 'p': Polarity of the event, dtype `np.short`
- 'b': Batch index or additional field, dtype `np.int64`
"""

DAVIS_TORCH_DTYPES = {
    't': torch.float64,
    'x': torch.short,
    'y': torch.short,
    'p': torch.short,
    'b': torch.int64
}
"""
Data types for the DAVIS 240C event camera data fields when using PyTorch.

Dictionary mapping:
- 't': Timestamp (seconds), dtype `torch.float64`
- 'x': Horizontal pixel coordinate, dtype `torch.short`
- 'y': Vertical pixel coordinate, dtype `torch.short`
- 'p': Polarity of the event, dtype `torch.short`
- 'b': Batch index or additional field, dtype `torch.int64`
"""