import torch
import numpy as np
import pandas as pd
from typing import List
from typing import Dict
from typing import Union
import traceback
import inspect
# from senpi.data_manip.computation import __auto_str
# ev Int to Vid
# rpg eToVid (u-nets) (recurrent fully conv neural network)
# eToVid
# 240x180 resolution (DAVIS 240C event camera)
# 1280x720px resolution (Prophesee event camera)

class Alg:
    """
    A base class for algorithms that transform event data.

    This class should be subclassed to implement specific transformation methods for event data.
    """

    def __call__(self, events: Union[pd.DataFrame, np.ndarray, torch.Tensor], **kwargs) -> Union[pd.DataFrame, np.ndarray, torch.Tensor]:
        """
        Apply the transformation algorithm to the event data.

        :param events: The event data to transform.
        :type events: Union[pd.DataFrame, np.ndarray, torch.Tensor]
        :param kwargs: Additional parameters for the transform method.
        :return: The transformed event data.
        :rtype: Union[pd.DataFrame, np.ndarray, torch.Tensor]
        """
        return self.transform(events=events, **kwargs)
    
    def __str__(self):
        """
        Return a string representation of the Alg instance.

        :return: A string representation of the instance's attributes.
        :rtype: str
        """
        attrs = vars(self)
        attrs_str = ', '.join(f"{key}={value}" for key, value in attrs.items())
        return f"{self.__class__.__name__}({attrs_str})"

    def transform(self, events: Union[pd.DataFrame, np.ndarray, torch.Tensor], **kwargs) -> Union[pd.DataFrame, np.ndarray, torch.Tensor]:
        """
        Transform the event data.

        This method should be implemented by subclasses to define the specific transformation behavior.

        :param events: The event data to transform.
        :type events: Union[pd.DataFrame, np.ndarray, torch.Tensor]
        :param kwargs: Additional parameters for the transform method.
        :return: The transformed event data.
        :rtype: Union[pd.DataFrame, np.ndarray, torch.Tensor]
        :raises NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError("Subclasses must implement this method")

class FlipXAlg(Alg):
    """
    A class for flipping events along the X-axis (across the Y-axis).

    This class provides methods to transform event data by flipping it along the X-axis. It supports operations on data in the form of Pandas DataFrames, NumPy arrays, and PyTorch tensors.

    :param max_width_index: The maximum width index (width minus one) used for the transformation.
    :type max_width_index: int
    """
    def __init__(self, max_width_index):
        """
        Initialize the FlipXAlg with the maximum width index.

        :param max_width_index: The maximum width index (width minus one) used for the transformation.
        :type max_width_index: int
        """
        self.max_width_index = max_width_index  # width minus one
    
    def __call__(self, events: Union[pd.DataFrame, np.ndarray, torch.Tensor], order: List[str]=None, modify_original: bool=True) -> Union[pd.DataFrame, np.ndarray, torch.Tensor]:
        return self.transform(events=events, order=order, modify_original=modify_original)

    def transform(self, events: Union[pd.DataFrame, np.ndarray, torch.Tensor], order: List[str]=None, modify_original: bool=True) -> Union[pd.DataFrame, np.ndarray, torch.Tensor]:
        """
        Transform the event data by flipping it along the X-axis.

        This method determines the type of the input data and calls the appropriate transformation method for Pandas DataFrame, NumPy array, or PyTorch Tensor.

        :param events: The event data to transform.
        :type events: Union[pd.DataFrame, np.ndarray, torch.Tensor]
        :param order: The order of columns in the event data (default is None, which uses the default order).
        :type order: List[str], optional
        :param modify_original: Whether to modify the original data or return a modified copy (default is True).
        :type modify_original: bool
        :return: The transformed event data.
        :rtype: Union[pd.DataFrame, np.ndarray, torch.Tensor]
        """
        if isinstance(events, torch.Tensor):
            return self.transform_tensor(events=events, order=order, modify_original=modify_original)
        else:
            return self.transform_pd_np(events=events, modify_original=modify_original)
    
    def transform_pd_np(self, events: Union[pd.DataFrame, np.ndarray], modify_original: bool=True) -> Union[pd.DataFrame, np.ndarray]:
        """
        Transform the event data by flipping it along the X-axis for Pandas DataFrame or NumPy array.

        :param events: The event data to transform.
        :type events: Union[pd.DataFrame, np.ndarray]
        :param modify_original: Whether to modify the original data or return a modified copy (default is True).
        :type modify_original: bool
        :return: The transformed event data.
        :rtype: Union[pd.DataFrame, np.ndarray]
        """
        if modify_original:
            events['x'] = self.max_width_index - events['x']
            return events
        else:
            events_ref = events.copy()
            events_ref['x'] = self.max_width_index - events_ref['x']
            return events_ref
    
    def transform_tensor(self, events: torch.Tensor, order: List[str]=None, modify_original: bool=True) -> torch.Tensor:
        """
        Transform the event data by flipping it along the X-axis for PyTorch Tensor.

        :param events: The event data to transform.
        :type events: torch.Tensor
        :param order: The order of columns in the event data (default is None, which uses the default order).
        :type order: List[str], optional
        :param modify_original: Whether to modify the original data or return a modified copy (default is True).
        :type modify_original: bool
        :return: The transformed event data.
        :rtype: torch.Tensor
        """
        if order is None:
            order = ['b', 't', 'x', 'y', 'p']
            print(f"Warning: No order provided. Assuming default order {order}.")
        
        param_ind = {order[ind] : ind for ind in range(len(order))}
        
        if modify_original:
            events[:, param_ind['x']] = self.max_width_index - events[:, param_ind['x']]
            return events
        else:
            events_ref = events.detach().clone()
            events_ref[:, param_ind['x']] = self.max_width_index - events_ref[:, param_ind['x']]
            return events_ref
    
    def set_max_width_index(self, max_width_index):
        """
        Set the maximum width index for the transformation.

        :param max_width_index: The maximum width index (width minus one) to set.
        :type max_width_index: int
        """
        self.max_width_index = max_width_index


class FlipYAlg(Alg):
    """
    A class for flipping events along the Y-axis (across the X-axis).

    This class provides methods to transform event data by flipping it along the Y-axis. It supports operations on data in the form of Pandas DataFrames, NumPy arrays, and PyTorch tensors.

    :param max_height_index: The maximum height index (height minus one) used for the transformation.
    :type max_height_index: int
    """
    def __init__(self, max_height_index):
        """
        Initialize the FlipYAlg with the maximum height index.

        :param max_height_index: The maximum height index (height minus one) used for the transformation.
        :type max_height_index: int
        """
        self.max_height_index = max_height_index  # height minus one
    
    def __call__(self, events: Union[pd.DataFrame, np.ndarray, torch.Tensor], order: List[str]=None, modify_original: bool=True) -> Union[pd.DataFrame, np.ndarray, torch.Tensor]:
        return self.transform(events=events, order=order, modify_original=modify_original)

    def transform(self, events: Union[pd.DataFrame, np.ndarray, torch.Tensor], order: List[str]=None, modify_original: bool=True) -> Union[pd.DataFrame, np.ndarray, torch.Tensor]:
        """
        Transform the event data by flipping it along the Y-axis.

        This method determines the type of the input data and calls the appropriate transformation method for Pandas DataFrame, NumPy array, or PyTorch Tensor.

        :param events: The event data to transform.
        :type events: Union[pd.DataFrame, np.ndarray, torch.Tensor]
        :param order: The order of columns in the event data (default is None, which uses the default order).
        :type order: List[str], optional
        :param modify_original: Whether to modify the original data or return a modified copy (default is True).
        :type modify_original: bool
        :return: The transformed event data.
        :rtype: Union[pd.DataFrame, np.ndarray, torch.Tensor]
        """
        if isinstance(events, torch.Tensor):
            return self.transform_tensor(events=events, order=order, modify_original=modify_original)
        else:
            return self.transform_pd_np(events=events, modify_original=modify_original)
    
    def transform_pd_np(self, events: Union[pd.DataFrame, np.ndarray], modify_original: bool=True) -> Union[pd.DataFrame, np.ndarray]:
        """
        Transform the event data by flipping it along the Y-axis for Pandas DataFrame or NumPy array.

        :param events: The event data to transform.
        :type events: Union[pd.DataFrame, np.ndarray]
        :param modify_original: Whether to modify the original data or return a modified copy (default is True).
        :type modify_original: bool
        :return: The transformed event data.
        :rtype: Union[pd.DataFrame, np.ndarray]
        """
        if modify_original:
            events['y'] = self.max_height_index - events['y']
            return events
        else:
            events_ref = events.copy()
            events_ref['y'] = self.max_height_index - events_ref['y']
            return events_ref
    
    def transform_tensor(self, events: torch.Tensor, order: List[str]=None, modify_original: bool=True) -> torch.Tensor:
        """
        Transform the event data by flipping it along the Y-axis for PyTorch Tensor.

        :param events: The event data to transform.
        :type events: torch.Tensor
        :param order: The order of columns in the event data (default is None, which uses the default order).
        :type order: List[str], optional
        :param modify_original: Whether to modify the original data or return a modified copy (default is True).
        :type modify_original: bool
        :return: The transformed event data.
        :rtype: torch.Tensor
        """
        if order is None:
            order = ['b', 't', 'x', 'y', 'p']
            print(f"Warning: No order provided. Assuming default order {order}.")
        
        param_ind = {order[ind] : ind for ind in range(len(order))}
        
        if modify_original:
            events[:, param_ind['y']] = self.max_height_index - events[:, param_ind['y']]
            return events
        else:
            events_ref = events.detach().clone()
            events_ref[:, param_ind['y']] = self.max_height_index - events_ref[:, param_ind['y']]
            return events_ref
    
    def set_max_height_index(self, max_height_index):
        """
        Set the maximum height index for the transformation.

        :param max_height_index: The maximum height index (height minus one) to set.
        :type max_height_index: int
        """
        self.max_height_index = max_height_index


class InvertPolarityAlg(Alg):
    """
    A class for inverting the polarity of events.

    This class provides methods to invert the polarity of event data. It supports operations on data in the form of Pandas DataFrames, NumPy arrays, and PyTorch tensors. It also supports binary polarity mode.

    """
    def __call__(self, events: Union[pd.DataFrame, np.ndarray, torch.Tensor], order: List[str]=None, binary_polarity_mode: bool=False, modify_original: bool=True) -> Union[pd.DataFrame, np.ndarray, torch.Tensor]:
        return self.transform(events=events, order=order, binary_polarity_mode=binary_polarity_mode, modify_original=modify_original)
    
    def transform(self, events: Union[pd.DataFrame, np.ndarray, torch.Tensor], order: List[str]=None, binary_polarity_mode: bool=False, modify_original: bool=True) -> Union[pd.DataFrame, np.ndarray, torch.Tensor]:
        """
        Transform the event data by inverting the polarity.

        This method determines the type of the input data and calls the appropriate transformation method for Pandas DataFrame, NumPy array, or PyTorch Tensor.

        :param events: The event data to transform.
        :type events: Union[pd.DataFrame, np.ndarray, torch.Tensor]
        :param order: The order of columns in the event data (default is None, which uses the default order).
        :type order: List[str], optional
        :param binary_polarity_mode: Whether the events are represented such that 0 denotes OFF events and 1 denotes ON events (default is False; assuming that OFF events are denoted by -1 and ON events are denoted by 1).
        :type binary_polarity_mode: bool
        :param modify_original: Whether to modify the original data or return a modified copy (default is True).
        :type modify_original: bool
        :return: The transformed event data.
        :rtype: Union[pd.DataFrame, np.ndarray, torch.Tensor]
        """
        if isinstance(events, torch.Tensor):
            return self.transform_tensor(events=events, order=order, binary_polarity_mode=binary_polarity_mode, modify_original=modify_original)
        else:
            return self.transform_pd_np(events=events, binary_polarity_mode=binary_polarity_mode, modify_original=modify_original)
    
    def transform_pd_np(self, events: Union[pd.DataFrame, np.ndarray], binary_polarity_mode: bool=False, modify_original: bool=True) -> Union[pd.DataFrame, np.ndarray]:
        """
        Transform the event data by inverting the polarity for Pandas DataFrame or NumPy array.

        :param events: The event data to transform.
        :type events: Union[pd.DataFrame, np.ndarray]
        :param binary_polarity_mode:  Whether the events are represented such that 0 denotes OFF events and 1 denotes ON events (default is False; assuming that OFF events are denoted by -1 and ON events are denoted by 1).
        :type binary_polarity_mode: bool
        :param modify_original: Whether to modify the original data or return a modified copy (default is True).
        :type modify_original: bool
        :return: The transformed event data.
        :rtype: Union[pd.DataFrame, np.ndarray]
        """
        if modify_original:
            if binary_polarity_mode:
                events['p'] = 1 - events['p']
            else:
                events['p'] = -events['p']
            return events
        else:
            events_ref = events.copy()
            if binary_polarity_mode:
                events_ref['p'] = 1 - events_ref['p']
            else:
                events_ref['p'] = -events_ref['p']
            return events_ref
    
    def transform_tensor(self, events: torch.Tensor, order: List[str]=None, binary_polarity_mode: bool=False, modify_original: bool=True) -> torch.Tensor:
        """
        Transform the event data by inverting the polarity for PyTorch Tensor.

        :param events: The event data to transform.
        :type events: torch.Tensor
        :param order: The order of columns in the event data (default is None, which uses the default order).
        :type order: List[str], optional
        :param binary_polarity_mode:  Whether the events are represented such that 0 denotes OFF events and 1 denotes ON events (default is False; assuming that OFF events are denoted by -1 and ON events are denoted by 1).
        :type binary_polarity_mode: bool
        :param modify_original: Whether to modify the original data or return a modified copy (default is True).
        :type modify_original: bool
        :return: The transformed event data.
        :rtype: torch.Tensor
        """
        if order is None:
            order = ['b', 't', 'x', 'y', 'p']
            print(f"Warning: No order provided. Assuming default order {order}.")
        
        param_ind = {order[ind] : ind for ind in range(len(order))}
        
        if modify_original:
            if binary_polarity_mode:
                events[:, param_ind['p']] = 1 - events[:, param_ind['p']]
            else:
                events[:, param_ind['p']] = -events[:, param_ind['p']]
            return events
        else:
            events_ref = events.detach().clone()
            if binary_polarity_mode:
                events_ref[:, param_ind['p']] = 1 - events_ref[:, param_ind['p']]
            else:
                events_ref[:, param_ind['p']] = -events_ref[:, param_ind['p']]
            return events_ref
