import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Union, List, Dict
import inspect
from senpi.data_manip.algs import *
from senpi.data_manip.filters import *

def has_parameter(func, param_name):
    """
    Check if the function has a parameter with the given name.

    :param func: The function to check.
    :type func: function
    :param param_name: The name of the parameter to look for.
    :type param_name: str
    :return: True if the parameter exists, False otherwise.
    :rtype: bool
    """
    signature = inspect.signature(func)
    return param_name in signature.parameters

class SequentialCompute:
    """
    A class to sequentially apply a list of computations (algorithms or filters) on event data.

    Attributes:
        computation_list (list): A list of computations to apply.
        param_dict (dict): A dictionary of parameters to pass to each computation.

    Methods:
        set_param_dict(param_dict): Sets the parameter dictionary.
        __call__(events): Applies the list of computations to the event data.
        compute(events): Applies the list of computations to the event data.
    """
    def __init__(self, *computation_list):
        """
        Initializes the SequentialCompute class with a list of computations.

        :param computation_list: A list of computations (algorithms or filters) to apply.
        :type computation_list: list
        """
        if (len(computation_list) == 1) and isinstance(computation_list[0], list):
            self.computation_list = computation_list[0]
        else:
            self.computation_list = list(computation_list)
        self.param_dict = {'order': None}
    
    def set_param_dict(self, param_dict):
        """
        Sets the parameter dictionary to be used for each computation.

        :param param_dict: A dictionary of parameters to pass to each computation.
        :type param_dict: dict
        """
        self.param_dict = param_dict

    def __call__(self, events: Union[torch.Tensor, np.ndarray, pd.DataFrame]) -> Union[torch.Tensor, np.ndarray, pd.DataFrame]:
        """
        Applies the list of computations to the event data by calling the compute method.

        :param events: The event data to process.
        :type events: Union[torch.Tensor, np.ndarray, pd.DataFrame]
        :return: The processed event data.
        :rtype: Union[torch.Tensor, np.ndarray, pd.DataFrame]
        """
        return self.compute(events)

    def compute(self, events: Union[torch.Tensor, np.ndarray, pd.DataFrame]) -> Union[torch.Tensor, np.ndarray, pd.DataFrame]:
        """
        Applies the list of computations to the event data.

        :param events: The event data to process.
        :type events: Union[torch.Tensor, np.ndarray, pd.DataFrame]
        :return: The processed event data.
        :rtype: Union[torch.Tensor, np.ndarray, pd.DataFrame]
        """
        for computation in tqdm(self.computation_list):
            print("-" * 25)
            print(f"Started {computation}")
            kwargs_dict = {}
            
            if isinstance(computation, Filter):
                for param in self.param_dict:
                    if has_parameter(computation.filter_events, param):
                        kwargs_dict[param] = self.param_dict[param]
                events = computation.filter_events(events, **kwargs_dict)
            elif isinstance(computation, Alg):
                for param in self.param_dict:
                    if has_parameter(computation.transform, param):
                        kwargs_dict[param] = self.param_dict[param]
                events = computation.transform(events, **kwargs_dict)
        print("-" * 25)
        return events