import torch
import pandas as pd
import numpy as np
from typing import Dict, Union, List
import matplotlib.pyplot as plt
import seaborn as sns

def plot_events(events: Union[torch.Tensor, pd.DataFrame, np.ndarray], order: List[str] = None, separate_polarity: bool = True):
    """
    Plot event data in a 3D scatter plot.

    :param events: The event data to plot. Can be a torch.Tensor, pd.DataFrame, or np.ndarray.
    :type events: Union[torch.Tensor, pd.DataFrame, np.ndarray]
    :param order: The order of the parameters in the event data. Defaults to ['b', 't', 'x', 'y', 'p'] if None.
    :type order: List[str], optional
    :param separate_polarity: Whether to separate events by polarity. Defaults to True.
    :type separate_polarity: bool, optional
    """
    event_fig = plt.figure(figsize=(120, 120))

    if isinstance(events, torch.Tensor):
        if order is None:
            order = ['b', 't', 'x', 'y', 'p']
            print(f"Warning: No order provided. Assuming default order {order}.")
        param_ind = {order[ind]: ind for ind in range(len(order))}
        if separate_polarity:
            positive_events_mask = (events[:, param_ind['p']] == 1)
    elif separate_polarity:
        positive_events_mask = (events['p'] == 1)

    event_plot = event_fig.add_subplot(projection='3d')

    if separate_polarity:
        positive_events = events[positive_events_mask]
        negative_events = events[~positive_events_mask]
        if isinstance(events, torch.Tensor):
            p_t = positive_events[:, param_ind['t']].detach().cpu().numpy()
            p_x = positive_events[:, param_ind['x']].detach().cpu().numpy()
            p_y = positive_events[:, param_ind['y']].detach().cpu().numpy()

            n_t = negative_events[:, param_ind['t']].detach().cpu().numpy()
            n_x = negative_events[:, param_ind['x']].detach().cpu().numpy()
            n_y = negative_events[:, param_ind['y']].detach().cpu().numpy()
        else:
            p_t = positive_events['t']
            p_x = positive_events['x']
            p_y = positive_events['y']

            n_t = negative_events['t']
            n_x = negative_events['x']
            n_y = negative_events['y']
        
        event_plot.scatter(p_x, p_t, p_y, c='r')
        event_plot.scatter(n_x, n_t, n_y, c='b')
    else:
        if isinstance(events, torch.Tensor):
            t = events[:, param_ind['t']].detach().cpu().numpy()
            x = events[:, param_ind['x']].detach().cpu().numpy()
            y = events[:, param_ind['y']].detach().cpu().numpy()
        else:
            t = events['t']
            x = events['x']
            y = events['y']
        event_plot.scatter(x, t, y, c='g')

    event_plot.set_xlabel("Pixel Width", fontsize=100)
    event_plot.set_ylabel("Time", fontsize=100)
    event_plot.set_zlabel("Pixel Height", fontsize=100)

    event_plot.tick_params(axis='x', labelsize=40)
    event_plot.tick_params(axis='y', labelsize=40)
    event_plot.tick_params(axis='z', labelsize=40)
    
    plt.show()

def plot_time_surface(time_surface: torch.Tensor, plot_type='3d'):
    """
    Plot the time surface in either 2D or 3D.

    This function visualizes the given time surface tensor. The visualization can be either a 2D heatmap or a 3D surface plot.

    :param time_surface: The time surface tensor to plot.
    :type time_surface: torch.Tensor
    :param plot_type: The type of plot to generate. Options are '2d' or '3d'.
    :type plot_type: str
    
    :raises ValueError: If an unsupported plot type is provided.
    """
    if plot_type == '2d':
        fig, ax = plt.subplots(time_surface.shape[0], 1, figsize=(max(int(time_surface.shape[2] / 5), 10), max(int(time_surface.shape[1] / 5), 10)))

        for i in range(time_surface.shape[0]):
            sns.heatmap(time_surface[i].detach().cpu().numpy(), ax=ax[i] if time_surface.shape[0] != 1 else ax, annot=True, cmap='viridis')
            if time_surface.shape[0] == 1:
                ax.set_xlabel('Pixel Width')
                ax.set_ylabel('Pixel Height')
                break
            ax[i].set_xlabel('Pixel Width')
            ax[i].set_ylabel('Pixel Height')

        # plt.tight_layout(pad=2.0)

        plt.show()

    elif plot_type == '3d':
        fig = plt.figure()
        
        if time_surface.shape[1] < time_surface.shape[2]:
            time_surface_arr = torch.concat((time_surface, torch.zeros(time_surface.shape[0], time_surface.shape[2] - time_surface.shape[1], time_surface.shape[2])), dim=1).detach().cpu().numpy()
        elif time_surface.shape[2] < time_surface.shape[1]:
            time_surface_arr = torch.concat((time_surface, torch.zeros(time_surface.shape[0], time_surface.shape[1], time_surface.shape[1] - time_surface.shape[2])), dim=2).detach().cpu().numpy()
        else:
            time_surface_arr = time_surface.detach().cpu().numpy()
        
        plot_disp_num = 100 + (time_surface.shape[0] * 10) + 1

        x = torch.arange(time_surface_arr.shape[2])
        y = torch.arange(time_surface_arr.shape[1])
        x, y = torch.meshgrid(x, y)
        x = x.numpy()
        y = y.numpy()

        for i in range(time_surface.shape[0]):
            events_ts_vis = fig.add_subplot(plot_disp_num, projection='3d')
            events_ts_vis.plot_surface(x, y, time_surface_arr[i], cmap='viridis')
            events_ts_vis.set_xlabel('Pixel Height')
            events_ts_vis.set_ylabel('Pixel Width')
            # events_ts_vis.set_zlabel('Time Surface Value')
            plot_disp_num += 1
        
        # fig.tight_layout(pad=10.0)

        plt.tight_layout(pad=2.0)

        plt.show()
    else:
        raise ValueError(f"Unsupported plot type \"{plot_type}\"")