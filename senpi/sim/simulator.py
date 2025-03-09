# Physical simulator v2 - Joes edition

# import appropariate libraries

import torch
from torch import nn
import numpy as np
import math
import random
import warnings
from senpi.data_manip.conversions import *
from senpi.custom_classes.classes import *

class EventSimulator(nn.Module):

    def __init__(self, params: dict):
        """
        Initialize an intance of the simulator.
        Args:
            params (dict): dictionary of user defined parameters to run SENPI
            v1 (torch.Tensor) [optional]: reference voltage used to kickoff simulation, else reference voltage is set by parameters in params
        """
        # save params for current model
        self.params = params

        # save important params keys & values as attributes for easy access
        self.refractory = params["refractory"]
        self.dt = params["dt"]
        self.well_cap = params["well_cap"]
        self.qe = params["QE"]
        self.pos_th = params["pos_th"]
        self.neg_th = params["neg_th"]
        self.sigma_pos = params["sigma_pos"]
        self.sigma_neg = params["sigma_neg"]
        self.leak_rate = params["leak_rate"]
        self.sigma_leak = params["sigma_leak"]
        self.hot_prob = params["hot_prob"]
        self.hot_rate = params["hot_rate"]
        self.v1 = params["v1"]
        self.sigma_v1 = params["sigma_v1"]
        self.device = params["device"]
        self.return_frame = params["return_frame"]
        self.return_events = params["return_events"]
        self.shot_noise = params["shot_noise"]
        self.sensor_noise = params["sensor_noise"]

        # if random seed utilized, fix seed across all random libraries
        if params["seed"] != 0:
            torch.manual_seed(params["seed"])
            np.random.seed(params["seed"])
            random.seed(params["seed"])

        # initialize physical quantities
        self.gain = []
        self.noise_map = []
        self.hot_map = []
        self.pos_map = []
        self.neg_map = []
        self.refractory_map = []
        self.clock = 0

        # Trackers for physical quantities
        self.clock = 0
        self.gain = 1 / (math.log(self.well_cap))
    
    # Initialize the simulator
    def _reset(self, params: dict = []):
        """
        Reset all internal properties and trackers

        Args:
            params [optional]: feed in new parameter dictionary to envoke on-the-flyer user changes
        """

        # First reset everything!
        self.gain = []
        self.noise_map = []
        self.hot_map = []
        self.pos_map = []
        self.neg_map = []
        self.refractory_map = []
        self.clock = 0

        # Second make new params if input
        if len(params) != 0:
            # save params for current model
            self.params = params

            # save important params keys & values as attributes for easy access
            # save important params keys & values as attributes for easy access
            self.refractory = params["refractory"]
            self.dt = params["dt"]
            self.well_cap = params["well_cap"]
            self.qe = params["QE"]
            self.pos_th = params["pos_th"]
            self.neg_th = params["neg_th"]
            self.sigma_pos = params["sigma_pos"]
            self.sigma_neg = params["sigma_neg"]
            self.leak_rate = params["leak_rate"]
            self.sigma_leak = params["sigma_leak"]
            self.hot_ratio = params["hot_prob"]
            self.hot_rate = params["hot_rate"]
            self.v1 = params["v1"]
            self.sigma_v1 = params["sigma_v1"]
            self.device = params["device"]
            self.return_frame = params["return_frame"]
            self.return_events = params["return_events"]
            self.sensor_noise = params["shot_noise"]
            self.sensor_noise = params["sensor_noise"]

            if params["seed"] != 0:
                torch.manual_seed(params["seed"])
                np.random.seed(params["seed"])
                random.seed(params["seed"])



        # set gain such that simulation is normalized between 0 -> 1
        self.gain = 1 / (math.log(self.well_cap))


    def _generate_internal_properties(self, im_size):
        """
        Generate internal maps and properties to track pixel-wise variation in event stream reporting

        Args:
            im_size: size of input images for forward model
        """

        # Create threshold maps that include the per-pixel variation in the positive and negative threshold
        self.pos_map = torch.clamp(torch.normal(self.pos_th, self.sigma_pos, size=im_size, dtype=torch.float32, device=self.device), min=0, max=1)
        self.neg_map = torch.clamp(torch.normal(self.neg_th, self.sigma_neg, size=im_size, dtype=torch.float32, device=self.device), min=-1, max=0)
        
        # Create noise and hot pixel map
        self.noise_map = torch.clamp(torch.normal(self.leak_rate, self.sigma_leak, size=im_size, dtype=torch.float32, device=self.device), min=0, max=1)  # get nominal noise map
        self.hot_map =  torch.where(torch.rand(size=im_size) < self.hot_prob, 1, 0).to(self.device)  # randomly assign hot pixels and label them as "1" (hot) or "0" (normal)
        self.noise_map = self.hot_map * self.hot_rate + (1-self.hot_map) * self.noise_map  # perform noise map fusion

        # Create refractory map
        self.refractory_map = torch.ones(size=im_size, dtype=torch.float, device=self.device) * self.clock

    # Implemented WITHOUT custom functions - used to compare forward model to ensure accuracy
    def _stack_to_events(self, stack):
        """
        Generate internal maps and properties to track pixel-wise variation in event stream reporting

        Args:
            stack: stack of frames to process in the form of [t, x, y], as tensors and in units of photons

        Output:
            events: tensor of events containing the [t, x, y, p] information for all events generated by forward model
            frames: [optional] event based frames calculated by forward model for GT testing
        """
        # short form for size of the image    
        im_size = stack.size()[-2:]  # extract size of each image

        # step 0: preallocate any quantities:
        if self.return_events:
            events = torch.tensor([0, 0, 0, 0])[None, :].to(self.device)
        else:
            events = []

        if self.return_frame:
            eframes = torch.zeros(stack.size())  # oon't waste memory if unneeded
        else:
            eframes = torch.empty()  # else just return an empty array

        # step 1: convert stack to units of voltage
        ## Shot noise + dark noise
        if self.params["shot_noise"] & (self.params["dark_e"] > 0.0):
            stack = torch.clamp(self.gain * torch.log(torch.poisson(stack) + (self.params["dark_e"] * torch.randn(stack.size())).to(self.device)), min=0, max=1)
        
        ## Shot noise
        elif self.params["shot_noise"] & (self.params["dark_e"] <= 0.0):
            stack = torch.clamp(self.gain * torch.log(torch.poisson(stack)), min=0, max=1)
        
        ## Dark noise
        elif not self.params["shot_noise"] & (self.params["dark_e"] > 0.0):
            stack = torch.clamp(self.gain * torch.log(stack + (self.params["dark_e"] * torch.randn(stack.size())).to(self.device)), min=0, max=1)

        ## No noise & photodiode
        else:
            stack = torch.clamp(self.gain * torch.log(stack), min=0, max=1)

        # step 2: loop through stack and determine event frames
        for i in range(0,stack.shape[0]):

            # Determine differential voltage
            cur_v = torch.squeeze(stack[i, :])
            diff_v = cur_v - self.ref_v
            # diff_v = cur_v - torch.squeeze(stack[i-1, :])

            # determine if events occur and their polarity

            # if sensor noise, add random variable that selects indicies with 1/2 total prob for each polarity: Equivicol to XOR signal event with noise event maps
            if self.sensor_noise:
                
                # Determine true events - note from inspection noise does not appear in pixels triggered by true signal differential (i.e. won't invert polarity of true signal)
                pos_ind = (diff_v > self.pos_map) & (self.refractory_map <= self.clock)   # use logic based for easy combination of events at next step
                neg_ind = (diff_v < self.neg_map) & (self.refractory_map <= self.clock)
                
                # determine a modified noise instance where true events are filtered out
                tmp = torch.rand(size=im_size, device=self.device)  # instance of noise map
                tmp[pos_ind] = 0.5  # force noise at true events to be turned off
                tmp[neg_ind] = 0.5
                
                # synthesize event and noise instances
                pos_ind = torch.where(pos_ind | (tmp < self.noise_map / 2), 1, 0)
                neg_ind =  torch.where(neg_ind | (tmp > 1-self.noise_map / 2), 1, 0)
                
            # else just signals dependent
            else:
                pos_ind = torch.where((diff_v > self.pos_map) & (self.refractory_map <= self.clock), 1, 0)  # use logic based for easy combination of events at next step
                neg_ind = torch.where((diff_v < self.neg_map) & (self.refractory_map <= self.clock), 1, 0)

            # if requested, determine event frame
            if self.return_frame:
                eframes[i, :] = pos_ind - neg_ind  # now can use boolean masks to quickly update frame without headache

            # Get true 2D indicies to update appropriate quantities
            pos_ind = torch.where(pos_ind == 1)  # get true 2D indicies
            neg_ind = torch.where(neg_ind == 1)

            # if requested, return event stream
            if self.return_events:
                # update positive events in stream
                x = pos_ind[1][:, None]
                y = pos_ind[0][:, None]
                t = self.clock*torch.ones_like(y, device=self.device)
                p = 1*torch.ones_like(y, device=self.device)
                events = torch.cat((events, torch.hstack([t, x, y, p])))

                # update negative events in stream
                x = neg_ind[1][:, None]
                y = neg_ind[0][:, None]
                t = self.clock*torch.ones_like(y, device=self.device)
                p = -1*torch.ones_like(y, device=self.device)
                events = torch.cat((events, torch.hstack([t, x, y, p])))

            # Update key internal maps as appropriate
            self.ref_v[pos_ind] = cur_v[pos_ind]
            self.ref_v[neg_ind] = cur_v[neg_ind]
            self.refractory_map[pos_ind] = self.clock + self.refractory
            self.refractory_map[neg_ind] = self.clock + self.refractory

            # Increment clock
            self.clock = self.clock + self.dt

        # Step 3: Return events
        return events[1:], eframes
    

    # same implementation of _stack_to_events but uses custom differentiable classes
    def _stack_to_events_diff(self, stack):
        """
        Generate internal maps and properties to track pixel-wise variation in event stream reporting

        Args:
            stack: stack of frames to process in the form of [t, x, y], as tensors and in units of photons

        Output:
            events: tensor of events containing the [t, x, y, p] information for all events generated by forward model
            frames: [optional] event based frames calculated by forward model for GT testing
        """
        # short form for size of the image      
        im_size = stack.size()[-2:]  # extract size of each imag

        # step 0: preallocate any quantities:
        if self.return_events:
            events = torch.tensor([0, 0, 0, 0])[None, :].to(self.device)
        else:
            events = []

        if self.return_frame:
            eframes = torch.zeros(stack.size())  # oon't waste memory if unneeded
        else:
            eframes = torch.empty()  # else just return an empty array

        # step 1: convert stack to units of voltage
        ## Shot noise + dark noise
        if self.params["shot_noise"] & (self.params["dark_e"] > 0.0):
            stack = torch.clamp(self.gain * CustomLog.apply(PoissonReinforce.apply(stack) + (self.params["dark_e"] * torch.randn(stack.size())).to(self.device)), min=0, max=1)
        
        ## Shot noise
        elif self.params["shot_noise"] & (self.params["dark_e"] <= 0.0):
            stack = torch.clamp(self.gain * CustomLog.apply(PoissonReinforce.apply(stack)), min=0, max=1)
        
        ## Dark noise
        elif (self.params["shot_noise"] == 0) & (self.params["dark_e"] > 0.0):
            stack = torch.clamp(self.gain * CustomLog.apply(stack + (self.params["dark_e"] * torch.randn(stack.size())).to(self.device)), min=0, max=1)

        ## No noise & photodiode
        else:
            stack = torch.clamp(self.gain * CustomLog.apply(stack), min=0, max=1)

        # step 2: loop through stack and determine event frames
        for i in range(0,stack.shape[0]):

            # Determine differential voltage
            cur_v = torch.squeeze(stack[i, :])
            diff_v = cur_v - self.ref_v
            # diff_v = cur_v - torch.squeeze(stack[i-1, :])

            # determine if events occur and their polarity

            # if sensor noise, add random variable that selects indicies with 1/2 total prob for each polarity: Equivicol to XOR signal event with noise event maps
            if self.sensor_noise:
                
                # Determine true events - note from inspection noise does not appear in pixels triggered by true signal differential (i.e. won't invert polarity of true signal)
                pos_ind = CustomComparison.apply(diff_v, self.pos_map, ">", torch.tensor(0.0), torch.tensor(1.0)) * CustomComparison.apply(self.refractory_map, self.clock, "<=", torch.tensor(0.0), torch.tensor(1.0))
                neg_ind = CustomComparison.apply(diff_v, self.neg_map, "<", torch.tensor(0.0), torch.tensor(1.0)) * CustomComparison.apply(self.refractory_map, self.clock, "<=", torch.tensor(0.0), torch.tensor(1.0))
                
                # determine a modified noise instance where true events are filtered out
                tmp = torch.rand(size=im_size, device=self.device)  # instance of noise map

                # use detached versions of indicies as map to turn off noise - signal will dominate output so noise cannot exist there and will set random downstream pixels to zero erroneously
                tmp[(pos_ind.detach() > 0).to(torch.bool)] = 0.5
                tmp[(neg_ind.detach() > 0).to(torch.bool)] = 0.5
                
                # synthesize event and noise instances
                # To synthesize the noise maps, use a max operation which is a proxy logical "or" operation that returns gradient values when = 1
                pos_ind = torch.max(pos_ind, CustomComparison.apply(tmp, self.noise_map/2, "<", torch.tensor(0.0), torch.tensor(1.0)))
                neg_ind =  torch.max(neg_ind, CustomComparison.apply(tmp, 1-self.noise_map/2, ">", torch.tensor(0.0), torch.tensor(1.0)))

            # else just signals dependent
            else:
                # To determine correct indicies, we find a map of diff_v values greater than pos_map (vice versa for neg_ind) and filter it by a map of clock values beyond the refractory period
                pos_ind = CustomComparison.apply(diff_v, self.pos_map, ">", torch.tensor(0.0), torch.tensor(1.0)) * CustomComparison.apply(self.refractory_map, self.clock, "<=", torch.tensor(0.0), torch.tensor(1.0))
                neg_ind = CustomComparison.apply(diff_v, self.neg_map, "<", torch.tensor(0.0), torch.tensor(1.0)) * CustomComparison.apply(self.refractory_map, self.clock, "<=", torch.tensor(0.0), torch.tensor(1.0))
                

            # if requested, determine event frame
            if self.return_frame:
                eframes[i, :] = pos_ind - neg_ind  # now can use boolean masks to quickly update frame without headache

            # Get true 2D indicies to update appropriate quantities
            pos_ind = torch.where(pos_ind == 1)  # get true 2D indicies
            neg_ind = torch.where(neg_ind == 1)

            # if requested, return event stream
            if self.return_events:
                # update positive events in stream
                x = pos_ind[1][:, None]
                y = pos_ind[0][:, None]
                t = self.clock*torch.ones_like(y, device=self.device)
                p = 1*torch.ones_like(y, device=self.device)
                events = torch.cat((events, torch.hstack([t, x, y, p])))

                # update negative events in stream
                x = neg_ind[1][:, None]
                y = neg_ind[0][:, None]
                t = self.clock*torch.ones_like(y, device=self.device)
                p = -1*torch.ones_like(y, device=self.device)
                events = torch.cat((events, torch.hstack([t, x, y, p])))

            # Update key internal maps as appropriate
            self.ref_v[pos_ind] = cur_v[pos_ind]
            self.ref_v[neg_ind] = cur_v[neg_ind]
            self.refractory_map[pos_ind] = self.clock + self.refractory
            self.refractory_map[neg_ind] = self.clock + self.refractory

            # Increment clock
            self.clock = self.clock + self.dt

        # Step 3: Return events
        return events[1:], eframes


    def forward(self, x: torch.tensor, reset: bool= True, v1: torch.tensor = []):
        """
        Takes in a stack of image frames and returns the corresponding event signal

        Args:
            x (torch.Tensor): input of shape (frames, x, y)
        """
        
        # Check status of input stack
        if not torch.is_tensor(x): x = torch.from_numpy(x, dtype=torch.float)  # convert from numpy if not tensor input
        if len(x.shape) == 2: x = x[None, :]  # if fed in an image, expand dim to 3
        assert len(x.shape) == 3, "Input stack must be of shape [Frames, x, y]"
        x = x.to(self.device)  # ensure on correct device
        im_size = x.size()[-2:]

        # if events and frames set not to return, return event stream
        if not self.return_frame and not self.return_events:
            warnings.warn('Params Set to Return Neither Event Stream Nor Frames. Setting Return Stream to True')
            self.return_events = True

        # Check to reset simulator between calls or to retain last parameters
        if reset:
            self._reset()  # reset internal trackers
            self._generate_internal_properties(im_size) # generate internal per-pixel quantities in the event camera such as thresholds and hot pixels

            # check if initial voltage provided or if needed to be calculated
            if len(v1) != 0:
                self.ref_v = v1.to(self.device)
            else:
                self.ref_v = torch.clamp(torch.normal(self.v1, self.sigma_v1, size=im_size), min = 0, max = 1).to(self.device)
        
        # if selecting NOT to reset
        else:
            # check if initial voltage provided or if needed to be calculated
            if len(v1) != 0:
                self.ref_v = v1.to(self.device)
            else:
                self.ref_v = torch.clamp(torch.normal(self.v1, self.sigma_v1, size=im_size), min = 0, max = 1).to(self.device)
            
            # check if any essential properties are missing and if so regenerate
            if len(self.noise_map) == 0 or len(self.pos_map) == 0 or len(self.neg_map) == 0:
                self._generate_internal_properties(im_size)

        # Now that setup is complete, run forward model
        events, frames = self._stack_to_events_diff(x)
        return events, frames