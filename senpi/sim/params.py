import torch

def make_params():
    
    params = dict()

    # CAMERA PROPERTIES
    params["refractory"] = 50 # Refractory period (us, 1e6 us/s) Right now, assuming input frames are at a temporal resolution that this doesn't matter
    params["well_cap"] = 15000 # Max well capacity of each pixel.
    params["QE"] = 1 # Quantum efficiency of all the pixels
    params["dt"] = 50 # Time in us between each frame. If this is less than refractory, refratory periods will possibly limit event generation
    params["pos_th"] = 0.1 # Mean of the positive threshold
    params["neg_th"] = -0.1 # Mean of the negative threshold
    params["sigma_pos"] = .0 # Standard deviation of the pos thresholds. Studies have shown threshold variation to be .03
    params["sigma_neg"] = .0 # Standard deviation of the neg thresholds. Studies have shown threshold variation to be .03
    params["leak_rate"] = 0.05  # probability that a leak event occurs at each time point (i.e. 1/background_activity)
    params["sigma_leak"] = 0.01  # standard deviaton in per-pixel leakage
    params["return_frame"] = 1  # returns the event frame alongside the event stream in the simulation. 1: on, 0: off (returns empty tensor)
    params["return_events"] = 0  # returns the event stream alongside the event frames in the simulation. 1: on, 0: off (returns empty tensor)

    # NOISE PARAMETERS
    params["shot_noise"] = 0  # 1: on, 0: off
    params["sensor_noise"] = 1  # 1: on, 0: off
    params["hot_prob"] = 0  # prob of a pixel being determined to be "hot"
    params["hot_rate"] = 0.9  # prob of erronous events from hot pixel in average number of time points. Uniform distribution
    params["dark_e"] = 2.0 # number of electrons contributed by dark current: conversion of current to photoelectrons is current * dt / electron_charge
    params["seed"] = 1  # fix random seed for error testing / validation

    # INITIALIZATION
    params["v1"] = 0.05  # initial reference voltage at t1
    params["sigma_v1"] = 0.01  # Standard deviation of normal distribution center at v1 for reference voltage


    # ACCELERATION
    params["device"] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
    return params