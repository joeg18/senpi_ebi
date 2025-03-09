import torch
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

def display_total_events(shape, events: torch.Tensor):
    """
    Display all events on matplotlib graph
    """
    frame = torch.zeros(shape)
    if events.dtype != torch.int:
        events = events.to(torch.int)
    for e in events:
        frame[int(e[2]), int(e[1])] += abs(e[3])
    frame = torch.where(frame==0, 1, frame)
    frame = torch.log(frame)
    plt.figure()
    plt.title("Log Total Number of Shot Noise Events At Different Photon Counts")
    plt.imshow(frame, cmap="RdPu")
    plt.xlabel('Number of Photons')
    plt.yticks([])
    plt.colorbar(location="bottom", label="Log Number of Events")
    plt.show()

def display_sum_events(shape, events):
    frame = torch.zeros(shape)
    for e in events:
        frame[int(e[2]), int(e[1])] += e[3]
    plt.figure()
    plt.title("Sum of Shot Noise Events")
    plt.imshow(frame, cmap="PiYG", norm=TwoSlopeNorm(0, frame.min(), frame.max()/2))
    plt.xlabel('Number of Photons')
    plt.yticks([])
    plt.colorbar(location="bottom", label="Sum of Event Polarities")
    plt.show()