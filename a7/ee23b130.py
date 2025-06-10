import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike

def get_mic_positions(pitch: float, Nmics: int)->ArrayLike:
    mics = []
    offset = (Nmics/2-0.5)*pitch
    for i in range(0, Nmics):
        mics.append((0, (i*pitch - offset)))
    return np.array(mics)

def calculate_distance(pos1: Tuple, pos2: Tuple) -> float:
    """Function to find distance between two points"""
    return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** (0.5)

# Distance from src to a mic after reflecting through pt
def total_distance(src, pt, mic):
    d1 = calculate_distance(src, pt) # distance from src to pt
    d2 = calculate_distance(pt, mic) # distance from pt to mic
    return d1 + d2

# Main system parameters: number of mics, number of samples in time
Nmics = 64
Nsamp = 200
# Source: x,y coordinates: x: 0+, y: [-Y, +Y] where Y determined by pitch and Nmics
src = (0, 0)
# Spacing between microphones
pitch = 0.1
# proxy for sampling rate
dist_per_samp = 0.1
# Speed of sound in the medium
c = 2.0
# Time dilation factor for sinc pulse: how narrow
SincP = 5.0
# Locations of microphones
mics = get_mic_positions(pitch=pitch, Nmics=Nmics)

# Location of point obstacle
obstacle = (3, -1)

# Source sound wave - time axis and wave
# sinc wave with narrowness determined by parameter
t = 0 # CODE Nsamp time instants with spacing of dist_per_samp
def wsrc(t):
    return np.sinc(SincP*t)

time_stamps = []
for j in range(Nsamp):
        time_stamps.append(j*dist_per_samp/c)

outputs_list = []
for i in range(Nmics):
    delay_dist = total_distance(src=src, pt=obstacle, mic=mics[i])
    samples = []
    for j in range(Nsamp):
        samples.append(wsrc(time_stamps[j]-(delay_dist/c)))
    outputs_list.append(samples)
# outputs_list = np.loadtxt("rx3.txt")

# for i in range(len(outputs_list)):
#     plt.plot(time_stamps, [y+4*mics[i][1] for y in outputs_list[i]])
#     # Adding 4 times the y coordinate of the mic to space out the graphs of each mic
# plt.show()

grid = [[0] * Nsamp for _ in range(Nmics)]
for i in range(len(grid)):
    for j in range(len(grid[0])):
        point = (j*dist_per_samp, i*pitch-(Nmics/2-0.5)*pitch)
        for k in range(Nmics):
            delay_dist = total_distance(src=src, pt=point, mic=mics[k])
            delay_samples = max(int(delay_dist/dist_per_samp),0)
            delay_samples = min(delay_samples, Nsamp-1)
            grid[i][j]+=outputs_list[k][delay_samples]

plt.imshow(grid)
plt.show()

