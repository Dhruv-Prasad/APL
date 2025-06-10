# Dhruv Prasad, EE23B130
# Assignment 5, Keyboard Optimization
# To run: python3 ee23b130.py

# Imports
from collections import defaultdict
import numpy as np
import random

import matplotlib.pyplot as plt

from copy import deepcopy
from typing import Tuple, Dict, Any

# Importing functions from the previous assignment to help plot keyboards
import my_kbd_plotting as mkp

# The initial keyboard layout python file
# I have modified the file given in previous assignment to have an enter key so that I can now handle new lines
import kbd_layout as layout
original_kbd = deepcopy(layout.keys)
key_presses_required_data = deepcopy(layout.characters)

# Modify this text to test out for different input
sample_text = '''Heat maps originated in 2D displays of the values in a data matrix. Larger values were 
represented by small dark gray or black squares (pixels) and smaller values by lighter squares. Sneath (1957) 
displayed the results of a cluster analysis by permuting the rows and the columns of a matrix to place similar 
values near each other according to the clustering. Jacques Bertin used a similar representation to display data 
that conformed to a Guttman scale. The idea for joining cluster trees to the rows and columns of the data matrix 
originated with Robert Ling in 1973. Ling used overstruck printer characters to represent different shades of gray, 
one character-width per pixel.'''


# Function defs
def distance_between(pos1: Tuple, pos2: Tuple) -> float:
    """Function to find distance between two points"""
    return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** (0.5)

def calculate_distance(string: str, kbd: Dict[str, dict[str, Any]], presses_reqd: Dict[str, Any]) -> float:
    """Function to find finger travel given a string and keyboard layout"""
    distance = 0
    for character in string:
        if character in presses_reqd:
            for key_pressed in presses_reqd[character]:
                d = distance_between(
                    kbd[key_pressed]["pos"],
                    kbd[kbd[key_pressed]["start"]]["pos"],
                )
                distance += d
    return distance

def swap_keys(key_1: str, key_2: str, kbd: Dict[str, dict[str, Any]]):
    '''Function to swap 2 keys in a layout given their names'''
    if(kbd[key_1]['start']!=key_1 and kbd[key_2]['start']!=key_2):
        kbd[key_1], kbd[key_2] = kbd[key_2], kbd[key_1]
    # If home row key is swapped, we need to check the full dict and update accordingly
    elif(kbd[key_1]['start']==key_1 and kbd[key_2]['start']!=key_2):
        for key in kbd:
            if (kbd[key]['start']==key_1):
                kbd[key]['start']=key_2
        kbd[key_1], kbd[key_2] = kbd[key_2], kbd[key_1]
    elif(kbd[key_1]['start']!=key_1 and kbd[key_2]['start']==key_2):
        for key in kbd:
            if (kbd[key]['start']==key_2):
                kbd[key]['start']=key_1
        kbd[key_1], kbd[key_2] = kbd[key_2], kbd[key_1]
    elif(kbd[key_1]['start']==key_1 and kbd[key_2]['start']==key_2):
        for key in kbd:
            if (kbd[key]['start']==key_1):
                kbd[key]['start']=key_2
            elif (kbd[key]['start']==key_2):
                kbd[key]['start']=key_1
        kbd[key_1], kbd[key_2] = kbd[key_2], kbd[key_1]

def generate_neighbour_layout(kbd: Dict[str, dict[str, Any]]) -> Dict[str, dict[str, Any]]:
    '''Function to generate a neighbour layout by randomly swapping two keys'''
    new_kbd = deepcopy(kbd)
    i, j = random.sample(range(len(new_kbd)), 2)
    key_i = list(new_kbd.keys())[i]
    key_j = list(new_kbd.keys())[j]
    swap_keys(key_1=key_i, key_2=key_j, kbd=new_kbd)
    return new_kbd

# Constant parameters for the simulated annealing
initial_temp = 1000
cooling_rate = 0.998
num_iterations = 500

# Simulated annealing
current_kbd = deepcopy(original_kbd)
current_distance = calculate_distance(sample_text, current_kbd, key_presses_required_data)
original_distance = current_distance

best_kbd = deepcopy(current_kbd)
best_distance = current_distance

current_temp = initial_temp
distances = [current_distance]
best_distances = [best_distance]

for _ in range(num_iterations):
    neighbour_kbd = generate_neighbour_layout(current_kbd)
    neighbour_distance = calculate_distance(sample_text, neighbour_kbd, key_presses_required_data)
    
    # I wanted a probabilty linear with my temperature
    p = current_temp*np.exp(current_distance - neighbour_distance)

    # Select worse layouts with probability p
    if neighbour_distance < current_distance or random.random() < p:
        current_kbd = neighbour_kbd
        current_distance = neighbour_distance
        
        if current_distance < best_distance:
            best_kbd = deepcopy(current_kbd)
            best_distance = current_distance
    current_temp *= cooling_rate
    distances.append(current_distance)
    best_distances.append(best_distance)

# Set up the figure and subplots
fig, axes_dict = plt.subplot_mosaic([['upper left', 'right'],
                               ['lower left', 'right']],
                              figsize=(5.5, 3.5), layout="constrained")
ax1 = axes_dict['upper left']
ax2 = axes_dict['lower left']
ax3 = axes_dict['right']

fig.suptitle("Simulated Annealing for Keyboard finger travel Optimization")

# Plot distances over iterations
xs = [i for i in range(0,num_iterations+1)]
ax3.set_xlim(0, num_iterations)
ax3.set_ylim(min(distances) * 0.95, max(distances) * 1.05)
ax3.set_title("Best Distance over Iterations")
ax3.set_xlabel("Iteration")
ax3.set_ylabel("Distance")
ax3.plot(xs, distances, label = "All distances")
ax3.plot(xs, best_distances, label = "Best distances")


# Plot initial and final keyboard
# Get the widths
widths_original = defaultdict(float)
plot_limits_original = mkp.get_widths(original_kbd, widths_original)
widths_best = defaultdict(float)
plot_limits_best = mkp.get_widths(best_kbd, widths_best)

# Plot both keyboards
mkp.plot_kbd(ax=ax1, kbd=original_kbd, widths=widths_original, plot_lims=plot_limits_original)
mkp.plot_kbd(ax=ax2, kbd=best_kbd, widths=widths_best, plot_lims=plot_limits_best)

# Draw heatmaps on both keyboards
key_press_frequency = defaultdict(int)
max_freq = mkp.get_freqs(string=sample_text, presses_reqd=key_presses_required_data, dictionary=key_press_frequency)
mkp.draw_map(ax=ax1, kbd=original_kbd, widths=widths_original, plot_lims=plot_limits_original, freqs=key_press_frequency, max_freq=max_freq)
mkp.draw_map(ax=ax2, kbd=best_kbd, widths=widths_best, plot_lims=plot_limits_best, freqs=key_press_frequency, max_freq=max_freq)

# Label both keyboards
ax1.set_title("Original keyboard: "+str(original_distance))
ax2.set_title("Best keyboard: "+str(best_distance))

# Save the figure before showing it
plt.savefig("ee23b130_a5.png", format="png", dpi=300, bbox_inches="tight")

# Show the figure
plt.show()