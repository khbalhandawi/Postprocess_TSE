# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 05:20:27 2020

@author: Khalil
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rcParams
import random
from itertools import permutations 
import math
import random
from random import randrange
from simanneal import Annealer
import csv

# %% Import raw data and stip permutation indices = -1
from scipy.io import loadmat
from plot_tradespace import plot_tradespace

# one-liner to read a single variable
P_analysis = loadmat('DOE_permutations.mat')['P_analysis']
#P_analysis = P_analysis[0:44]

P_analysis_strip = []
for item in P_analysis:
    # Get permutation index
    permutation_index = []
    for arg in item:
        if not int(arg) == -1:
            permutation_index += [int(arg)] # populate permutation index
    P_analysis_strip += [permutation_index]

#attribute = ['n_f_th','Safety factor ($n_{safety}$)']
attribute = ['resiliance_th','Probability of satisfying requirement $\mathbb{P}(\mathbf{T} \in C)$']

[fig, ax, dictionary, start, wave, cross] = plot_tradespace(attribute)

# %% read MADS log file
opt_points = []
with open('mads_x_opt_pareto.log') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=' ')
    line_count = 0
    for row in csv_reader:
        row = [int(item) for item in row]
        row_strip = row[1::]
                
        opt_points += [row_strip]
        line_count += 1

# iterate through MADS bb evals
current_path = os.getcwd()
print('\nNumber of pareto points: %i' %(line_count))

x_data = []; y_data = []
for point in opt_points:
    ind = P_analysis_strip.index(point)
    x_data += [dictionary['weight'][ind]]
    y_data += [dictionary[attribute[0]][ind]]
    

pareto, = plt.plot(x_data, y_data, 'd', color = 'm', linewidth = 4.0, markersize = 10.0 )
   
ax.legend((start, cross, wave, pareto), ('initial design', 'cross concepts', 'wave concepts', 'Pareto point'))
fig.savefig(os.path.join(current_path,'tradespace_pareto.pdf'), format='pdf', dpi=1000,bbox_inches='tight')
