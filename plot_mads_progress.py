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
import csv
from scipy.io import loadmat
from plot_tradespace import plot_tradespace

def system_command(command):
    import subprocess
    from subprocess import PIPE,STDOUT
    #CREATE_NO_WINDOW = 0x08000000 # Creat no console window flag

    p = subprocess.Popen(command,shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT,
                         ) # disable windows errors

    for line in iter(p.stdout.readline, b''):
        line = line.decode('utf-8')
        print(line.rstrip()) # print line by line
        # rstrip() to reomove \n separator

def NOMAD_call_SINGLEOBJ(call_type,req_index,reliability_threshold,weight_file,
                        res_th_file,excess_th_file):

    command = "categorical %i %i %f %s %s %s" %(call_type,req_index,reliability_threshold,weight_file,res_th_file,excess_th_file)
    
    print(command)
    system_command(command)

class PlotOptimizationProgress():

    """Display and move step by step to visualize optimization progress
    """

    # pass extra data (the distance matrix) into the constructor
    def __init__(self, state, P_analysis_strip, dictionary, opt_bb_calls, attribute, fig, ax):
        self.P_analysis_strip = P_analysis_strip
        self.dictionary = dictionary
        self.opt_bb_calls = opt_bb_calls
        self.fig = fig
        self.ax = ax
        self.attribute = attribute
        self.n_fcalls = 0
        self.state = []
        self.line = []
        
#        super(PlotOptimizationProgress, self).__init__(state)  # important!

    def move(self):
        """Get the branch components"""
        self.state = self.opt_bb_calls[self.n_fcalls]
        self.energy()
        
    def energy(self):
        """Calculates the length of the route."""
        ind = self.P_analysis_strip.index(self.state)
        x_data = self.dictionary['weight'][ind]
        y_data = self.dictionary[attribute[0]][ind]
        
        e = -y_data
        self.n_fcalls += 1
        #print('Number of function calls: %i' %(self.n_fcalls))
        #=====================================================================#
        # Plot progress
        # generate random color for branch
                
        # x_data = [0.0]; y_data = [2.378925]; # initial point
        # x_data = [0.0]; y_data = [0.0] # initial point
        x_data = []; y_data = [] # initial point

        for it in range(len(self.state[1::])):
            ind = self.P_analysis_strip.index(self.state[0:it+2])
            x_data += [dictionary['weight'][ind]]
            y_data += [dictionary[attribute[0]][ind]]
        
        if len(self.line) > 0:
            self.line[0].remove()
        
        if self.n_fcalls == 1:
                # to prevent autoscaling
                xlim = self.ax.get_xlim(); self.ax.set_xlim(xlim)
                ylim = self.ax.get_ylim(); self.ax.set_ylim(ylim)

        self.line = self.ax.plot(x_data, y_data, 's-', color = 'm', linewidth = 3.0, markersize = 7.5 )
        plt.draw()
        self.ax.set_position(mpl.transforms.Bbox(np.array([[0.125,0.10999999999999999],[0.9,0.88]])))

        current_path = os.getcwd()

        self.fig.savefig(os.path.join(current_path,'progress','tradespace_%i.png' %(self.n_fcalls)), 
                    format='png', dpi=100)
        
        plt.pause(0.0005)
        #=====================================================================#
        return e

# %% Import raw data and stip permutation indices = -1

# one-liner to read a single variable
P_analysis = loadmat('DOE_permutations.mat')['P_analysis']
#P_analysis = P_analysis[0:44]

# attribute = ['n_f_th','Safety factor ($n_{safety}$)']
attribute = ['resiliance_th_gau','Reliability $\mathbb{P}(\mathbf{p} \in C)$']
# attribute = ['capability_th_uni','Volume of capability set ($V_c$)']

[fig, ax, dictionary, start, wave, cross, tube] = plot_tradespace(attribute)

# Append initial base design to dictionary
# Creating a Dictionary  
new_dict = dict()
for key in dictionary.keys():
    if key == 'n_f_th':
        new_key = np.append( dictionary[key], np.array([2.378925]) )
    elif key in ['i1', 'i2', 'i3', 'i4', 'i5']:
        new_key = np.append( dictionary[key], np.array([-1.0]) )
    else:
        new_key = np.append( dictionary[key], np.array([0.0]) )
        
    new_dict[key] = new_key

dictionary = new_dict

# Get unsorted design points
P_analysis_strip = []
for c,i1,i2,i3,i4,i5 in zip(dictionary['concept'],dictionary['i1'],dictionary['i2'],dictionary['i3'],dictionary['i4'],dictionary['i5']):
    # Get permutation index
    permutation_index = []
    for arg in [c,i1,i2,i3,i4,i5]:
        if not int(arg) == -1:
            permutation_index += [int(arg)] # populate permutation index
    
    P_analysis_strip += [permutation_index]

# %% Begin combinatorial optimization

# read MADS log file
bb_evals = []
with open('./MADS_output/mads_bb_calls.log') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        
        row_strip = []
        for item in row:
            if int(item) != -1:
                row_strip += [int(item)]
                
        bb_evals += [row_strip]
        line_count += 1

# run MADS for single objective problem
call_type = 0; req_index = 0 # Nominal case
reliability_threshold = 0.99
weight_file = 'varout_opt_log.log'
res_th_file = 'resiliance_th.log'
excess_th_file = 'excess_th.log'
NOMAD_call_SINGLEOBJ(call_type,req_index,reliability_threshold,weight_file,res_th_file,excess_th_file)

# iterate through MADS bb evals
current_path = os.getcwd()
optproblem = PlotOptimizationProgress(bb_evals[0],P_analysis_strip, dictionary, bb_evals, attribute, fig, ax)

for bb_call in bb_evals:
    optproblem.move()

print('\nNumber of function calls: %i' %(optproblem.n_fcalls))

# read MADS log file
opt_points = []
with open('./MADS_output/mads_x_opt.log') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        row = [int(item) for item in row]
        row_strip = row[1::]
        row_strip = [i for i in row_strip if i != -1]

        opt_points += [row_strip]
        line_count += 1

x_data = []; y_data = []
for it in range(len(opt_points[0][1::])):
    ind = P_analysis_strip.index(opt_points[0][0:it+2])
    x_data += [dictionary['weight'][ind]]
    y_data += [dictionary[attribute[0]][ind]]

optproblem.line[0].remove()
ax.plot(x_data, y_data, '-', color = 'r', linewidth = 4.0 )
fig.savefig(os.path.join(current_path,'progress','tradespace_%i.png' %(optproblem.n_fcalls)), format='png', dpi=100)