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

class Optimizationproblem(Annealer):

    """Test annealer with a travelling salesman problem.
    """

    # pass extra data (the distance matrix) into the constructor
    def __init__(self, state, P_analysis_strip, dictionary, fig):
        self.P_analysis_strip = P_analysis_strip
        self.dictionary = dictionary
        self.fig = fig
        self.n_fcalls = 0
        self.state = []
        self.line = []
        
        super(Optimizationproblem, self).__init__(state)  # important!

    def move(self):
        """Swaps two cities in the route."""
        # no efficiency gain, just proof of concept
        # demonstrates returning the delta energy (optional)
        initial_energy = self.energy()
        
        n_concepts = 2
                
#        print('\n========================')
#        print(''.join(('state before: ',str(self.state))))
        
        if len(self.state[1::]) < 3 and all([item not in self.state[1::] for item in [2,3]]): # if only two deposits exist
            self.state[0] = randrange(0,n_concepts) # randomly change concept
        
        if self.state[0] == 0: # wave concept
            num_hucs = 2; # two deposition steps
        elif self.state[0] == 1: # cross concept
            num_hucs = 4; # two deposition steps
        
        r = list(range(num_hucs))
        if len(self.state[1::]) == 1: # always add to state if theres only one deposit
            r.remove(self.state[1])
            huc = random.choice(r)
            self.state.append(huc)
        else:
            huc = random.choice(r)
            if huc in self.state[1::]: # if deposit already exists remove it
                deposit = self.state[1::]
                deposit.remove(huc)
                self.state[1::] = deposit
            else: # if deposit does not exist add it
                self.state.append(huc)
        
#        print(''.join(('state after: ',str(self.state))))
        
        return self.energy() - initial_energy

    def energy(self):
        """Calculates the length of the route."""
        ind = self.P_analysis_strip.index(self.state)
        x_data = self.dictionary['weight'][ind]
        y_data = self.dictionary['n_f_th'][ind]
        
        e = -y_data;
        self.n_fcalls += 1;
        #print('Number of function calls: %i' %(self.n_fcalls))
        #=====================================================================#
        # Plot progress
        # generate random color for branch
        r = random.random()
        g = random.random()
        b = random.random()
        rgb = [r,g,b]
                
        x_data = []
        y_data = []
        
        for it in range(len(self.state[1::])):
            ind = self.P_analysis_strip.index(self.state[0:it+2])
            x_data += [dictionary['weight'][ind]]
            y_data += [dictionary['n_f_th'][ind]]
        
        ax = self.fig.gca()
        if len(self.line) > 0:
            self.line[0].remove()
        
        self.line = ax.plot(x_data, y_data, '-', color = 'k', linewidth = 3.0 );
        
        plt.pause(0.0005)
        plt.show()
        #=====================================================================#
        return e

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
    
[fig, ax, dictionary, start, wave, cross] = plot_tradespace();

# %% Begin combinatorial optimization

# initial state, a randomly-ordered itinerary
#init_state = random.choice(P_analysis_strip)
init_state = P_analysis_strip[0]

optproblem = Optimizationproblem(init_state, P_analysis_strip, dictionary, fig)
#optproblem.set_schedule(optproblem.auto(minutes=0.1, steps=2000))
optproblem.Tmax = 100
optproblem.Tmin = 1.7
optproblem.steps = 120
optproblem.updates = 100

# since our state is just a list, slice is the fastest way to copy
optproblem.copy_strategy = "slice"
state, e = optproblem.anneal()
print('\nNumber of function calls: %i' %(optproblem.n_fcalls))

x_data = []; y_data = [];
for it in range(len(state[1::])):
    ind = P_analysis_strip.index(state[0:it+2])
    x_data += [dictionary['weight'][ind]]
    y_data += [dictionary['n_f_th'][ind]]
    
ax.plot(x_data, y_data, '-', color = 'r', linewidth = 4.0 );