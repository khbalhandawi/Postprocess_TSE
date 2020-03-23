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
from sample_requirements import NOMAD_call

class PlotOptimizationProgress():

    """Test annealer with a travelling salesman problem.
    """

    # pass extra data (the distance matrix) into the constructor
    def __init__(self, state, req_vec, req_thresh, MADS_output_dir, opt_bb_calls, fig_R, fig_W):
        self.req_vec = req_vec
        self.req_thresh = req_thresh
        self.MADS_output_dir = MADS_output_dir
        self.opt_bb_calls = opt_bb_calls
        self.fig_R = fig_R
        self.fig_W = fig_W
        self.n_fcalls = 0
        self.state = []
        self.line_R = []
        self.line_W = []

    def move(self):
        """Get the branch components"""
        self.state = self.opt_bb_calls[self.n_fcalls]
        self.energy()
        
    def energy(self):
        """Calculates the length of the route."""
        req_vec = self.req_vec
        req_thresh = self.req_thresh
        MADS_output_dir = self.MADS_output_dir

        eval_point = self.state
        call_type = 1
        [outs,weight] = NOMAD_call(call_type,req_vec,req_thresh,eval_point,MADS_output_dir)
        resiliance = [thresh - item  for thresh,item in zip(req_thresh,outs[1::])]

        self.n_fcalls += 1
        #print('Number of function calls: %i' %(self.n_fcalls))
        #=====================================================================#
        # Plot progress

        # Get design index    
        x_data = [0]; R_data = [0.0]; w_data = [0.0] # initial point
        print(self.state)
        for i in range(len(self.state[2::])):
            x_data += [i+1]
            R_data += [resiliance[i]]
            w_data += [weight[i]]

        e = sum(w_data)

        ax = self.fig_R.gca()
        if len(self.line_R) > 0:
            self.line_R[0].remove()
        
        self.line_R = ax.plot(x_data, R_data, 's-', color = 'm', linewidth = 3.0, markersize = 7.5 )
        current_path = os.getcwd()
        self.fig_R.savefig(os.path.join(current_path,'DOE_results','stagespace_res_%i.pdf' %(self.n_fcalls)), 
                    format='pdf', dpi=100)
        
        ax = self.fig_W.gca()
        if len(self.line_W) > 0:
            self.line_W[0].remove()
        
        self.line_W = ax.plot(x_data, w_data, 's-', color = 'm', linewidth = 3.0, markersize = 7.5 )
        current_path = os.getcwd()
        self.fig_W.savefig(os.path.join(current_path,'DOE_results','stagespace_weight_%i.pdf' %(self.n_fcalls)), 
                    format='pdf', dpi=100)
        

        plt.pause(0.0005)
        plt.show()
        #=====================================================================#
        return e

#==============================================================================
# MAIN CALL
def main():
    # %% Import raw data and stip permutation indices = -1
    import os
    from scipy.io import loadmat
    from plot_stage_space import plot_stagespace

    attribute = ['$P(n_{safety}(\mathbf{T}) \ge n_{th})$']

    current_path = os.getcwd()
    MADS_output_folder = 'MADS_output'
    MADS_output_dir = os.path.join(current_path,MADS_output_folder)

    # plot 1
    plot_id = 1
    req_thresh = [ 0.01, 0.1, 0.3, 0.3, 0.8, 0.9 ]
    ds_s = [ [5 , 1 , 2 , 1 , -1 , -1 , 0 ] ]
    req_vec = [36, 36, 36, 36, 36, 36]

    [fig1, fig2] = plot_stagespace(attribute,ds_s,req_vec,req_thresh,MADS_output_dir,plot_id)

    # %% Begin combinatorial optimization

    list_points_file = os.path.join(current_path,'test_points.log')

    # read MADS log file
    bb_evals = []
    with open(list_points_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            bb_evals += [row]
            line_count += 1

    # iterate through MADS bb evals
    current_path = os.getcwd()
    optproblem = PlotOptimizationProgress(bb_evals[0], req_vec, req_thresh, MADS_output_dir, bb_evals, fig1, fig2)
        
    for bb_call in bb_evals:
        optproblem.move()

    print('\nNumber of function calls: %i' %(optproblem.n_fcalls))

#==============================================================================
if __name__ == "__main__":
    main()