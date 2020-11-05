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
    def __init__(self, state, req_vec, req_thresh, MADS_output_dir, opt_bb_calls, fig_R, fig_W,
                 obj_type, weight_file,res_ip_file,excess_ip_file,res_th_file,excess_th_file,
                 output_dir='Stagespace_output'):
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
        self.obj_type = obj_type
        self.weight_file = weight_file
        self.res_ip_file = res_ip_file
        self.excess_ip_file = excess_ip_file
        self.res_th_file = res_th_file
        self.excess_th_file = excess_th_file
        self.output_dir = output_dir

    def move(self):
        """Get the branch components"""
        self.state = self.opt_bb_calls[self.n_fcalls]
        self.energy()
        
    def energy(self):
        """Calculates the length of the route."""
        req_vec = self.req_vec
        req_thresh = self.req_thresh
        MADS_output_dir = self.MADS_output_dir

        obj_type = self.obj_type
        weight_file = self.weight_file
        res_ip_file = self.res_ip_file
        excess_ip_file = self.excess_ip_file
        res_th_file = self.res_th_file
        excess_th_file = self.excess_th_file

        eval_point = self.state
        call_type = 1
        [outs,weights,excesses] = NOMAD_call(call_type,obj_type,weight_file,res_ip_file,
                                   excess_ip_file,res_th_file,excess_th_file,
                                   req_vec,req_thresh,eval_point,MADS_output_dir)

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
            w_data += [excesses[i]]

        e = sum(w_data)

        ax = self.fig_R.gca()
        if len(self.line_R) > 0:
            self.line_R[0].remove()
        
        self.line_R = ax.plot(x_data, R_data, 's-', color = 'm', linewidth = 3.0, markersize = 7.5 )
        current_path = os.getcwd()
        self.fig_R.savefig(os.path.join(current_path,'DOE_results',self.output_dir,'stagespace_res_%i.pdf' %(self.n_fcalls)), 
                           bbox_inches='tight', format='pdf', dpi=300)
        
        ax = self.fig_W.gca()
        if len(self.line_W) > 0:
            self.line_W[0].remove()
        
        self.line_W = ax.plot(x_data, w_data, 's-', color = 'm', linewidth = 3.0, markersize = 7.5 )
        current_path = os.getcwd()
        self.fig_W.savefig(os.path.join(current_path,'DOE_results',self.output_dir,'stagespace_obj_%i.pdf' %(self.n_fcalls)), 
                           bbox_inches='tight', format='pdf', dpi=300)
        

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
    from plot_stage_space import plot_stagespace,check_folder

    attribute = ['Reliability ($\mathbb{P}(\mathbf{p} \in C)$)']

    current_path = os.getcwd()
    MADS_output_folder = 'MADS_output'
    MADS_output_dir = os.path.join(current_path,MADS_output_folder)

    Stagespace_output_folder = 'Stagespace_output'
    Stagespace_output_dir = os.path.join(current_path,'DOE_results',Stagespace_output_folder)
    check_folder(Stagespace_output_dir)

    weight_file = 'varout_opt_log_R4.log'
    res_ip_file = 'resiliance_ip_R4.log'
    excess_ip_file = 'excess_ip_R4.log'
    res_th_file = 'resiliance_th_R4.log'
    excess_th_file = 'excess_th_R4.log'

    obj_type = 1 # optimize with respect to excess

    # plot 1
    plot_id = 1
    req_thresh = [ 0.01, 0.1, 0.3, 0.3, 0.8, 0.9 ]
    ds_s = [ [5 , 1 , 2 , 1 , -1 , -1 , 0 ],
             [6 , 1 , 2 , 1 , 0 , 4 , -1 , 3 ] ]
    req_vec = [36, 50,  1, 46, 13, 31]

    # generate random color for branch
    colors = [[1,0,0],
              [0,0,1]]

    trans = [0.9, 0.4]

    [fig1, fig2] = plot_stagespace(attribute,ds_s,req_vec,req_thresh,MADS_output_dir,plot_id,
                                   weight_file, res_ip_file, excess_ip_file, res_th_file, 
                                   excess_th_file,colors=colors,trans=trans,output_dir=Stagespace_output_folder)

    # %% Begin combinatorial optimization

    list_points_file = os.path.join(current_path,'test_points_design_3.log')

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
    optproblem = PlotOptimizationProgress(bb_evals[0], req_vec, req_thresh, MADS_output_dir, bb_evals, fig1, fig2,
                                          obj_type,weight_file,res_ip_file,excess_ip_file,res_th_file,excess_th_file)
        
    for bb_call in bb_evals:
        optproblem.move()

    print('\nNumber of function calls: %i' %(optproblem.n_fcalls))

#==============================================================================
if __name__ == "__main__":
    main()