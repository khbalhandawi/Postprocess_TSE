# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 05:20:27 2020

@author: Khalil
"""

import os
import numpy as np
from sample_requirements import NOMAD_call, DOE_generator

#==============================================================================
# MAIN CALL
def main():
    # %% Import raw data and stip permutation indices = -1

    attribute = ['Reliability ($\mathbb{P}(\mathbf{p} \in C)$)']

    current_path = os.getcwd()
    MADS_output_folder = 'MADS_output'
    MADS_output_dir = os.path.join(current_path,MADS_output_folder)

    DOE_filename = 'req_DOE'
    DOE_folder = 'LHS_DOE'
    DOE_dir = os.path.join(current_path,DOE_folder)

    index = 1
    n_points = 100000
    lb = 1; ub = 50
    n_var = 6

    weight_file = 'varout_opt_log_R4.log'
    res_ip_file = 'resiliance_ip_R4.log'
    excess_ip_file = 'excess_ip_R4.log'
    res_th_file = 'resiliance_th_R4.log'
    excess_th_file = 'excess_th_R4.log'

    obj_type = 1 # optimize with respect to excess

    req_thresh = [ 0.01, 0.1, 0.3, 0.3, 0.8, 0.9 ]
    eval_point = [ 6, 1, 1, -1, -1, -1, -1, 0 ]
    # req_vec = [ 23, 43, 49, 15, 22, 8 ]

    req_vectors = DOE_generator(index,n_points,n_var,lb,ub,DOE_dir,DOE_filename,False)
    req_vec = req_vectors[594]
    req_vec = req_vectors[1]
    req_vec = req_vectors[1760]

    call_type = 0
    [eval_point] = NOMAD_call(call_type,obj_type,weight_file,res_ip_file,
                                excess_ip_file,res_th_file,excess_th_file,
                                req_vec,req_thresh,eval_point,MADS_output_dir)

    call_type = 1
    [outs,weights,excesses] = NOMAD_call(call_type,obj_type,weight_file,res_ip_file,
                                excess_ip_file,res_th_file,excess_th_file,
                                req_vec,req_thresh,eval_point,MADS_output_dir)

    print(req_vec)
    print(outs)
    print(excesses)

#==============================================================================
if __name__ == "__main__":
    main()