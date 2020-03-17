# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 01:12:10 2020

@author: Khalil
"""
import csv

#==============================================================================#
# %% Execute system commands and return output to console
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

def NOMAD_call(call_type,req_vec,req_thresh,eval_point,MADS_output_dir):
    
    import os
    
    req_vec_str = ' '.join(map(str,req_vec)) # print variables as space demilited string
    req_thresh_str = ' '.join(map(str,req_thresh)) # print parameters as space demilited string
    eval_point_str = ' '.join(map(str,eval_point)) # print parameters as space demilited string
    
    command = "categorical_MSSP %i %s %s %s" %(call_type,req_vec_str,req_thresh_str,eval_point_str)
    print(command)
    system_command(command)
    
    if call_type == 0: # read optimization result
    
        # read MADS log file
        opt_file = os.path.join(MADS_output_dir,'mads_x_opt.log')
        
        with open(opt_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                row = [int(item) for item in row]
            
            opt_points = [row]
                
        return opt_points
    
    elif call_type == 1: # read eval result
        
        # read eval output log file
        eval_file = os.path.join(MADS_output_dir,'eval_point_out.log')
        
        with open(eval_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                row = [float(item) for item in row]
            
            outs = row
        
        # read weight log file
        weight_file = os.path.join(MADS_output_dir,'weight_design.log')
        
        with open(weight_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                row = [float(item) for item in row]
            
            weights = row
                
        return outs,weights
    
def lhs_function(n_points,n_var,lb,ub,DOE_dir):
    from scipy import io
    import os
    # Outputs a latin hypercube that is augmentables via R lhs package
    
    command = 'RScript --vanilla lhs_int.R %i %i %i %i' %(n_points, n_var, lb, ub)
    print(command)
    system_command(command)
    
    
    output_dir = os.path.join(DOE_dir,'LHS_samples.mat')
    
    
    mat = io.loadmat(output_dir) # get optitrack data
    data = mat['A']
    
    return data

#==============================================================================#
# %% DOE FOR LOADCASE LOADS
def DOE_generator(index,n_points,n_var,lb,ub,DOE_dir,DOE_filename,regenerate):
    import os
    import numpy as np
    
        
    DOE_full_name = DOE_filename+'.npy'
    DOE_filepath = os.path.join(DOE_dir,DOE_full_name)
    
    if regenerate:
        points = lhs_function(n_points,n_var,lb,ub,DOE_dir)
        np.save(DOE_filepath, points) # save DOE array
        
        i_prev = []
        for f in os.listdir(DOE_dir):
            if f.find('%i_%s' %(index,DOE_filename)) == 0: # make sure string is at beginning
                i_prev += [int(f.split('_')[-1][:-4])]
    
        if i_prev:
            i_prev = max(i_prev)
        else:
            i_prev = 0
        
        DOE_copy_filepath = os.path.join(DOE_dir,'%i_%s_%i.npy' %(index,DOE_filename,i_prev+1))
        np.save(DOE_copy_filepath, points) # save DOE array
        
    else:
        points = np.load(DOE_filepath) # save DOE array
    
    return points

def scaling(x,l,u,operation):
    # scaling() scales or unscales the vector x according to the bounds
    # specified by u and l. The flag type indicates whether to scale (1) or
    # unscale (2) x. Vectors must all have the same dimension.
    
    if operation == 1:
        # scale
        x_out=(x-l)/(u-l)
    elif operation == 2:
        # unscale
        x_out = l + x*(u-l)
    
    return x_out

#==============================================================================#
# %% MAIN DOE LOOP
def main():
    
    import numpy as np
    import os
    
    index = 1
    n_points = 100000
    lb = 1; ub = 50
    n_var = 6
    
    current_path = os.getcwd()
    DOE_filename = 'req_DOE'
    MADS_output_folder = 'MADS_output'
    DOE_folder = 'LHS_DOE'
    DOE_out_folder = 'DOE_results'
    
    MADS_output_dir = os.path.join(current_path,MADS_output_folder)
    DOE_dir = os.path.join(current_path,DOE_folder)
    DOE_out_dir = os.path.join(current_path,DOE_out_folder)
    
    points = DOE_generator(index,n_points,n_var,lb,ub,DOE_dir,DOE_filename,False)
    print(points)
    
    # req_vec = [ 2 , 1 , 4 , 5 , 6 , 8 ]
    
    #[('uniform', array([0.375, 0.8  , 0.8  , 0.625]), array([0.00390625, 0.00173611, 0.00173611, 0.00390625])), mu_1, sigma_1
    # ('uniform', array([0.375, 0.8  , 0.8  , 0.625]), array([0.015625  , 0.00694444, 0.00694444, 0.015625  ])), mu_1, sigma_2
    # ('uniform', array([0.625, 0.2  , 0.2  , 0.375]), array([0.00390625, 0.00173611, 0.00173611, 0.00390625])), mu_2, sigma_1
    # ('uniform', array([0.625, 0.2  , 0.2  , 0.375]), array([0.015625  , 0.00694444, 0.00694444, 0.015625  ])), mu_2, sigma_2
    # ('guassian', array([0.375, 0.8  , 0.8  , 0.625]), array([0.00390625, 0.00173611, 0.00173611, 0.00390625])), mu_1, sigma_1
    # ('guassian', array([0.375, 0.8  , 0.8  , 0.625]), array([0.015625  , 0.00694444, 0.00694444, 0.015625  ])), mu_1, sigma_2
    # ('guassian', array([0.625, 0.2  , 0.2  , 0.375]), array([0.00390625, 0.00173611, 0.00173611, 0.00390625])), mu_2, sigma_1
    # ('guassian', array([0.625, 0.2  , 0.2  , 0.375]), array([0.015625  , 0.00694444, 0.00694444, 0.015625  ]))] mu_2, sigma_2
    
    #========================== OUTPUT VARIABLES LOG ==============================#
    filename = "req_opt_log.log"
    full_filename = os.path.join(DOE_out_dir,filename)
    
    index = 82949
    points = points[82949::]
    
    for point in points:
        
        index += 1
        
        req_thresh = [ 0.01, 0.1, 0.3, 0.3, 0.3, 0.8 ]
        eval_point = []
        call_type = 0
        req_vec = point
        [opt] = NOMAD_call(call_type,req_vec,req_thresh,eval_point,MADS_output_dir)
        print(opt)
        
        eval_point = [ 6 , 1 , 3 , -1 , -1 , -1 , -1 , 2]
        eval_point = opt
        call_type = 1
        [outs,weights] = NOMAD_call(call_type,req_vec,req_thresh,eval_point,MADS_output_dir)
        
        resiliance = [thresh - item  for thresh,item in zip(req_thresh,outs[1::])]
        
        f = outs[0]
        
        if index == 1: # initialize log file for writing
            resultsfile=open(full_filename,'w')
            resultsfile.write('index'+','+'n_stages'+','+'concept'+','+'s1'+','+'s2'+','+'s3'+','+'s4'+','+'s5'+','+'s6'+','
                              +'w1'+','+'w2'+','+'w3'+','+'w4'+','+'w5'+','+'w6'+','
                              +'R1'+','+'R2'+','+'R3'+','+'R4'+','+'R5'+','+'R6'+','
                              +'Total_weight'+'\n')
        
        resultsfile=open(full_filename,'a+')
        resultsfile.write(str(index)+','+','.join(map(str,opt))+','
                          +','.join(map(str,weights))+','
                          +','.join(map(str,resiliance))+','+str(f)+'\n')
        resultsfile.close()
        
        
        print(resiliance)
        print(weights)
    
if __name__ == "__main__":
    main()