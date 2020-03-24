# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 01:52:11 2020

@author: Khalil
"""

def plot_tradespace(attribute):
    
    import os
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    import random
    from itertools import permutations 
    
    filename = 'varout_opt_log.log'
    current_path = os.getcwd()
    filepath = os.path.join(current_path,'Optimization_studies',filename)
    
    data = [];
    
    attribute_name = attribute[0]
    attribute_label = attribute[1]
    
    # Read raw data
    with open(filepath) as f:
        lis = [line.split() for line in f]        # create a list of lists
        for i, x in enumerate(lis):              #print the list items 
            if i == 0:
                list_names = x[0].split(',')
            elif i == 1:
                line_data = x[0].split(',')
                line_data = [float(i) for i in line_data]
                data = np.array([line_data])
            else:
                line_data = x[0].split(',')
                line_data = [float(i) for i in line_data]
                data = np.vstack((data,line_data))
            
    # assign data to dictionary
    data = data.transpose()
    dictionary = dict(zip(list_names, data))
    
    # %% Initialize combinatorial optimization
    
    # This is not necessary if `text.usetex : True` is already set in `matplotlibrc`.    
    mpl.rc('text', usetex = True)
    rcParams['font.family'] = 'serif'
    my_dpi = 100
    fig = plt.figure(figsize=(700/my_dpi, 500/my_dpi), dpi=my_dpi);
    
    # CROSS CONCEPT
    l = list(permutations(range(0, 4))) # permutate indices
    branch_id = 0;
    for branch in l:
        branch = list(branch)
        
        # loop over data points
        i = 0; x_data = []; y_data = [];
        for c,i1,i2,i3,i4 in zip(dictionary['concept'],dictionary['i1'],dictionary['i2'],dictionary['i3'],dictionary['i4']):
                       
            # Get permutation index
            permutation_index = []
            for arg in [i1,i2,i3,i4]:
                if not int(arg) == -1:
                    permutation_index += [int(arg)] # populate permutation index
            
            if c == 1: # cross concept
                k = [];
                for item in permutation_index:
                    k += [pos for pos,x in enumerate(branch) if x == item]
                    
                # check if permutation index contained within branch
                if k == list(range(len(permutation_index))):
                    x_data += [dictionary['weight'][i]]
                    y_data += [dictionary[attribute_name][i]]
        
            i += 1;
        
        # generate random color for branch
        r = random.random()
        g = random.random()
        b = random.random()
        rgb = [r,g,b]
        
        plt.plot(x_data, y_data, ':', color = rgb, linewidth = 2.5 - (0.05*branch_id) );
        plt.plot(x_data, y_data, 'o', markersize=10, markevery=len(x_data), color = [0,0,0]);
        cross, = plt.plot(x_data, y_data, 'o', color = [0,0,1], markersize=6 );
        branch_id += 1
    
    # WAVE CONCEPT
    l = list(permutations(range(0, 3))) # permutate indices
    branch_id = 0;
    for branch in l:
        branch = list(branch)
        
        # loop over data points
        i = 0; x_data = []; y_data = [];
        for c,i1,i2,i3,i4 in zip(dictionary['concept'],dictionary['i1'],dictionary['i2'],dictionary['i3'],dictionary['i4']):
                       
            # Get permutation index
            permutation_index = []
            for arg in [i1,i2,i3,i4]:
                if not int(arg) == -1:
                    permutation_index += [int(arg)] # populate permutation index
            
            if c == 0: # cross concept
                k = [];
                for item in permutation_index:
                    k += [pos for pos,x in enumerate(branch) if x == item]
                
                 # check if permutation index contained within branch
                if k == list(range(len(permutation_index))):
                    x_data += [dictionary['weight'][i]]
                    y_data += [dictionary[attribute_name][i]]
        
            i += 1;
        
        # generate random color for branch
        r = random.random()
        g = random.random()
        b = random.random()
        rgb = [r,g,b]
        
        plt.plot(x_data, y_data, '-', color = rgb, linewidth = 2.5 - (0.05*branch_id) );
        start, = plt.plot(x_data, y_data, 'o', markersize=10, markevery=len(x_data), color = [0,0,0]);
        wave, = plt.plot(x_data, y_data, 'o', color = [1,0,0], markersize=6 );
        branch_id += 1
    
    ax = plt.gca() 
    ax.tick_params(axis='both', which='major', labelsize=14) 
    
    plt.title("Tradespace", fontsize=20);
    plt.xlabel('Weight of stiffener ($W$) - kg', fontsize=14)
    plt.ylabel('Requirement satisfaction ratio ($V_{{C}\cap{R}}/V_{R}$)', fontsize=14)
    plt.ylabel(attribute_label, fontsize=14)
    plt.ylabel('Safety factor ($n_{safety}$)', fontsize=14)
    
    return fig, ax, dictionary, start, wave, cross

if __name__ == "__main__":
    
    #attribute = ['n_f_th','Safety factor ($n_{safety}$)']
    attribute = ['resiliance_th','Requirement satisfaction ratio ($V_{{C}\cap{R}}/V_{R}$)']
    
    [fig, ax, dictionary, start, wave, cross] = plot_tradespace(attribute)