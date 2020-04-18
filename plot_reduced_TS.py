# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 18:08:48 2020

@author: Khalil
"""

def plot_tradespace_reduced(attribute):
    
    from scipy.io import loadmat
    from plot_tradespace import plot_tradespace
    from SBD_opt.post_process_DOE import NOMAD_call_BIOBJ
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    import random
    
    
    plt.close('all')
    attribute_name = attribute[0]
    attribute_label = attribute[1]
    
    [fig, ax, dictionary, start, wave, cross] = plot_tradespace(attribute)
    
    # Append initial base design to dictionary
    
    # Creating a Dictionary  
    new_dict = dict()
    for key in dictionary.keys():
        if key == 'n_f_th':
            new_key = np.append( dictionary[key], np.array([2.378925]) )
        elif key in ['i1', 'i2', 'i3', 'i4']:
            new_key = np.append( dictionary[key], np.array([-1.0]) )
        else:
            new_key = np.append( dictionary[key], np.array([0.0]) )
            
        new_dict[key] = new_key
    
    dictionary = new_dict
    
    # Get unsorted design points
    P_analysis_strip = []
    for c,i1,i2,i3,i4 in zip(dictionary['concept'],dictionary['i1'],dictionary['i2'],dictionary['i3'],dictionary['i4']):
        # Get permutation index
        permutation_index = []
        for arg in [c,i1,i2,i3,i4]:
            if not int(arg) == -1:
                permutation_index += [int(arg)] # populate permutation index
        
        P_analysis_strip += [permutation_index]

    # Get sorted design points wrt attribute
    i = np.argsort(dictionary[attribute[0]])
    
    n_f_th = dictionary['n_f_th'][i]
    weight = dictionary['weight'][i]
    c = dictionary['concept'][i]
    i1 = dictionary['i1'][i]
    i2 = dictionary['i2'][i]
    i3 = dictionary['i3'][i]
    i4 = dictionary['i4'][i]
    
    sorted_designs = []
    for n in range(len(n_f_th)):
        
        # Get permutation index
        permutation_index = []
        for arg in [c[n],i1[n],i2[n],i3[n],i4[n]]:
            if not int(arg) == -1:
                permutation_index += [int(arg)] # populate permutation index
        
        sorted_designs += [permutation_index]
    
    # This is not necessary if `text.usetex : True` is already set in `matplotlibrc`.    
    mpl.rc('text', usetex = True)
    mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}',
                                           r'\usepackage{amssymb}']
    rcParams['font.family'] = 'serif'
    my_dpi = 100
    magnify = 1.0 # 1.75
    fig = plt.figure(figsize=(magnify * 700/my_dpi, magnify * 500/my_dpi), dpi=my_dpi)
    
    ax = plt.gca() 
    ax.tick_params(axis='both', which='major', labelsize=14 * magnify) 
    # ax.set_xlim([3.2040081632482558, 27.856089113953463])
    # ax.set_ylim([1.4864630521225, 4.6238964381875])
    ax.set_xlim([-1.1615516283612148, 24.392584195585513])
    ax.set_ylim([-0.2240642914865, 4.7053501212165])
    
    plt.title("Tradespace", fontsize=20 * magnify)
    plt.xlabel('Weight of stiffener ($W$) - kg', fontsize=14 * magnify)
    plt.ylabel(attribute_label, fontsize=14 * magnify)

    #reduced_designs = [sorted_designs[n] for n in [60, 58,-1, 3]]
    #pt_labels = ['d', 'o', '+', 's']
    #marker_sizes = [ 8, 8, 22, 8 ]
    #marker_widths = [ 2, 2, 3, 2 ]
    # reduced_designs = [[1,0,3,1,2],[1,0,1,2],[0,0,1,2]] # Interesting cases in P(R) domain
    # reduced_designs = [[1,2,1,0],[1,3,2],[1,1,2,0]] # Interesting cases in n_safety domain
    reduced_designs = [[1, 1, 0],[1, 3, 2],[1, 1, 0, 2],[1, 3],[1, 1, 2, 0]] # Top 5 design for R50 DOE
    pt_labels = ['+', 's', 'o', 'x', '*']
    marker_sizes = [ 16, 8, 8, 8, 8]
    marker_widths = [ 3, 2, 2, 2, 8]
    
    legend_handles = []; legend_labels = []; d_i = 1
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] # ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', ...]
    
    req_index = 0
    attribute = dictionary['n_f_th']
    cost = dictionary['weight']
    [x_data, y_data] = NOMAD_call_BIOBJ(req_index,P_analysis_strip,attribute,cost,False) # get Pareto optimal points
    pareto, = plt.plot(x_data, y_data, '-o', color = 'm', linewidth = 2.0, markersize = 5.0 )

    legend_handles += [pareto]
    legend_labels += ['Pareto front']
    ax.legend(legend_handles, legend_labels, loc = 'lower right', fontsize=10 * magnify )
    fig.savefig('progress/tradespace_%i.pdf' %(d_i), format='pdf', dpi=100,bbox_inches='tight')

    for design,pt_label,e_width,e_size in zip(reduced_designs,pt_labels,marker_widths,marker_sizes):
    
        # x_data = [0.0]; y_data = [2.378925]
        x_data = [0.0]; y_data = [0.0]
        print(design)
        for it in range(len(design[1::])):
            ind = P_analysis_strip.index(design[0:it+2])
            x_data += [dictionary['weight'][ind]]
            y_data += [dictionary[attribute_name][ind]]
        
        plt.plot(x_data, y_data, '-', color = [0,0,0], linewidth = 1.5 )
        plt.plot(x_data, y_data, 'o', color = [0,0,0], markersize=6 )
        design_lg, = plt.plot( x_data[-1], y_data[-1], pt_label, markersize = e_size, markeredgewidth = e_width, color = colors[d_i] )
        
        legend_handles += [design_lg]
        # legend label generation
        label = ''.join(['$n = %i,' %(len(design[1::])),
                         '~c = %i,' %(design[0]),
                         '~D = \{', 
                         ',~'.join(map(str,design[1::])),'\}$'])

        legend_labels += [label]
        ax.legend(legend_handles, legend_labels, loc = 'lower right', fontsize=10 * magnify )

        d_i += 1
        fig.savefig('progress/tradespace_%i.pdf' %d_i, format='pdf', dpi=100,bbox_inches='tight')
    
    # print(ax.get_xlim())
    # print(ax.get_ylim())

    return fig, ax, dictionary, P_analysis_strip

if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt

    attribute = ['n_f_th','Safety factor ($n_{safety}$)']
    # attribute = ['resiliance_th','Probability of satisfying requirement $\mathbb{P}(\mathbf{T} \in C)$']
    
    [fig, ax, dictionary, P_analysis_strip] = plot_tradespace_reduced(attribute)
    fig.savefig(os.path.join(os.getcwd(),'tradespace_pareto_reduced.svg'), format='svg')

    plt.show()