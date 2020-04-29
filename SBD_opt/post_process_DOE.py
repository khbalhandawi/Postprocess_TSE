# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 18:06:53 2020

@author: Khalil
"""

#==============================================================================
# Acquire requirements DOE design results and 
# get dictionary data from design log and resiliance log
def plot_tradespace(attribute,filename_opt,filename_res,filename_weight,plot_true):
    
    import numpy as np
    import random
    import copy
        
    if plot_true:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from matplotlib import rcParams
        plt.close('all')
    
    current_path = os.getcwd()
    filepath = os.path.join(current_path,'DOE_results',filename_opt)
    
    data = []
    
    attribute_label = attribute[0]
    
    data = np.loadtxt(filepath, skiprows = 1, delimiter=",")
    
    # Read raw data
    with open(filepath) as f:
        line = f.readline()
        list_names = line.split(',')
        list_names[-1] = list_names[-1].split('\n')[0]
        
    # assign data to dictionary
    data = data.transpose()
    dictionary = dict(zip(list_names, data))
    #==========================================================================
    # read resiliance data
    current_path = os.getcwd()
    filepath = os.path.join(current_path,'Input_files',filename_res)
    
    data = []
    
    attribute_label = attribute[0]
    
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
    dictionary_res = dict(zip(list_names, data))
    #==========================================================================
    # read weight data
    current_path = os.getcwd()
    filepath = os.path.join(current_path,'Input_files',filename_weight)
    
    data = []
    
    attribute_label = attribute[0]
    
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
    dictionary_weight = dict(zip(list_names, data))
    
    # %% Initialize combinatorial optimization
    
    if plot_true:
        # This is not necessary if `text.usetex : True` is already set in `matplotlibrc`.    
        mpl.rc('text', usetex = True)
        rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}',
                                           r'\usepackage{amssymb}']
        rcParams['font.family'] = 'serif'
        my_dpi = 100
        fig = plt.figure(figsize=(700/my_dpi, 500/my_dpi), dpi=my_dpi)
    
    branch_id = 1
    designs = []; designs_padded = []
    # loop over data points
    for data in zip(dictionary['concept'],
                    dictionary['s1'],dictionary['s2'],dictionary['s3'],
                    dictionary['s4'],dictionary['s5'],dictionary['s6'],
                    dictionary['w1'],dictionary['w2'],dictionary['w3'],
                    dictionary['w4'],dictionary['w5'],dictionary['w6'],
                    dictionary['R1'],dictionary['R2'],dictionary['R3'],
                    dictionary['R4'],dictionary['R5'],dictionary['R6']):
        
        [c,s1,s2,s3,s4,s5,s6,w1,w2,w3,w4,w5,w6,R1,R2,R3,R4,R5,R6] = data
        
        if c != -1:
            
            resiliance = [R1,R2,R3,R4,R5,R6]
            weight = [w1,w2,w3,w4,w5,w6]
            
            # Get design index
            i = 0
            design_index = [int(c),]; x_data = []; y_data = []
            for arg in [s1,s2,s3,s4,s5,s6]:
                if not int(arg) == -1:
                    design_index += [int(arg)] # populate permutation index
                
                x_data += [i+1]
                y_data += [resiliance[i]]
                
                i += 1
            
            design_index_padded = copy.deepcopy(design_index)
            if len(design_index) < 5:
                design_index_padded += [-1]*(5-len(design_index))
            
            designs_padded += [design_index_padded]
            designs += [design_index]
            
            if plot_true:
                # generate random color for branch
                r = random.random()
                g = random.random()
                b = random.random()
                rgb = [r,g,b]

                plt.plot(x_data, y_data, ':', color = rgb, linewidth = 2.5 )
                # plt.plot(x_data, y_data, 'o', markersize=10, markevery=len(x_data), color = [0,0,0])
                plot_h, = plt.plot(x_data, y_data, 'o', color = [0,0,1], markersize=6 )
            
            branch_id += 1
            
            if branch_id == 1000000 + 1:
                break
      
    if plot_true:
        ax = plt.gca() 
        ax.tick_params(axis='both', which='major', labelsize=14) 
        
        plt.title("Tradespace", fontsize=20)
        plt.xlabel('Design stage number', fontsize=14)
        plt.ylabel(attribute_label, fontsize=14)
    
        return dictionary,dictionary_res,dictionary_weight,designs,designs_padded,fig,ax,plot_h
    else:
        return dictionary,dictionary_res,dictionary_weight,designs,designs_padded

#==============================================================================
# Import feasibility data
def import_feasibility(filename_feas):
    
    import numpy as np
    
    current_path = os.getcwd()
    filepath = os.path.join(current_path,'DOE_results',filename_feas)
    
    data = np.loadtxt(filepath, skiprows = 1, delimiter=",")
    
    # Read raw data
    with open(filepath) as f:
        line = f.readline()
        list_names = line.split(',')
        list_names[-1] = list_names[-1].split('\n')[0]
        
    # assign data to dictionary
    data = data.transpose()
    dictionary = dict(zip(list_names, data))
    feasiblity = (np.sum(data[1:], axis = 1) * 100) / len(data[0])

    return data, feasiblity
    
#==============================================================================
# Calculate filtered outdegree
def filtered_outdegree(dictionary_weight):
    
    design_list = []; FO = []
    for data in zip(dictionary_weight['concept'],
                    dictionary_weight['i1'],dictionary_weight['i2'],
                    dictionary_weight['i3'],dictionary_weight['i4']):
        design_list += [[int(x) for x in data]]

        deposits = data[1:]
        if data[0] == 0: # wave stiffener
            FO += [2 - sum(d != -1 for d in deposits)]
        elif data[0] == 1: # hatched stiffener
            FO += [4 - sum(d != -1 for d in deposits)]

    return FO
    

#==============================================================================
# Call MADS and get Pareto optimal solutions from MADS
def NOMAD_call_BIOBJ(req_index,weight_file,res_ip_file,res_th_file,
                     P_analysis_strip,attribute,cost,evaluate):
    
    if evaluate: # run bi-objective optimization
        from sample_requirements import system_command

        command = "categorical_biobj %i %s %s %s" %(req_index,weight_file,res_ip_file,res_th_file)
        print(command)
        system_command(command)
    
    [x_data, y_data] = get_pareto(P_analysis_strip,attribute,cost)

    return x_data, y_data

#==============================================================================
# Get Pareto optimal solutions from MADS
def get_pareto(P_analysis_strip,attribute,cost):
    import os
    import csv
    
    filename = 'mads_x_opt_pareto.log'
    current_path = os.getcwd()
    filepath = os.path.join(current_path,'MADS_output',filename)
    
    # %% read MADS log file
    opt_points = []
    with open(filepath) as csv_file:
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
    for point in opt_points:
        print(point)
    
    x_data = []; y_data = []
    for point in opt_points:
        ind = P_analysis_strip.index(point)

        attribute_Pareto = attribute[ind]
        cost_Pareto = cost[ind]
        
        x_data += [cost_Pareto]
        y_data += [attribute_Pareto]
        
    return x_data, y_data

#==============================================================================
# rank designs and get histogram distribution
def rank_designs(designs):
    
    from scipy.io import loadmat
    import os
    
    # one-liner to read a single variable
    filename = 'DOE_permutations.mat'
    current_path = os.getcwd()
    filepath = os.path.join(current_path,'Input_files',filename)
    
    P_analysis = loadmat(filepath)['P_analysis']
    
    index = 0; P_analysis_strip = []
    histogram = np.zeros(len(P_analysis), dtype = int) 
    
    P_analysis_strip = []
    for P_i in P_analysis:

        # Get permutation index
        permutation_index = []
        for arg in P_i:
            if not int(arg) == -1:
                permutation_index += [int(arg)] # populate permutation index
        P_analysis_strip += [permutation_index]
        
        # loop over data points
        for design in designs:

            # check if permutation index contained within branch
            if design == permutation_index:
                histogram[index] += 1
        
        index += 1 

    return histogram,P_analysis_strip

#==============================================================================
# Get pie chart distribution for each deposit and concept
def get_dstribution(d_types,k,design_list):
    
    dist = np.zeros(len(d_types), dtype = int) 
    
    index = 0  
    for d_type in d_types:
        
        # loop over data points
        for design in design_list:
            # check if permutation index contained within branch
            if design[k] == d_type:
                dist[index] += int(1)   
        
        index += 1
        
    return dist

#==============================================================================
# Convert color floats to hex
def rgb2hex(color):
    """Converts a list or tuple of color to an RGB string

    Args:
        color (list|tuple): the list or tuple of integers (e.g. (127, 127, 127))

    Returns:
        str:  the rgb string
    """
    return f"#{''.join(f'{hex(c)[2:].upper():0>2}' for c in color)}"

#==============================================================================
# MAIN CALL
if __name__ == "__main__":
    
    import os
    import random
    import pickle
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    from matplotlib.ticker import MaxNLocator
    
    mpl.rcParams['font.size'] = 14.0
    
    plt.close('all')

    filename_opt = 'req_opt_log_R50.log'
    filename_res = 'resiliance_th.log'
    filename_res_ip = 'resiliance_ip.log'
    filename_weight = 'varout_opt_log.log'
    filename_feas = 'feasiblity_log_th0.99.log'

    #attribute = ['n_f_th','Safety factor ($n_{safety}$)']
    attribute = ['$\mathbb{P}(\mathbf{T} \in C)$']
    
    # [fig, ax, dictionary, plot_h, designs] = plot_tradespace(attribute,True)
    [dictionary, dictionary_res, dictionary_weight, designs, designs_padded] = plot_tradespace(attribute,filename_opt,filename_res,filename_weight,False)
    [histogram,P_analysis_strip] = rank_designs(designs)
    
    design_list = []
    for data in zip(dictionary_weight['concept'],
                    dictionary_weight['i1'],dictionary_weight['i2'],
                    dictionary_weight['i3'],dictionary_weight['i4']):
        design_list += [[int(x) for x in data]]
    
    #==============================================================================
    # Set based sorting

    xS = range(0,len(histogram))
    yS = sorted(histogram,reverse = True)
    indices = [x for _,x in sorted(zip(histogram,xS),reverse = True)]
    sorted_designs = [x for _,x in sorted(zip(histogram,design_list),reverse = True)]
    
    print('Top 5 designs:')
    for design in sorted_designs[:5]:
        print(design)
    # data for constructing tradespace of feasible designs
    # req_index = 50
    # attribute = dictionary_res['req_index_%i' %(req_index)]
    req_index = 0
    attribute = dictionary_weight['n_f_th']
    cost = dictionary_weight['weight']

    # Data for plotting SBD designs
    attribute_SBD = attribute[indices]
    cost_SBD = cost[indices]
    
    #==============================================================================
    # Feasiblity sorting
    [data,robustness] = import_feasibility(filename_feas)

    xR = range(0,len(robustness))
    yR = sorted(robustness,reverse = True)
    indices_robust = [x for _,x in sorted(zip(robustness,xR),reverse = True)]
    sorted_designs_robust = [x for _,x in sorted(zip(robustness,design_list),reverse = True)]
    
    print('Top 5 robust designs:')
    for design in sorted_designs_robust[:5]:
        print(design)
    # data for constructing tradespace of feasible designs
    # req_index = 50
    # attribute = dictionary_res['req_index_%i' %(req_index)]

    # Data for plotting SBD designs
    attribute_robust = attribute[indices_robust]
    cost_robust = cost[indices_robust]

    #==============================================================================
    # flexibility sorting
    FO = filtered_outdegree(dictionary_weight)

    xF = range(0,len(FO))
    yF = sorted(FO,reverse = True)
    indices_flexible = [x for _,x in sorted(zip(FO,xF),reverse = True)]
    sorted_designs_flexible = [x for _,x in sorted(zip(FO,design_list),reverse = True)]
    
    print('Top 5 flexible designs:')
    for design in sorted_designs_flexible[:5]:
        print(design)
    # data for constructing tradespace of feasible designs
    # req_index = 50
    # attribute = dictionary_res['req_index_%i' %(req_index)]

    # Data for plotting SBD designs
    attribute_flexible = attribute[indices_flexible]
    cost_flexible = cost[indices_flexible]

    #==========================================================================
    # Histogram colors generator

    new_colors = False
    color_file = 'colors_histogram.pkl'
    # ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', ...]
    colors_default = plt.rcParams['axes.prop_cycle'].by_key()['color']

    top_10 = indices[:10]
    if new_colors:
        colors = []
        for i in range(0,len(histogram)):
            r = random.random()
            g = random.random()
            b = random.random()
            rgb = [int(r*256),int(g*256),int(b*256)]
            colors += [rgb2hex(rgb).lower()]
        
        # color top 10 designs using default color palette
        for i,color in zip(top_10,colors_default):
            colors[i] = color

        with open(color_file, 'wb') as pickle_file:
            pickle.dump(colors,pickle_file)
    else:
        fid = open(color_file, 'rb')
        colors = pickle.load(fid)
        fid.close()

    #==========================================================================
    # Histogram plot for robust designs
    # data for plotting histogram feasibility

    n_bins = 20
    x = xR[:n_bins]
    y = yR[:n_bins]
    design_indices = [i + 1 for i in indices_robust[:n_bins]]
    design_colors = [colors[i] for i in indices_robust[:n_bins]]

    # design_indices = list(range(1,21)

    mpl.rc('text', usetex = True)
    rcParams['font.family'] = 'serif'
    my_dpi = 100
    fig = plt.figure(figsize=(700/my_dpi, 500/my_dpi), dpi=my_dpi)
    
    barlist = plt.bar(x, y, width=0.8, bottom=None, align='center', data=None, color=design_colors )
    # for bar in barlist[:5]: # set color of first five bars
    #     bar.set_edgecolor('r')
    #     bar.set_linewidth(2.3)

    plt.xlabel('Design index', fontsize=14)
    plt.ylabel('$\%$ of multi-stage problems satisfied', fontsize=14) 
    plt.xticks(list(range(n_bins)), list(map(str,design_indices)))

    tight_bbox = fig.get_tightbbox(fig.canvas.get_renderer()) # get bbox for figure canvas
    fig.savefig(os.path.join(os.getcwd(),'DOE_results','histogram_DOE_R.pdf'), format='pdf', dpi=1000,bbox_inches=tight_bbox)

    #==========================================================================
    # Histogram plot for flexible designs
    # data for plotting histogram feasibility

    n_bins = 20
    x = xF[:n_bins]
    y = yF[:n_bins]
    design_indices = [i + 1 for i in indices_flexible[:n_bins]]
    design_colors = [colors[i] for i in indices_flexible[:n_bins]]

    # design_indices = list(range(1,21)

    mpl.rc('text', usetex = True)
    rcParams['font.family'] = 'serif'
    my_dpi = 100
    fig = plt.figure(figsize=(700/my_dpi, 500/my_dpi), dpi=my_dpi)
    
    barlist = plt.bar(x, y, width=0.8, bottom=None, align='center', data=None, color=design_colors )
    # for bar in barlist[:5]: # set color of first five bars
    #     bar.set_edgecolor('r')
    #     bar.set_linewidth(2.3)

    plt.xlabel('Design index', fontsize=14)
    plt.ylabel('Filtered outdegree', fontsize=14) 
    plt.xticks(list(range(n_bins)), list(map(str,design_indices)))
    plt.yticks(list(range(max(yF)+1)), list(map(str,range(max(yF)+1)))) # Force integer ticks
    fig.savefig(os.path.join(os.getcwd(),'DOE_results','histogram_DOE_F.pdf'), format='pdf', dpi=1000,bbox_inches=tight_bbox)

    #==========================================================================
    # Histogram plot
    # data for plotting histogram SBD

    n_bins = 20
    x = xS[:n_bins]
    y = yS[:n_bins]/(sum(yS)*0.01)
    design_indices = [i + 1 for i in indices[:n_bins]]
    design_colors = [colors[i] for i in indices[:n_bins]]

    # design_indices = list(range(1,21)

    mpl.rc('text', usetex = True)
    rcParams['font.family'] = 'serif'
    my_dpi = 100
    fig = plt.figure(figsize=(700/my_dpi, 500/my_dpi), dpi=my_dpi)
    
    barlist = plt.bar(x, y, width=0.8, bottom=None, align='center', data=None, color=design_colors )
    # for bar in barlist[:5]: # set color of first five bars
    #     bar.set_edgecolor('r')
    #     bar.set_linewidth(2.3)

    plt.xlabel('Design index', fontsize=14)
    plt.ylabel('$\%$ of multi-stage problems solved', fontsize=14) 
    plt.xticks(list(range(n_bins)), list(map(str,design_indices)))

    fig.savefig(os.path.join(os.getcwd(),'DOE_results','histogram_DOE.pdf'), format='pdf', dpi=1000,bbox_inches=tight_bbox)
    
    #==========================================================================
    # Tradespace plot
    # data for plotting Pareto front
    [x_data, y_data] = NOMAD_call_BIOBJ(req_index,filename_weight,filename_res_ip,filename_res,
                                        P_analysis_strip,attribute,cost,False) # get Pareto optimal points

    my_dpi = 100
    fig = plt.figure(figsize=(700/my_dpi, 500/my_dpi), dpi=my_dpi)
    
    markersizes = [ (1/20)*(n**4) for n in reversed(range(3,len(cost_SBD[:10])+3)) ]

    feasible, = plt.plot(cost, attribute, 'x', color = [0,0,0], linewidth = 1, markersize = 6 )
    pareto, = plt.plot(x_data, y_data, '-d', color = 'm', linewidth = 4.0, markersize = 7.0 )
    Robust_design = plt.scatter( cost_robust[:5], attribute_robust[:5], s = 600, color = [0,0,1], marker = '.', zorder=3)
    flexible_design = plt.scatter( cost_flexible[:5], attribute_flexible[:5], s = 300, color = [34/256,139/256,34/256], marker = '.', zorder=4 )
    SBD_design = plt.scatter( cost_SBD[:5], attribute_SBD[:5], s = 150, color = [1,0,0], marker = '.', zorder=5 )
    plt.xlabel('Weight of stiffener ($W$) - kg', fontsize=14)
    # plt.ylabel('Requirement satisfaction ratio ($V_{{C}\cap{R}}/V_{R}$)', fontsize=14)
    plt.ylabel('Safety factor ($n_{safety}$)', fontsize=14)
    
    ax = plt.gca() 
    ax.tick_params(axis='both', which='major', labelsize=14)
    # ax.legend((feasible, pareto), 
    #           ('Feasible designs $\in \Omega$', 'Pareto optimal designs'), loc = 'lower right',
    #            fontsize = 9.0) 
    # ax.legend((feasible, pareto, Robust_design), 
    #           ('Feasible designs $\in \Omega$', 'Pareto optimal designs', 
    #            'Robust designs'), loc = 'lower right',
    #            fontsize = 9.0)
    # ax.legend((feasible, pareto, Robust_design, flexible_design), 
    #           ('Feasible designs $\in \Omega$', 'Pareto optimal designs', 
    #            'Robust designs', 'Flexible designs'), loc = 'lower right',
    #            fontsize = 9.0)
    ax.legend((feasible, pareto, Robust_design, flexible_design, SBD_design), 
              ('Feasible designs $\in \Omega$', 'Pareto optimal designs', 
               'Robust designs', 'Flexible designs', 'Set-based designs'), loc = 'lower right',
               fontsize = 9.0)

    fig.savefig(os.path.join(os.getcwd(),'DOE_results','tradespace_pareto.pdf'), format='pdf', dpi=1000,bbox_inches='tight')
    
    # print(ax.get_xlim())
    # print(ax.get_ylim())
    #==========================================================================
    # Pie chart plot   
    import matplotlib.gridspec as gridspec

    d_types = [0,1,2,3]
    i1 = get_dstribution(d_types,1,designs_padded)
    i2 = get_dstribution(d_types,2,designs_padded)
    i3 = get_dstribution(d_types,3,designs_padded)
    i4 = get_dstribution(d_types,4,designs_padded)
    
    d_types = [0,1]
    c = get_dstribution(d_types,0,designs_padded)
    
    gs = gridspec.GridSpec(2,2, 
                           width_ratios = np.ones(2,dtype=int), 
                           height_ratios = np.ones(2,dtype=int))
    
    my_dpi = 100
    fig = plt.figure(figsize=(1400/my_dpi, 1000/my_dpi), dpi=my_dpi)
    
    # Plot all stages
    iteraton = -1
    for item in [i1,i2,i3,i4]:
        
        iteraton += 1
        
        # Pie chart, where the slices will be ordered and plotted counter-clockwise:
        labels = 'deposit 0', 'deposit 1', 'deposit 2', 'deposit 3'
        sizes = item
        explode = (0, 0, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')
        
        ax = fig.add_subplot(gs[iteraton]) # subplot
        ax.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        
    # Plot 1st stage only
    my_dpi = 100
    fig = plt.figure(figsize=(700/my_dpi, 500/my_dpi), dpi=my_dpi)
        
    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    labels = 'deposit 0', 'deposit 1', 'deposit 2', 'deposit 3'
    sizes = i1
    explode = (0, 0, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')
    
    ax = plt.gca()
    patches, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                                shadow=False, startangle=90)
    
    for i in range(len(labels)):
        texts[i].set_fontsize(18)
        autotexts[i].set_fontsize(18)
    
    # Color a section of the pie
    # patches[1].set_edgecolor('b')
    # patches[1].set_linewidth(3)
    # patches[1].set_hatch('\\')
        
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    fig.savefig(os.path.join(os.getcwd(),'DOE_results','1st_stage_pie.pdf'), format='pdf', dpi=1000,bbox_inches='tight')
    
    # Plot concept only
    my_dpi = 100
    fig = plt.figure(figsize=(700/my_dpi, 500/my_dpi), dpi=my_dpi)
        
    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    labels = 'concept 1', 'concept 2'
    sizes = c
    explode = (0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')
    
    ax = plt.gca()
    ax.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=False, startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    
    #==========================================================================
    plt.show()