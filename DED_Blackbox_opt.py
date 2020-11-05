# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 01:34:24 2017

@author: Khalil
"""

#==============================================================================#
# %% HIGH-CYCLE FATIGUE PARAMTERS
def fatigue_parameters(S_ut,S_y,compression):
    
    import numpy as np
    
    a = 4.51
    b = -0.265
    ka = a*(S_ut**(b)) # surface condition modification factor
    kb = 1.0 # size modification factor
    kc  = 0.85 # load modification factor
    kd = 0.8 # temperature modification factor
    ke = 0.7 # reliability factor
    kf = 1.0 # miscellaneous-effects modification factor
    
    K_f = 1.0 # Stress concentration factors (SAME SIZE AS HIST REGIONS)
    
    S_e = ka*kb*kc*kd*ke*kf*S_ut
    
    f = 0.76 # Fig 6-18 fatigue strength fraction
    a = ((f*S_ut)**2)/S_e
    b = (-1/3)*np.log10((f*S_ut)/S_e)
    # print("Se: %f" %(S_e))
    # print("a: %f, b: %f \n" %(a,b))
    
    return f,a,b,S_e

#==============================================================================#
# %% CALCULATE FATIGUE LIFE AND SAFETY FACTOR
def fatigue_life_calc(sa,sm):
    # import scipy.integrate as integrate
    import numpy as np
    from numpy import pi
    
    smax = sa + sm
    smin = sm - sa
    
    sopen = max([smin,0.0]) # to account for crack closure effects
    smax = max([smax,1.0]) # to make sure there is no division by zero or inf
    
    K1c = 30 # MPa/sqrt(m)
    
    # it_integral = 5.0
    # C = 4e-10; # From J. M. Barsom and S. T. Rolfe
    it_integral = 100.0
    C = 2e-7 # From J. M. Barsom and S. T. Rolfe
    m = 1 # From J. M. Barsom and S. T. Rolfe
    # it_integral = 100.0
    # C = 2e-9; # From J. M. Barsom and S. T. Rolfe
    # m = 3; # From J. M. Barsom and S. T. Rolfe
    ai = 5e-4 # m
    beta = 1.00
    af = ( 1/pi ) * ( ( K1c / (beta*smax) ) ** 2 )
    # print('critical crack size: %f mm' %(af*1000.0))
    
    dN_j = (af - ai)*100*500
    
    a_j = ai; N_j = 0.0; a_j_plot = []; N_j_plot = []
    
    while a_j < af:
    
        Kmax = beta*smax*np.sqrt(pi*a_j)
        Kmin = beta*sopen*np.sqrt(pi*a_j)
        R = Kmin/Kmax
    
        dK_1 = Kmax - Kmin
        da_j = ( ( C * ( (dK_1) ** m ) ) / ( ((1 - R) * K1c) - dK_1) ) * dN_j # Foreman relation
        # da_j = ( ( C * ( (dK_1) ** m ) ) / ( 1.0 ) ) * dN_j # Paris law
        dN_j = ((af - ai)/da_j) * it_integral
        a_j += da_j
        N_j += dN_j
    
        # a_j_plot += [a_j]
        # N_j_plot += [N_j]
    
        # print(a_j*1000)
    
    # print(N_j)
    
    # plt.figure()
    # plt.plot(N_j_plot, a_j_plot, '-r', label='Nominal')
    # plt.show()
    
    # result = integrate.quad(lambda a: 1/((beta*s_eff*np.sqrt(pi*a))**m), ai, af)
    # Nf = (1/C)*result
    
    return N_j

#==============================================================================#
# %% FATIGUE CALCULATION SUBROUTINE
def fatigue_calculation( s1,s2,p1,p2 ):
    
    import numpy as np
    
    s1 = np.array(s1); s2 = np.array(s2)
    p1 = np.array(p1); p2 = np.array(p2)
    
    # Find sign of Mises stress (-: comp, +: tension)
    s1 = s1*np.sign(p1)
    s2 = s2*np.sign(p2)
    
    # Find sign of Mises stress (-: comp, +: tension)
    s1 = s1* -1.0; s2 = s2* -1.0
    s_m0 = ((s1+s2)/2.0)
    s_a0 = abs(s2-s1)/2.0
    
    K_f = 1.0
    
    s_m = np.asarray(s_m0) * K_f
    s_a = np.asarray(s_a0) * K_f
    
    i = -1
    N = np.empty(len(s_a))
    n_f = np.empty(len(s_a))
    S_rev = np.empty(len(s_a))
    for s_an,s_mn in zip(s_a,s_m):
        i += 1
        S_ut = 1332.0 # Ultimate strength
        S_y = 1100.0 # Yield strength
        compression = True
        [f,a,b,S_e] = fatigue_parameters(S_ut,S_y,compression)
        N[i] = fatigue_life_calc(s_an,s_mn)
    
        if s_mn <= 0: # Compressive mean stress  
            if abs(s_mn) > S_ut or s_an > S_y: # Mean stress or Amplitude stress greater than ultimate stress
                #S_rev[i] = s_an;
                #N[i] = (S_rev[i]/a)**(1/b) # Estimate remaining life
                n_f[i] = min([S_ut/abs(s_mn), S_ut/s_an]) # Ultimate failure safety factor
                # n_f[i] = min([S_ut/abs(s_mn), S_y/s_an]) # More conservative
            else:
                n_f[i] = min([S_ut/abs(s_mn), S_ut/s_an]) # Ultimate failure or endurance limit safety factor
                # n_f[i] = min([S_ut/abs(s_mn), S_e/s_an]) # More conservative
        else: # Tensile mean stress
            if abs(s_mn) + s_an >= S_y:
                # S_rev[i] = s_an;
                # N[i] = 0.0; # First cycle tensile yielding
                n_f[i] = 1/((s_an/S_y)+(abs(s_mn)/S_y)); # Langer safety factor
            else:
                n_f[i] = min([1/((s_an/S_e)+(abs(s_mn)/S_ut)), # Modified goodman safety factor
                              1/((s_an/S_y)+(abs(s_mn)/S_y))]) # Langer safety factor
    
    return s_m0, s_a0, N, n_f
   
#==============================================================================#
# %% SCALING BY A RANGE
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
# %% GET AM PROCESS RESULTS
def get_AM_results(index,current_path):
    from abaqus_postprocess import text_read_element_AM, text_read_nodal
    import os
    import numpy as np
    
    print("-------------------- %s -------------------\n" %('TRANSIENT RESULTS'))
    
    #--------------------------------------------------------------------------#
    # Get elemental static results
    filename = "%i_hist_out_e_file.log" %(index)
    hist_e_full_name = os.path.join(current_path,'Job_results','Results_log',filename)
    
    [time_table,stress_table_max,stress_table_min,stress_table_avg,
        temp_table_max,temp_table_min,temp_table_avg,
        S11_table_max,S11_table_min,S11_table_avg,
        S33_table_max,S33_table_min,S33_table_avg,
        fig_title,time_norm] = text_read_element_AM( hist_e_full_name )

    s_res = np.array([])
    s_11_res = np.array([])
    s_33_res = np.array([])
    t_max = np.array([])
    # (REMOVE NORMALIZED TIME LAST ROW)
    for n in range(len(time_table)-2): # iterate over number of sets
        s_res = np.append(s_res,stress_table_max[n][-1]) # get residual stress at last time frame
        s_11_res = np.append(s_11_res,S11_table_max[n][-1]) # get S11 at last time frame
        s_33_res = np.append(s_33_res,S33_table_min[n][-1]) # get S33 at last time frame

    #--------------------------------------------------------------------------#
    # Get nodal static results
    filename = "%i_hist_out_n_file.log" %(index)
    hist_n_full_name = os.path.join(current_path,'Job_results','Results_log',filename)

    [time_table,U_table_max,U_table_min,U_table_avg,time_norm] = text_read_nodal( hist_n_full_name )

    U_res = np.array([])
    # (REMOVE NORMALIZED TIME LAST ROW)
    for n in range(len(time_table)-2): # iterate over number of sets
        U_res = np.append(U_res,U_table_max[n][-1]) # get U at last time frame
    
    #--------------------------------------------------------------------------#
    index_s_res = max(range(len(s_res)), key=s_res.__getitem__)
    print('The maximum residual stress is: %f MPa| at %s' %(max(s_res),fig_title[index_s_res]))
    print('The maximum displacement is: %f mm' %(U_res[0]))
    
    return s_res,s_res,s_11_res,s_33_res,U_res

#==============================================================================#
# %% GET THERMAL LOADCASE RESULTS
def get_loadcase_results(e_file,n_file,current_path,DOE_results_folder):
    from abaqus_postprocess import text_read_loadcase, text_read_nodal
    import os
    import numpy as np

    # print("-------------------- %s ----------------------\n" %('LOAD CASE RESULTS'))
    
    #--------------------------------------------------------------------------#
    # Get elemental static results
    filename = e_file
    e_full_name = os.path.join(current_path,'Job_results',DOE_results_folder,filename)
    
    [time_table,stress_table_max,stress_table_min,stress_table_avg,
            temp_table_max,temp_table_min,temp_table_avg,
            S_hoop_table_max,S_hoop_table_min,S_hoop_table_avg,
            S_M_table_max,S_M_table_min,S_M_table_avg,
            S_A_table_max,S_A_table_min,S_A_table_avg,
            N_table_max,N_table_min,N_table_avg,
            n_f_table_max,n_f_table_min,n_f_table_avg,
            fig_title,time_norm] = text_read_loadcase( e_full_name )

    s = np.array([])
    N = np.array([])
    n_f = np.array([])
    for n in range(len(time_table)-2): # (REMOVE NORMALIZED TIME LAST ROW)
        s = np.append(s,S_hoop_table_max[n][-1])
        N = np.append(N,N_table_min[n][-1])
        n_f = np.append(n_f,n_f_table_min[n][-1])
        #=======================================================================
        # s = np.append(s,S_hoop_table_avg[n][-1])
        # N = np.append(N,N_table_avg[n][-1])
        # n_f = np.append(n_f,n_f_table_avg[n][-1])
        #=======================================================================
        
    index_nf = min(range(len(N)), key=N.__getitem__)
    # print('\nThe minimum fatigue life is: %f | at %s\n' %(min(N),fig_title[index_nf]))
    index_nf = min(range(len(n_f)), key=n_f.__getitem__)
    # print('\nThe minimum safety factor is: %f | at %s\n' %(min(n_f),fig_title[index_nf]))
    #--------------------------------------------------------------------------#
    # Get nodal static results
    filename = n_file
    n_full_name = os.path.join(current_path,'Job_results',DOE_results_folder,filename)

    [time_table,U_table_max,U_table_min,U_table_avg,time_norm] = text_read_nodal( n_full_name )

    U = np.array([])
    # (REMOVE NORMALIZED TIME LAST ROW)
    for n in range(len(time_table)-2): # iterate over number of sets
        # U = np.append(U,U_table_max[n][-1]) # get U at last time frame
        U = np.append(U,U_table_avg[n][-1]) # get U at last time frame
        
    return s,N,n_f,U

#==============================================================================#
# %% PLOT AM PROCESS SPATIAL DATA
def plot_AM_process_line_data(index,current_path,plot_true):
    import pickle
    import matplotlib.pyplot as plt
    import os
    import numpy as np
    #--------------------------------------------------------------------------#
    # PLOT NODAL LINE PATH RESULTS
    # figure commands

    filename = "%i_hline_out_n_file.pkl" %(index)
    nodal_line_full_name = os.path.join(current_path,'Job_results','Results_log',filename)
    resultsfile=open(nodal_line_full_name,'rb')

    # Get nodal results at line path location

    picklefile = pickle._Unpickler(resultsfile)
    picklefile.encoding = 'latin1'
    x_p = picklefile.load()
    y_p = picklefile.load()
    z_p = picklefile.load()
    angle_p = picklefile.load()
    n_label_p = picklefile.load()
    time_p = picklefile.load()
    U1_p = picklefile.load()
    U2_p = picklefile.load()
    U3_p = picklefile.load()
    U_p = picklefile.load()

    resultsfile.close()

    # Initialize variables and obtain distance plotting vector
    x_prev = x_p[0]; y_prev = y_p[0]; z_prev = z_p[0]; d_vec = []; d_prev = 0
    for x,y,z in zip(x_p,y_p,z_p):
        dx = x - x_prev; dy = y - y_prev; dz = z - z_prev # obtain consecutive distance
        d_i = np.linalg.norm([dx,dy,dz]) # obtain magnitude
        d_prev = d_prev + d_i
        d_vec += [d_prev] # iterate distance vector
        x_prev = x; y_prev = y; z_prev = z

    # Plot displacement data
    U_p_line = []
    for U_p_e in U_p:
        U_p_line += [U_p_e[-1]] # extract last time entry for data type
    
    if plot_true:
        plt.figure()
        plt.plot(angle_p, U_p_line, '-')
        # fig options
        plt.xlabel('Angle ($^o$)') # x Axis labels
        plt.ylabel('Displacement magnitude (mm)') # y Axis label
        plt.tight_layout()
    
        fig_name = '%i_displacement_profile.pdf' %(int(index))
        fige_file_name = os.path.join(current_path,'Job_results','Results_log',fig_name)
        plt.savefig(fige_file_name, bbox_inches='tight')
    
        # Plot displacement timeseries data
        plt.figure()
        plt.plot(time_p, U_p[int(len(U_p)/2)], '-')
        # fig options
        plt.xlabel('Time (s)') # x Axis labels
        plt.ylabel('Displacement magnitude (mm)') # y Axis label
        plt.tight_layout()
    
        fig_name = '%i_displacement_timeseries.pdf' %(int(index))
        fige_file_name = os.path.join(current_path,'Job_results','Results_log',fig_name)
        plt.savefig(fige_file_name, bbox_inches='tight')

    #--------------------------------------------------------------------------#
    # PLOT ELEMENT LINE PATH RESULTS

    # Get elemental results at line path location

    filename = "%i_hline_out_e_file.pkl" %(index)
    element_line_full_name = os.path.join(current_path,'Job_results','Results_log',filename)
    resultsfile=open(element_line_full_name,'rb')

    picklefile = pickle._Unpickler(resultsfile)
    picklefile.encoding = 'latin1'

    x_p = picklefile.load()
    y_p = picklefile.load()
    z_p = picklefile.load()
    angle_p = picklefile.load()
    e_label_p = picklefile.load()
    time_p = picklefile.load()
    stress_p = picklefile.load()
    temperature_p = picklefile.load()
    SP1_p = picklefile.load()
    SP2_p = picklefile.load()
    SP3_p = picklefile.load()
    S11_p = picklefile.load()
    S22_p = picklefile.load()
    S33_p = picklefile.load()
    S12_p = picklefile.load()
    S13_p = picklefile.load()
    S23_p = picklefile.load()
    Press_p = picklefile.load()
    Press_hoop_p = picklefile.load()
    stress_hoop_p = picklefile.load()

    resultsfile.close()

    # Initialize variables and obtain distance plotting vector
    x_prev = x_p[0]; y_prev = y_p[0]; z_prev = z_p[0]; d_vec = []; d_prev = 0
    for x,y,z in zip(x_p,y_p,z_p):
        dx = x - x_prev; dy = y - y_prev; dz = z - z_prev # obtain consecutive distance
        d_i = np.linalg.norm([dx,dy,dz]) # obtain magnitude
        d_prev = d_prev + d_i
        d_vec += [d_prev] # iterate distance vector
        x_prev = x; y_prev = y; z_prev = z

    # Plot stress data
    S_line = []; SP1_line = []; SP2_line = []; SP3_line = []; pressure_line = []
    S_hoop_line= []; pressure_hoop_line = []
    for S_e,SP1_e,SP2_e,SP3_e,S11_e,S22_e,S33_e,S12_e,S13_e,S23_e,Press_e,S_hoop_e,Press_hoop_e in zip(stress_p,SP1_p,SP2_p,SP3_p,S11_p,S22_p,S33_p,S12_p,S13_p,S23_p,Press_p,stress_hoop_p,Press_hoop_p):
        S_line += [S_e[-1]] # extract last time entry for data type
        SP1_line += [SP1_e[-1]] # extract last time entry for data type
        SP2_line += [SP2_e[-1]] # extract last time entry for data type
        SP3_line += [SP3_e[-1]] # extract last time entry for data type
        pressure_line += [Press_e[-1]]
        S_hoop_line += [S_hoop_e[-1]]
        pressure_hoop_line += [Press_hoop_e[-1]]
        
    if plot_true:
        plt.figure()
        plt.plot(angle_p, S_line, '-b', label='$\sigma_{mises}$')
        plt.plot(angle_p, SP1_line, '-r', label='$\sigma_1$')
        plt.plot(angle_p, SP3_line, '-g', label='$\sigma_3$')
        plt.legend(loc = 1)
        # fig options
        plt.xlabel('Angle ($^o$)') # x Axis labels
        plt.ylabel('Stress (MPa)') # y Axis label
        plt.tight_layout()
    
        fig_name = '%i_stress_profile.pdf' %(int(index))
        fige_file_name = os.path.join(current_path,'Job_results','Results_log',fig_name)
        plt.savefig(fige_file_name, bbox_inches='tight')
    
        plt.figure()
        plt.plot(angle_p, pressure_line, '-k', label='$Pressure$')
        # fig options
        plt.xlabel('Angle ($^o$)') # x Axis labels
        plt.ylabel('Stress (MPa)') # y Axis label
        plt.tight_layout()
    
        fig_name = '%i_pressure_profile.pdf' %(int(index))
        fige_file_name = os.path.join(current_path,'Job_results','Results_log',fig_name)
        plt.savefig(fige_file_name, bbox_inches='tight')
    
        # Plot temperature timeseries data
        plt.figure()
        plt.plot(time_p, temperature_p[int(len(temperature_p)/2)], '-')
        # fig options
        plt.xlabel('Time (s)') # x Axis labels
        plt.ylabel('Temperature ($^oC$)') # y Axis label
        plt.tight_layout()
    
        fig_name = '%i_temperature_timeseries.pdf' %(int(index))
        fige_file_name = os.path.join(current_path,'Job_results','Results_log',fig_name)
        plt.savefig(fige_file_name, bbox_inches='tight')
    
        # Plot mises timeseries data
        plt.figure()
        plt.plot(time_p, stress_p[int(len(stress_p)/2)], '-')
        # fig options
        plt.xlabel('Time (s)') # x Axis labels
        plt.ylabel('Von mises stress (MPa)') # y Axis label
        plt.tight_layout()
    
        fig_name = '%i_stress_timeseries.pdf' %(int(index))
        fige_file_name = os.path.join(current_path,'Job_results','Results_log',fig_name)
        plt.savefig(fige_file_name, bbox_inches='tight')
    
    # plt.show()
    return U_p_line, S_line, S_hoop_line, pressure_hoop_line
   
#==============================================================================#
# %% PLOT LOAD CASE SPATIAL DATA
def plot_loadcase_line_data(index,filename_e,filename_n,DOE_folder,current_path,plot_true,S_hoop_res_line,press_hoop_res_line):
    import pickle, os
    import numpy as np
    import matplotlib.pyplot as plt
    #--------------------------------------------------------------------------#
    # PLOT NODAL LINE PATH RESULTS
    # figure commands
    
    nodal_line_full_name = os.path.join(current_path,'Job_results',DOE_folder,filename_n)
    resultsfile=open(nodal_line_full_name,'rb')
    
    # Get nodal results at line path location

    picklefile = pickle._Unpickler(resultsfile)
    picklefile.encoding = 'latin1'
    x_p = picklefile.load()
    y_p = picklefile.load()
    z_p = picklefile.load()
    angle_p = picklefile.load()
    n_label_p = picklefile.load()
    time_p = picklefile.load()
    U1_p = picklefile.load()
    U2_p = picklefile.load()
    U3_p = picklefile.load()
    U_p = picklefile.load()

    resultsfile.close()

    # Plot displacement data
    U_p_line = []
    for U_p_e in U_p:
        U_p_line += [U_p_e[-1]] # extract last time entry for data type
    
    if plot_true:
        plt.figure()
        plt.plot(angle_p, U_p_line, '-')
        # fig options
        plt.xlabel('Angle ($^o$)') # x Axis labels
        plt.ylabel('Displacement magnitude (mm)') # y Axis label
        plt.tight_layout()
    
        fig_name = '%i_displacement_profile.pdf' %(int(index))
        fig_file_name = os.path.join(current_path,'Job_results',DOE_folder,fig_name)
        plt.savefig(fig_file_name, bbox_inches='tight')

    #--------------------------------------------------------------------------#
    # PLOT ELEMENT LINE PATH RESULTS

    # Get elemental results at line path location

    element_line_full_name = os.path.join(current_path,'Job_results',DOE_folder,filename_e)
    resultsfile=open(element_line_full_name,'rb')
    
    picklefile = pickle._Unpickler(resultsfile)
    picklefile.encoding = 'latin1'

    x_p = picklefile.load()
    y_p = picklefile.load()
    z_p = picklefile.load()
    angle_p = picklefile.load()
    e_label_p = picklefile.load()
    time_p = picklefile.load()
    stress_p = picklefile.load()
    temperature_p = picklefile.load()
    SP1_p = picklefile.load()
    SP2_p = picklefile.load()
    SP3_p = picklefile.load()
    S11_p = picklefile.load()
    S22_p = picklefile.load()
    S33_p = picklefile.load()
    S12_p = picklefile.load()
    S13_p = picklefile.load()
    S23_p = picklefile.load()
    Press_p = picklefile.load()
    Press_hoop_p = picklefile.load()
    stress_hoop_p = picklefile.load()
    S_M_p = picklefile.load()
    S_A_p = picklefile.load()
    N_p = picklefile.load()
    n_f_p = picklefile.load()

    resultsfile.close()
    
    # Plot stress data
    S_line = []; S_M_line = []; S_A_line = []; pressure_line = []; n_f_line = []; N_line = []
    S_hoop_line= []; Press_hoop_line = []
    plot_data = zip(stress_p,S_M_p,S_A_p,Press_p,N_p,n_f_p,stress_hoop_p,Press_hoop_p)
    for S_e,S_M_e,S_A_e,Press_e,N_e,n_f_e,S_hoop_e,Press_hoop_e in plot_data:
        S_line += [S_e[-1]] # extract last time entry for data type
        S_M_line += [S_M_e[-1]] # extract last time entry for data type
        S_A_line += [S_A_e[-1]] # extract last time entry for data type
        pressure_line += [Press_e[-1]]
        S_hoop_line += [S_hoop_e[-1]]
        Press_hoop_line += [Press_hoop_e[-1]]
        #n_f_line += [n_f_e[-1]]
        #N_line += [N_e[-1]]
        
    [S_M_line, S_A_line, N_line, n_f_line] = fatigue_calculation( S_hoop_res_line,S_hoop_line,
        press_hoop_res_line,Press_hoop_line ) # get fatigue stresses
    
    if plot_true:
        plt.figure()
        # plt.plot(angle_p, S_line, '-b', label='$\sigma_{mises}$')
        plt.plot(angle_p, S_M_line, '-r', label='$\sigma_m$')
        plt.plot(angle_p, S_A_line, '-g', label='$\sigma_a$')
        plt.legend(loc = 1)
        # fig options
        plt.xlabel('Angle ($^o$)') # x Axis labels
        plt.ylabel('Stress (MPa)') # y Axis label
        plt.tight_layout()
    
        fig_name = '%i_stress_profile.pdf' %(int(index))
        fig_file_name = os.path.join(current_path,'Job_results',DOE_folder,fig_name)
        plt.savefig(fig_file_name, bbox_inches='tight')
    
        plt.figure()
        plt.plot(angle_p, pressure_line, '-k', label='$Pressure$')
        # fig options
        plt.xlabel('Angle ($^o$)') # x Axis labels
        plt.ylabel('Stress (MPa)') # y Axis label
        plt.tight_layout()
    
        fig_name = '%i_pressure_profile.pdf' %(int(index))
        fig_file_name = os.path.join(current_path,'Job_results',DOE_folder,fig_name)
        plt.savefig(fig_file_name, bbox_inches='tight')
        
        plt.figure()
        plt.plot(angle_p, n_f_line, '-k', label='$Safety Factor$')
        # fig options
        plt.xlabel('Angle ($^o$)') # x Axis labels
        plt.ylabel('Safety Factor') # y Axis label
        # plt.ylim((0,4.0))
        plt.tight_layout()
    
        fig_name = '%i_safety_factor_profile.pdf' %(int(index))
        fig_file_name = os.path.join(current_path,'Job_results',DOE_folder,fig_name)
        plt.savefig(fig_file_name, bbox_inches='tight')
    
    # plt.show()
    return S_line, N_line, n_f_line, U_p_line

#==============================================================================#
# %% POSTPROCESS DOE DATA TO GET CAPABILITY, RESILIANCE AND EXCESS
        
def Rspace_calculation(server,bounds_req,mu,Sigma,req_type,bounds,res,threshold,
                           new_LHS_MCI,LHS_MCI_file):
    
    from abaqus_postprocess import gridsamp
    import numpy as np
    import copy 
    from scipy import integrate
    from pyDOE import lhs
    import pickle
    import os

    lob = bounds[:,0]
    upb = bounds[:,1]
    
    lob_req = bounds_req[:,0]
    upb_req = bounds_req[:,1]

    # LHS distribution
    DOE_full_name = LHS_MCI_file +'.pkl'
    DOE_filepath = os.path.join(os.getcwd(),'Optimization_studies',DOE_full_name)

    if new_LHS_MCI: # generate new LHS for each Monte-Carlo Integration operation
        # LHS distribution
        dFF_lhs = lhs(len(lob), samples=res, criterion='center')
        # Sample the requirements space only
        dFF = scaling(dFF_lhs,lob_req,upb_req,2) # unscale latin hypercube points to req
        dFF_n = scaling(dFF,lob,upb,1) # scale requirement to full space
        
        # Sample the parameter space only
        dFF_Pspace = scaling(dFF_lhs,lob,upb,2) # unscale latin hypercube points to req
        dFF_n_Pspace = dFF_lhs # scale requirement to full space

        resultsfile=open(DOE_filepath,'wb')
        
        pickle.dump(dFF, resultsfile)
        pickle.dump(dFF_n, resultsfile)
        pickle.dump(dFF_Pspace, resultsfile)
        pickle.dump(dFF_n_Pspace, resultsfile)

        resultsfile.close()
        
    else:
        resultsfile=open(DOE_filepath,'rb')
        
        dFF = pickle.load(resultsfile)
        dFF_n = pickle.load(resultsfile)
        dFF_Pspace = pickle.load(resultsfile)
        dFF_n_Pspace = pickle.load(resultsfile)
    
        resultsfile.close()

    if req_type == "guassian":

        # #===================================================================#
        # # Compute excess

        # # 1,2,3 Sigma level contour
        # L = []
        # for n in range(3):
        #     # Pack X and Y into a single 3-dimensional array
        #     pos = np.empty((1,1) + (len(lob),))
        #     x_l = [ mu[0] + ((n+1) * np.sqrt(Sigma[0,0])), # evaluate at Sigma not Sigma^2
        #             mu[1]                                ,
        #             mu[2]                                ,
        #             mu[3]                                ]
            
        #     level_index = 0
        #     for value in x_l:
        #         pos[:, :, level_index] = value
        #         level_index += 1
                
        #     LN = multivariate_gaussian(pos, mu, Sigma)
        #     L += [LN]

        # # Evaluate multivariate guassian
        # pos = np.empty((res,1) + (len(lob),))
           
        # for i in range(len(lob)):
        #     X_norm = np.reshape(dFF_n_Pspace[:,i],(res,1))
        #     # Pack X1, X2 ... Xk into a single 3-dimensional array
        #     pos[:, :, i] = X_norm
         
        # Z = multivariate_gaussian(pos, mu, Sigma)
        # Z = np.reshape(Z, np.shape(dFF_n_Pspace)[0])
            
        # Z_req = copy.deepcopy(Z)
        
        # cond_requirement = (Z_req - L[-1]) >= 0

        # # Design space volume
        # R_volume = len(cond_requirement[cond_requirement])/np.shape(dFF_n_Pspace)[0]

        # Analytical volume of an ellipsoid
        UB_n = scaling(upb_req,lob,upb,1)
        LB_n = scaling(lob_req,lob,upb,1)
        
        # Design space volume
        R_volume = np.prod(UB_n - LB_n) * ((np.pi**2)/32)

    elif req_type == "uniform":
            
        UB_n = scaling(upb_req,lob,upb,1)
        LB_n = scaling(lob_req,lob,upb,1)
        
        # Design space volume
        R_volume = np.prod(UB_n - LB_n)

    # print('Requirement volume: %f' %(R_volume))

    return R_volume

def resiliance_calculation(server,bounds_req,mu,Sigma,req_type,bounds,res,threshold,
                           new_LHS_MCI,LHS_MCI_file):
    
    from abaqus_postprocess import gridsamp
    import numpy as np
    import copy 
    from scipy import integrate
    from pyDOE import lhs
    import pickle
    import os
    
    lob = bounds[:,0]
    upb = bounds[:,1]
    
    lob_req = bounds_req[:,0]
    upb_req = bounds_req[:,1]

    # LHS distribution
    DOE_full_name = LHS_MCI_file +'.pkl'
    DOE_filepath = os.path.join(os.getcwd(),'Optimization_studies',DOE_full_name)

    if new_LHS_MCI: # generate new LHS for each Monte-Carlo Integration operation
        # LHS distribution
        dFF_lhs = lhs(len(lob), samples=res, criterion='center')
        # Sample the requirements space only
        dFF = scaling(dFF_lhs,lob_req,upb_req,2) # unscale latin hypercube points to req
        dFF_n = scaling(dFF,lob,upb,1) # scale requirement to full space
        
        # Sample the parameter space only
        dFF_Pspace = scaling(dFF_lhs,lob,upb,2) # unscale latin hypercube points to req
        dFF_n_Pspace = dFF_lhs # scale requirement to full space

        resultsfile=open(DOE_filepath,'wb')
        
        pickle.dump(dFF, resultsfile)
        pickle.dump(dFF_n, resultsfile)
        pickle.dump(dFF_Pspace, resultsfile)
        pickle.dump(dFF_n_Pspace, resultsfile)

        resultsfile.close()
        
    else:
        resultsfile=open(DOE_filepath,'rb')
        
        dFF = pickle.load(resultsfile)
        dFF_n = pickle.load(resultsfile)
        dFF_Pspace = pickle.load(resultsfile)
        dFF_n_Pspace = pickle.load(resultsfile)
    
        resultsfile.close()

    if req_type == "guassian":
        
        [YX, std, ei, cdf] = server.sgtelib_server_predict(dFF_n)
        # print("================== RESILIANCE ====================")
        # print(np.min(dFF_n, axis=0))
        # print(min(np.min(YX, axis=0)))
        # print(np.max(dFF_n, axis=0))
        # print(min(np.max(YX, axis=0)))
        #===================================================================#
        # capability constraints
         
        YX_cstr = np.reshape(YX - threshold, np.shape(dFF_n)[0])
        
        # Evaluate multivariate guassian
        res_sq = np.ceil(res**(0.5*len(lob))).astype(int) # size of equivalent square matrix
        pos = np.empty((res,1) + (len(lob),))
           
        for i in range(len(lob)):
            X_norm = np.reshape(dFF_n[:,i],(res,1))
            # Pack X1, X2 ... Xk into a single 3-dimensional array
            pos[:, :, i] = X_norm
         
        Z = multivariate_gaussian(pos, mu, Sigma)
        Z = np.reshape(Z, np.shape(dFF_n)[0])
            
        Z_feasible = copy.deepcopy(Z)
        Z_feasible[YX_cstr < 0] = 0.0 # eliminate infeasible regions from MCS
 
        # Design space volume
        resiliance = np.sum(Z_feasible)/np.sum(Z)

    elif req_type == "uniform":
            
        [YX, std, ei, cdf] = server.sgtelib_server_predict(dFF_n)
        cond_req_feas = (YX - threshold) > 0

        # print("================== RESILIANCE ====================")
        # print(np.min(dFF_n, axis=0))
        # print(min(np.min(YX, axis=0)))
        # print(np.max(dFF_n, axis=0))
        # print(min(np.max(YX, axis=0)))
        # Design space volume
        resiliance = len(cond_req_feas[cond_req_feas])/np.shape(dFF)[0]
        
    return resiliance

def capability_calculation(server,bounds_req,bounds,res,threshold,
                           new_LHS_MCI,LHS_MCI_file):
    
    from abaqus_postprocess import gridsamp
    import numpy as np
    import copy 
    from scipy import integrate
    from pyDOE import lhs
    import pickle
    import os
    
    lob = bounds[:,0]
    upb = bounds[:,1]
    
    lob_req = bounds_req[:,0]
    upb_req = bounds_req[:,1]

    # LHS distribution
    DOE_full_name = LHS_MCI_file +'.pkl'
    DOE_filepath = os.path.join(os.getcwd(),'Optimization_studies',DOE_full_name)

    if new_LHS_MCI: # generate new LHS for each Monte-Carlo Integration operation
        # LHS distribution
        dFF_lhs = lhs(len(lob), samples=res, criterion='center')
        # Sample the requirements space only
        dFF = scaling(dFF_lhs,lob_req,upb_req,2) # unscale latin hypercube points to req
        dFF_n = scaling(dFF,lob,upb,1) # scale requirement to full space
        
        # Sample the parameter space only
        dFF_Pspace = scaling(dFF_lhs,lob,upb,2) # unscale latin hypercube points to req
        dFF_n_Pspace = dFF_lhs # scale requirement to full space

        resultsfile=open(DOE_filepath,'wb')
        
        pickle.dump(dFF, resultsfile)
        pickle.dump(dFF_n, resultsfile)
        pickle.dump(dFF_Pspace, resultsfile)
        pickle.dump(dFF_n_Pspace, resultsfile)

        resultsfile.close()
        
    else:
        resultsfile=open(DOE_filepath,'rb')
        
        dFF = pickle.load(resultsfile)
        dFF_n = pickle.load(resultsfile)
        dFF_Pspace = pickle.load(resultsfile)
        dFF_n_Pspace = pickle.load(resultsfile)
    
        resultsfile.close()

    [YX, std, ei, cdf] = server.sgtelib_server_predict(dFF_n_Pspace)
    
    # print("================== CAPABILITY ====================")
    # print(min(np.min(YX, axis=0)))
    # print(min(np.max(YX, axis=0)))

    cond_req_feas = (YX - threshold) > 0
    
    # Design space volume
    capability = len(cond_req_feas[cond_req_feas])/np.shape(dFF_Pspace)[0]
        
    return capability

def integrand_multivariate_gaussian(*arg):
    import numpy as np
    """Return the multivariate Gaussian distribution on array pos.
    """
    # Mean vector and covariance matrix
    # mu = np.array([0., 1.])
    # Sigma = np.array([[ 1. , -0.5], [-0.5,  1.5]])
    
    x = arg[0:-4] 
    
    mu = arg[-4] 
    Sigma = arg[-3]
    threshold = arg[-2]
    server = arg[-1]
    
    # Pack X and Y into a single 3-dimensional array
    pos = np.empty((1,1) + (len(x),))
    X_eval = np.ones((1,len(x)))
    
    i = 0
    for value in x:
        X_eval[0,i] = value
        pos[:, :, i] = value
        i += 1
    
    #===========================================================================
    # [YX, std, ei, cdf] = server.sgtelib_server_predict(X_eval);
    #===========================================================================
    YX = 1000
    if YX - threshold < 0:
        Z = 0.0
    else:
        Z = multivariate_gaussian(pos, mu, Sigma)

    return Z

def multivariate_gaussian(pos, mu, Sigma):
    
    import numpy as np
    """Return the multivariate Gaussian distribution on array pos.

    pos is an array constructed by packing the meshed arrays of variables
    x_1, x_2, x_3, ..., x_k into its _last_ dimension.

    """

    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

    Z = np.exp(-fac / 2)

    return Z

#==============================================================================#
# %% POSTPROCESS DOE DATA
def postprocess_DOE(index,base_name,current_path,DOE_folder,bounds,variable_lbls_pc,
                    permutation_index,concept,process_DOE,S_hoop_res_line,press_hoop_res_line,
                    metric,plot_para_coords):
    
    from abaqus_postprocess import parallel_plots, define_SGTE_model
    from SGTE_library import SGTE_server
    import pickle, os
    import numpy as np
    
    if process_DOE:
        # Load DOE inputs and outputs
        DOE_filename = '%i_%s_1.npy' %(index,base_name[0])
        DOE_inputs = np.load(os.path.join(current_path,'Job_results','Results_log',DOE_filename))
        # DOE_inputs = [DOE_inputs[7]]
        
        index_DOE = 0; outputs = np.array([]); backup_results = []
        print("+================================================================+")
        print("ANALYSIS START")
        for point in DOE_inputs: # conduct DOE         
            index_DOE += 1
            
            filename_n = "%i_%i_%s_n_file.pkl" %(index,index_DOE,base_name[1])
            filename_e = "%i_%i_%s_e_file.pkl" %(index,index_DOE,base_name[1])
            filename_n_log = "%i_%i_%s_n_file.log" %(index,index_DOE,base_name[1])
            filename_e_log = "%i_%i_%s_e_file.log" %(index,index_DOE,base_name[1])
            
            [S_line, N_line, n_f_line, U_p_line] = plot_loadcase_line_data(index,filename_e,filename_n,DOE_folder,current_path,False,S_hoop_res_line,press_hoop_res_line)
             
            results = [max(S_line), min(N_line), min(n_f_line), max(U_p_line)]
            backup_results += results
            
            result = results[metric[1]] # choose type of output [s,N,n_f,U]
            outputs = np.append(outputs,[result])
            
            # outputs = np.append(outputs,[sum(n_f_line)/len(n_f_line)])
        
        outputs = np.reshape(outputs,(len(outputs),1)) # rearrange into a column vector (FOR SGTELIB)
    
    else:
        # Import DOE outputs from pickle file
        backup_file = '%i_%s_DOE_out.pkl' %(index,base_name[1])
        backup_filepath = os.path.join(os.getcwd(),'Job_results','Results_log',backup_file) 
        backupfile=open(backup_filepath,'rb')
        backup_results = pickle.load(backupfile)
        DOE_inputs = pickle.load(backupfile)
        concept = pickle.load(backupfile)
        permutation_index = pickle.load(backupfile)
        backupfile.close()
        
        print(backup_filepath)

        # Load results as outputs
        index_DOE = 0; outputs = np.array([])
        print("+================================================================+")
        print("ANALYSIS START")
        for results in backup_results: # conduct DOE         
            index_DOE += 1
            [s,N,n_f,U] = results
            result = results[metric[1]][0] # choose type of output [s,N,n_f,U]
            outputs = np.append(outputs,[result])
            
        outputs = np.reshape(outputs,(len(outputs),1)) # rearrange into a column vector (FOR SGTELIB)
        
    #======================== SURROGATE META MODEL ============================#
    # %% SURROGATE modeling
    lob = bounds[:,0]
    upb = bounds[:,1]
    
    Y = outputs; S_n = scaling(DOE_inputs, lob, upb, 1)
    # fitting_names = ['KRIGING','LOWESS','KS','RBF','PRS','ENSEMBLE'];
    # fit_type = 1; run_type = 2; # use pre-optimized hyperparameters
    fit_type = 0; run_type = 1 # optimize all hyperparameters
    model,sgt_file = define_SGTE_model(fit_type,run_type)
    server = SGTE_server(model)
    server.sgtelib_server_start()
    server.sgtelib_server_ping()
    server.sgtelib_server_newdata(S_n,Y)    
    #===========================================================================
    # M = server.sgtelib_server_metric('RMSECV')
    # print('RMSECV Metric: %f' %(M[0]))
    #===========================================================================  
    
    #===========================================================================
    # SAVE DOE results as parallel coordinates
    if plot_para_coords:
        output_lbls = ['n_safety']
        parallel_file = '%i_parallel_plot_%s.html' %(index,base_name[1])
        parallel_filepath = os.path.join(current_path,'Job_results','Results_log',parallel_file)
        parallel_plots(DOE_inputs,outputs,variable_lbls_pc,output_lbls,parallel_filepath)
    #===========================================================================  

    return server, DOE_inputs, outputs

def process_requirements(index,base_name,current_path,bounds,mu,Sigma,req_type,variable_lbls,
                         threshold,LHS_MCI_file,req_index,server,DOE_inputs,outputs,
                         plt,resolution=20,plot_R_space=False,new_LHS_MCI=False,compute_margins=True,
                         plot_index=4,plot_2D=False):
    
    from abaqus_postprocess import hyperplane_SGTE_vis_norm
    from matplotlib import rc
    import os
    import numpy as np
    import copy
    
    rc('text', usetex=True)
    metric_label = ['Stress - $\sigma_{mises}$ (MPa)','Number of cycles - $N$','Safety factor - $n_f$', 'Displacement - $U$ (mm)']
    
    #===========================================================================
    # Compute resiliance
    lob_req = mu - (3 * np.sqrt(Sigma))
    upb_req = mu + (3 * np.sqrt(Sigma))
    
    bounds_uniform_lob = lob_req # Sigma^2
    bounds_uniform_upb = upb_req # Sigma^2
    
    bounds_uniform = np.array( [bounds_uniform_lob, bounds_uniform_upb] )
    bounds_uniform = bounds_uniform.T
    # unscale requirement bounds for plotting and resiliance calculation
    bounds_req_lob = scaling(bounds_uniform_lob,bounds[:,0],bounds[:,1],2)
    bounds_req_upb = scaling(bounds_uniform_upb,bounds[:,0],bounds[:,1],2)
    
    check = bounds_req_lob < bounds[:,0]
    bounds_req_lob[check] = copy.deepcopy(bounds[check,0])
    check = bounds_req_upb > bounds[:,1]
    bounds_req_upb[check] = copy.deepcopy(bounds[check,1])
    
    bounds_req = np.array( [bounds_req_lob, bounds_req_upb] )
    bounds_req = bounds_req.T
    #===========================================================================

    Sigma = np.diag(Sigma)

    resiliance = resiliance_calculation(server,bounds_req,mu,Sigma,req_type, 
                                        bounds,resolution,threshold,new_LHS_MCI, 
                                        LHS_MCI_file)

    if compute_margins:
        R_volume = Rspace_calculation(server,bounds_req,mu,Sigma,req_type,bounds,resolution,threshold,
                                    new_LHS_MCI,LHS_MCI_file)

        capability = capability_calculation(server,bounds_req,bounds,resolution,threshold,
                                            new_LHS_MCI,LHS_MCI_file)

        Buffer = R_volume * resiliance
        Excess = capability - Buffer

    #===========================================================================
    # Plot 2D projections
    if plot_R_space:
        nominal = [0.5]*len(variable_lbls); nn = 80
        fig = plt.figure()  # create a figure object
        if plot_2D:
            fig_2D = plt.figure()  # create a figure object
        else:
            fig_2D = None

        hyperplane_SGTE_vis_norm(server,DOE_inputs,bounds,bounds_req,LHS_MCI_file,mu,Sigma,req_type,variable_lbls,
                                 nominal,threshold,outputs,nn,fig,plt,plot_index=plot_index,plot_2D=plot_2D,fig_2D=fig_2D)
        
        fig_name = '%i_req_%i_%s_RS_pi_%i.pdf' %(index,req_index,base_name[1],plot_index)
        fig_file_name = os.path.join(current_path,'Job_results','Results_log',fig_name)
        fig.savefig(fig_file_name, bbox_inches='tight')
        plt.close(fig)

        if plot_2D:
            fig_name = '%i_req_%i_%s_RS_2D_pi_%i.pdf' %(index,req_index,base_name[1],plot_index)
            fig_file_name = os.path.join(current_path,'Job_results','Results_log',fig_name)
            fig_2D.savefig(fig_file_name, bbox_inches='tight')
            plt.close(fig_2D)
    
    if compute_margins:
        return resiliance, R_volume, capability, Buffer, Excess
    else:
        return resiliance 

def postprocess_nominal(index,base_name,current_path,permutation_index,concept,process_DOE,S_hoop_res_line,press_hoop_res_line):
    from abaqus_postprocess import hyperplane_SGTE_vis_norm, parallel_plots
    import pickle, os
    import numpy as np
    
    outputs = np.array([])
    print("+================================================================+")
    print("ANALYSIS START")        
    
    if process_DOE:
        filename_n = "%i_%s_n_file.pkl" %(index,base_name[1])
        filename_e = "%i_%s_e_file.pkl" %(index,base_name[1])
        
        [ S_line, N_line, n_f_line, U_p_line ] = plot_loadcase_line_data(index,filename_e,filename_n,'Results_log',current_path,False,S_hoop_res_line,press_hoop_res_line)
        outputs = [max(S_line), min(N_line), min(n_f_line), max(U_p_line)]
    else:
        filename_n_log = "%i_%s_n_file.log" %(index,base_name[1])
        filename_e_log = "%i_%s_e_file.log" %(index,base_name[1])
        
        results = get_loadcase_results(filename_e_log,filename_n_log,current_path,'Results_log')
        
        for result in results: # choose type of output [S,N,n_f,U]
            # location [outer casing, outer weld roots, inner weld roots]
            result_location = result[0] # choose location of output
            outputs = np.append(outputs,[result_location])
        
    print("ANALYSIS COMPLETE")
    print("+================================================================+")
    
    return outputs

#==============================================================================#
# %% MAIN FILE
def DED_blackbox_evaluation(concept, permutation_index, run_base, run_nominal,
                            ax_pos, st_thick, st_width, laser_power, scanning_speed,
                            power_density, layer_length, layer_width, layer_thick,
                            n_layers, n_deposit, mesh_size, mesh_AM_size,
                            melting_T, b_thick, process_DOE_requirements, sampling, 
                            new_LHS_MCI, index, plot_index=4,plot_2D=False,req_indices=None):

    import os, time
    import numpy as np
    from abaqus_postprocess import gridsamp, hyperplane_SGTE_vis_norm, parallel_plots
    import matplotlib.pyplot as plt
    import itertools
    from pyDOE import lhs
    import pickle
    
    # plt.rc('text', usetex=True)

    current_path = os.getcwd() # Working directory of file
    start_time = time.time()
    
    design_variables = []; pi_file = []
    for p_i in permutation_index:
        design_variables += [p_i]
        if p_i != -1:
            pi_file += [p_i]
    
    suffix = ''.join(map(str,pi_file))
    destination = os.path.join(current_path,"Job_results","Results_log","%s_DED_static_job.odb" %(suffix))
    
    #--------------------------------------------------------------------------#
    # Get AM process results
    
    [s_res,s_res,s_11_res,s_33_res,U_res] = get_AM_results(index, current_path)
    [U_p_res_line,S_res_line,S_hoop_res_line,press_hoop_res_line] = plot_AM_process_line_data(index,current_path,False)
         
    process_DOE = False
    # Get Loadcase results
    #--------------------------------------------------------------------------#
    # Baseline case
    if run_base == 1:
        # IP loadcase
        # choose type of output [S,N,n_f,U]
        # location [outer casing, outer weld roots, inner weld roots]
        base_name = ['DOE_ip_inputs','base_static_out']
        S_hoop_res_line_empty = [0] * len(S_hoop_res_line)
        press_hoop_res_line_empty = [0] * len(press_hoop_res_line)
        base_ip_results = postprocess_nominal(index,base_name,current_path,design_variables,concept,process_DOE,S_hoop_res_line_empty,press_hoop_res_line_empty)
        
        # Thermal loadcase
        # choose type of output [S,N,n_f,U]
        # location [outer casing, outer weld roots, inner weld roots]
        base_name = ['DOE_th_inputs','base_thermal_out']
        base_th_results = postprocess_nominal(index,base_name,current_path,design_variables,concept,process_DOE,S_hoop_res_line_empty,press_hoop_res_line_empty)
    
        print('base safety factor pressure: %f' %(base_ip_results[2]))
        print('base safety factor thermal: %f' %(base_th_results[2]))
        print('base N cycles pressure: %f' %(base_ip_results[1]))
        print('base N cycles thermal: %f' %(base_th_results[1]))
        
    else:
        base_ip_results = [0.0, 0.0, 0.0, 0.0]
        base_th_results = [0.0, 0.0, 0.0, 0.0]
    
    # Nominal case
    if run_nominal == 1:
        # IP loadcase
        # choose type of output [S,N,n_f,U]
        # location [outer casing, outer weld roots, inner weld roots]
        base_name = ['DOE_ip_inputs','static_out']
        ip_results = postprocess_nominal(index,base_name,current_path,design_variables,concept,process_DOE,S_hoop_res_line,press_hoop_res_line)
        
        # Thermal loadcase
        # choose type of output [S,N,n_f,U]
        # location [outer casing, outer weld roots, inner weld roots]
        base_name = ['DOE_th_inputs','thermal_out']
        th_results = postprocess_nominal(index,base_name,current_path,design_variables,concept,process_DOE,S_hoop_res_line,press_hoop_res_line)
    
        print('safety factor pressure: %f' %(ip_results[2]))
        print('safety factor thermal: %f' %(th_results[2]))
        print('N cycles pressure: %f' %(ip_results[1]))
        print('N cycles thermal: %f' %(th_results[1]))
        
    else:
        ip_results = [0.0, 0.0, 0.0, 0.0]
        th_results = [0.0, 0.0, 0.0, 0.0]
        
    process_IP = False; LHS_MCI_file = 'LHS_MCI_IP'
    #--------------------------------------------------------------------------#
    # IP loadcase
    bounds_ip = np.array( [[-1.0, 1.0]] )
    bounds_req = np.array( [[-0.25, 0.25]] ) # unused
    # bounds_req = bounds_ip
    
    process_DOE = False
    resolution = 100 # sampling resolution for capability calculation (must be a square number)!
    # threshold = 4.0 # cutoff threshold for capability calculation
    threshold = 2.8 # cutoff threshold for capability calculation
    
    DOE_folder = 'IP_DOE_results'; base_name = ['DOE_ip_inputs','static_out']
    variable_lbls_pc = ['IP_V']
    variable_lbls = ['$P_{load}$ (MPa)']
     
    # choose type of output [s,N,n_f,U]
    # location [outer casing, outer weld roots, inner weld roots]
    metric = [0,2]
    
    if process_IP:
        # Train surrogate model for use in subsequent capability calculation and plotting
        [server,DOE_inputs,outputs] = postprocess_DOE(index,base_name,current_path,DOE_folder,bounds_ip,
                                              variable_lbls_pc,design_variables,concept,process_DOE,
                                              S_hoop_res_line,press_hoop_res_line,metric,True)
        
        
    # Sample the requierments space
    print("------------------------- %s -------------------------\n" %('RESILIANCE_IP'))
    #===========================================================================
    # IP loadcase guassian parameters (nominal values)
    # nominal values
    req_type_1 = "uniform"
    req_type_2 = "guassian"
    mu_nominal = np.array([0.5])
    Sigma_nominal = np.array([ (0.125/3)**2 ])
    
    req_list = [[req_type_1, req_type_2], [mu_nominal], [Sigma_nominal] ]
    req_combinations = list(itertools.product(*req_list)) 

    req_index = 0; resiliance_ip_vec_nominal = []; R_volume_ip_vec_nominal = []; capability_ip_vec_nominal = []
    buffer_ip_vec_nominal = []; excess_ip_vec_nominal = []
    for req in req_combinations: # iterate over all combinations of requirements
        
        req_index += 1
        [req_type, mu, Sigma] = req


        if process_IP:
            resiliance_ip, R_volume_ip, capability_ip, buffer_ip, excess_ip = process_requirements(
                index,['DOE_ip_inputs','static_out_nominal'],
                current_path,bounds_ip,
                mu,Sigma,req_type,variable_lbls,threshold,LHS_MCI_file,
                req_index,server,DOE_inputs,outputs,plt,resolution=resolution,
                plot_R_space=True,new_LHS_MCI=new_LHS_MCI)
        else:
            resiliance_ip = 0.0; R_volume_ip = 0.0; capability_ip = 0.0; buffer_ip = 0.0; excess_ip = 0.0
        
        resiliance_ip_vec_nominal += [resiliance_ip]
        R_volume_ip_vec_nominal += [R_volume_ip]  
        capability_ip_vec_nominal += [capability_ip]
        buffer_ip_vec_nominal += [buffer_ip]
        excess_ip_vec_nominal += [excess_ip]
        print('Nominal resiliance against pressure loads: %f' %(resiliance_ip))
        print('Nominal requirement volume: %f' %(R_volume_ip))
        print('Nominal capability: %f' %(capability_ip))
        print('Nominal buffer: %f' %(buffer_ip))
        print('Nominal excess: %f' %(excess_ip))
    
    # plt.show()
    
    #===========================================================================
    # IP loadcase guassian parameters (DOE values)
    DOE_full_name = 'req_distribution_IP_LHS_data'+'.pkl'
    DOE_filepath = os.path.join(current_path,'Optimization_studies',DOE_full_name)
    
    resultsfile=open(DOE_filepath,'rb')
    
    lob_req = pickle.load(resultsfile)
    upb_req = pickle.load(resultsfile)
    points = pickle.load(resultsfile)
    points_us = pickle.load(resultsfile)

    resultsfile.close()

    req_type_1 = 'uniform'
    req_type_2 = 'guassian'
    
    #===========================================================================
    # IP loadcase guassian parameters full factorial
    if sampling == 'fullfact':
    
        mu_1 = lob_req[:1]
        mu_2 = upb_req[:1]
        
        Sigma_1 = lob_req[1::] # Sigma^2
        Sigma_2 = upb_req[1::] # Sigma^2
    
        # linearly interpolate between two vectors
        from scipy.interpolate import interp1d
    
        linfit = interp1d([1,5], np.vstack([mu_1, mu_2]), axis=0)
        mus = list(linfit([1,2,3,4,5]))
        linfit = interp1d([1,5], np.vstack([Sigma_1, Sigma_2]), axis=0)
        Sigmas = list(linfit([1,2,3,4,5]))

        # # USE THIS IF YOU WANT TO PLOT A FEW CASES AS AN EXAMPLE
        # linfit = interp1d([1,5], np.vstack([mu_1, mu_2]), axis=0)
        # mus = list(linfit([2,4]))
        # linfit = interp1d([1,5], np.vstack([Sigma_1, Sigma_2]), axis=0)
        # Sigmas = list(linfit([2,5]))

        Sigma_2s = []
        for Sigma in Sigmas:
            Sigma_2 = (Sigma/3)**2
            Sigma_2s += [Sigma_2]
    
        req_list = [[req_type_1, req_type_2], mus, Sigma_2s ]
        req_combinations = list(itertools.product(*req_list)) 
    
    #===========================================================================
    # IP loadcase guassian parameters LHS sampling
    elif sampling == 'LHS':
    
        req_combinations = []
        for point in points_us:
            
            for req_type in [req_type_1,req_type_2]:
                
                mu_lhs = point[0:1]
                Sigma_lhs = point[1::]
                Sigma_lhs_2 = (Sigma_lhs/3)**2
                
                line = [req_type,mu_lhs,Sigma_lhs_2]
                req_combinations += [line]
    #===========================================================================

    
    if process_DOE_requirements:

        req_index = 0; resiliance_ip_vec = []; buffer_ip_vec = []; excess_ip_vec = []
        for req in req_combinations: # iterate over all combinations of requirements
            
            req_index += 1
            [req_type, mu, Sigma] = req
            
            if process_IP:
                resiliance_ip,_,_,buffer_ip, excess_ip = process_requirements(
                    index,base_name,current_path,bounds_ip,
                    mu,Sigma,req_type,variable_lbls,threshold,
                    resolution,False,new_LHS_MCI,LHS_MCI_file,
                    req_index,server,DOE_inputs,outputs,plt,resolution=resolution,
                    plot_R_space=False,new_LHS_MCI=new_LHS_MCI)
            else:
                resiliance_ip = 0.0; buffer_ip = 0.0; excess_ip = 0.0
            
            resiliance_ip_vec += [resiliance_ip]
            buffer_ip_vec += [buffer_ip]
            excess_ip_vec += [excess_ip]
            print('(IP) Resiliance: %f, Buffer: %f, Excess: %f' %(resiliance_ip,buffer_ip,excess_ip))
            
        if process_IP:
            server.sgtelib_server_stop()
            server.server_print(server.server_process) # print console output
    
        # plt.show()
    else:
        resiliance_ip_vec = [0.0] * len(req_combinations)
        buffer_ip_vec = [0.0] * len(req_combinations)
        excess_ip_vec = [0.0] * len(req_combinations)
        
    print("ANALYSIS COMPLETE")
    print("+================================================================+") 
       
    process_TH = True; LHS_MCI_file = 'LHS_MCI_TH'
    #--------------------------------------------------------------------------#
    # Thermal loadcase
    bounds_th = np.array( [[-100, 100],
                           [-100, 100],
                           [-100, 100],
                           [-100, 100]] )
    
    bounds_req = np.array( [[ -100  , 50 ],
                            [-25 , 25 ],
                            [-25 , 25 ],
                            [-50 , 100 ]] ) # unused
    
    process_DOE = False
    # resolution = 1296 # sampling resolution for capability calculation (must be a square number)!
    resolution = 10000 # sampling resolution for capability calculation (must be a square number)!
    # resolution = 50 # sampling resolution for capability calculation (must be a square number)!
    # threshold = 100000 # cutoff threshold for capability calculation
    # threshold = 4.0 # cutoff threshold for capability calculation
    threshold = 2.8 # cutoff threshold for capability calculation
    
    DOE_folder = 'Thermal_DOE_results'; base_name = ['DOE_th_inputs','thermal_out']
    variable_lbls_pc = ['T1','T2','T3','T4']
    variable_lbls = ['$T_1$ ($^o$C)','$T_2$ ($^o$C)','$T_3$ ($^o$C)','$T_4$ ($^o$C)']
    
    # choose type of output [s,N,n_f,U]
    # location [outer casing, outer weld roots, inner weld roots]
    metric = [0,2]
    
    if process_TH:
        # Train surrogate model for use in subsequent capability calculation and plotting
        [server,DOE_inputs,outputs] = postprocess_DOE(index,base_name,current_path,DOE_folder,bounds_th,
                                              variable_lbls_pc,design_variables,concept,process_DOE,
                                              S_hoop_res_line,press_hoop_res_line,metric,True)
        
    # Sample the requierments space
    print("------------------------- %s -------------------------\n" %('RESILIANCE_TH'))
    #===========================================================================
    # Thermal loadcase guassian parameters (nominal values)
    req_type_1 = 'uniform'
    req_type_2 = 'guassian'
    mu_nominal = np.array([0.375, 0.5, 0.5, 0.625])
    Sigma_nominal = np.array([(0.375/3)**2, (0.125/3)**2, (0.125/3)**2, (0.375/3)**2]) # sigma^2
    
    req_list = [[req_type_1, req_type_2], [mu_nominal], [Sigma_nominal] ]
    req_combinations = list(itertools.product(*req_list)) 
    
    req_index = 0; resiliance_th_vec_nominal = []; R_volume_th_vec_nominal = []; capability_th_vec_nominal = []
    buffer_th_vec_nominal = []; excess_th_vec_nominal = []
    for req in req_combinations: # iterate over all combinations of requirements
        
        req_index += 1
        [req_type, mu, Sigma] = req
        
        if process_TH:
            resiliance_th,R_volume_th,capability_th,buffer_th, excess_th = process_requirements(
                index,['DOE_th_inputs','thermal_out_nominal'],
                current_path,bounds_th,
                mu,Sigma,req_type,variable_lbls,threshold,
                LHS_MCI_file,req_index,server,DOE_inputs,outputs,plt,resolution=resolution,
                plot_R_space=True,new_LHS_MCI=new_LHS_MCI,plot_index=plot_index,
                plot_2D=plot_2D)
        else:
            resiliance_th = 0.0; R_volume_th = 0.0; capability_th = 0.0; buffer_th = 0.0; excess_th = 0.0

        resiliance_th_vec_nominal += [resiliance_th]
        R_volume_th_vec_nominal += [R_volume_th]  
        capability_th_vec_nominal += [capability_th]
        buffer_th_vec_nominal += [buffer_th]
        excess_th_vec_nominal += [excess_th]
        print('Nominal resiliance against thermal loads: %f' %(resiliance_th))
        print('Nominal requirement volume: %f' %(R_volume_th))
        print('Nominal capability: %f' %(capability_th))
        print('Nominal buffer: %f' %(buffer_th))
        print('Nominal excess: %f' %(excess_th))
    
    # plt.show()
    #===========================================================================
    # Thermal loadcase guassian parameters (DOE values)
    DOE_full_name = 'req_distribution_TH_LHS_data'+'.pkl'
    DOE_filepath = os.path.join(current_path,'Optimization_studies',DOE_full_name)
    
    resultsfile=open(DOE_filepath,'rb')
    
    lob_req = pickle.load(resultsfile)
    upb_req = pickle.load(resultsfile)
    points = pickle.load(resultsfile)
    points_us = pickle.load(resultsfile)

    resultsfile.close()

    req_type_1 = 'uniform'
    req_type_2 = 'guassian'

    #===========================================================================
    # Thermal loadcase guassian parameters full factorial
    if sampling == 'fullfact':
    
        mu_1 = lob_req[:4]
        mu_2 = upb_req[:4]
        
        Sigma_1 = lob_req[4::] # Sigma^2
        Sigma_2 = upb_req[4::] # Sigma^2
    
        # # linearly interpolate between two vectors
        from scipy.interpolate import interp1d
    
        linfit = interp1d([1,5], np.vstack([mu_1, mu_2]), axis=0)
        mus = list(linfit([1,2,3,4,5]))
        linfit = interp1d([1,5], np.vstack([Sigma_1, Sigma_2]), axis=0)
        Sigmas = list(linfit([1,2,3,4,5]))
    
        # USE THIS IF YOU WANT TO PLOT A FEW CASES AS AN EXAMPLE
        # linfit = interp1d([1,5], np.vstack([mu_1, mu_2]), axis=0)
        # mus = list(linfit([2,4]))
        # linfit = interp1d([1,5], np.vstack([Sigma_1, Sigma_2]), axis=0)
        # Sigmas = list(linfit([2,5]))

        Sigma_2s = []
        for Sigma in Sigmas:
            Sigma_2 = (Sigma/3)**2
            Sigma_2s += [Sigma_2]
    
        req_list = [[req_type_1, req_type_2], mus, Sigma_2s ]
        req_combinations = list(itertools.product(*req_list)) 
    #===========================================================================
    # Thermal loadcase guassian parameters LHS sampling
    elif sampling == 'LHS':
    
        req_combinations = []
        for point in points_us:
            
            for req_type in [req_type_1,req_type_2]:
                
                mu_lhs = point[0:4]
                Sigma_lhs = point[4::]
                Sigma_lhs_2 = (Sigma_lhs/3)**2
                
                line = [req_type,mu_lhs,Sigma_lhs_2]
                req_combinations += [line]

    if process_DOE_requirements:
        req_index = 0; resiliance_th_vec = []; buffer_th_vec = []; excess_th_vec = []
        for req in req_combinations: # iterate over all combinations of requirements
            
            req_index += 1
            [req_type, mu, Sigma] = req
            
            if req_indices is not None: # None means process all requirements
                process_this_req = req_index in req_indices
            else:
                process_this_req = True

            if process_TH and process_this_req:
                resiliance_th,_,_,buffer_th, excess_th = process_requirements(
                    index,base_name,current_path,bounds_th,
                    mu,Sigma,req_type,variable_lbls,threshold,
                    LHS_MCI_file,req_index,server,DOE_inputs,outputs,plt,
                    resolution=resolution,plot_R_space=True,new_LHS_MCI=new_LHS_MCI,
                    plot_index=plot_index,plot_2D=plot_2D)
            else:
                resiliance_th = 0.0; buffer_th = 0.0; excess_th = 0.0
                
            resiliance_th_vec += [resiliance_th]
            buffer_th_vec += [buffer_th]
            excess_th_vec += [excess_th]
            print('(TH) Resiliance: %f, Buffer: %f, Excess: %f' %(resiliance_th,buffer_th,excess_th))
            
        if process_TH:
            server.sgtelib_server_stop()
            server.server_print(server.server_process) # print console output
        
        # plt.show()
    else:
        resiliance_th_vec = [0.0] * len(req_combinations)
        buffer_th_vec = [0.0] * len(req_combinations)
        excess_th_vec = [0.0] * len(req_combinations)
        
    print("ANALYSIS COMPLETE")
    print("+================================================================+")   
      
    #--------------------------------------------------------------------------#
    # Get volume results
    print("------------------------- %s -------------------------\n" %('VOLUME'))
    filename = "%i_volume_out_file.log" %(index)
    body_full_name = os.path.join(current_path,'Job_results','Results_log',filename)
     
    # Read data from fatigue analysis result and output minimum fatigue life
    fileID = open(body_full_name,'r') # Open file
    InputText = np.loadtxt(fileID, delimiter = '\n', dtype=np.str) # \n is the delimiter
    volume = float(InputText)
    fileID.close()
    
    #===========================================================================
    density = 8.19e-06
    print('Volume: %f mm^3' %(volume))
    print('Weight: %f kg\n' %(volume*density))

    #--------------------------------------------------------------------------#
    # Get Temperature results
    filename = "%i_temp_out_file.log" %(index)
    temp_full_name = os.path.join(current_path,'Job_results','Results_log',filename)
    inputfile=open(temp_full_name,'r') # Read temperature file

    InputText = np.loadtxt(inputfile,
                           delimiter = ',',
                           dtype=np.float) # \n is the delimiter
    InputText = np.atleast_1d(InputText) # convert to 1d array

    t_max = []
    for value in InputText:
        t_max += [float(value)]

    print('The maximum temperature is: %f degC' %(min(t_max)))

    #--------------------------------------------------------------------------#
    # Get Heatflux Power results
    filename = "%i_heatpower_out_file.log" %(index)
    temp_full_name = os.path.join(current_path,'Job_results','Results_log',filename)
    inputfile=open(temp_full_name,'r') # Read temperature file

    InputText = np.loadtxt(inputfile,delimiter = ',',dtype=np.float) # \n is the delimiter
    InputText = np.atleast_1d(InputText) # convert to 1d array

    p_n = []
    for value in InputText:
        p_n += [float(value)]

    print('The average power is: %f W/mm^2' %(np.mean(p_n)))

    #--------------------------------------------------------------------------#
    # Get Heat Area results
    filename = "%i_heatarea_out_file.log" %(index)
    temp_full_name = os.path.join(current_path,'Job_results','Results_log',filename)
    inputfile=open(temp_full_name,'r') # Read temperature file

    InputText = np.loadtxt(inputfile,delimiter = ',',dtype=np.float) # \n is the delimiter
    InputText = np.atleast_1d(InputText) # convert to 1d array

    a_n = []
    for value in InputText:
        a_n += [float(value)]

    print('The average heat area is: %f mm^2\n' %(np.mean(a_n)))
    inputfile.close()

    #--------------------------------------------------------------------------#
    # Get build time results
    print("---------------------- %s ---------------------\n" %('BUILD TIME RESULTS'))
    filename = "%i_btime_out_file.log" %(index)
    btime_full_name = os.path.join(current_path,'Job_results','Results_log',filename)

    # Read data from fatigue analysis result and output minumum fatigue life
    fileID = open(btime_full_name,'r') # Open file
    InputText = np.loadtxt(fileID, delimiter = '\n', dtype=np.str) # \n is the delimiter
    btime = float(InputText)
    fileID.close()

    #--------------------------------------------------------------------------#
    # Write results summary
    summary_of_results = '%i_DED_result.log' %(index)
    SOR_full_name = os.path.join(current_path,'Job_results','Results_log',summary_of_results)

    resultsfile=open(SOR_full_name,'w+')
    resultsfile.write('---------------------------'+"\n")
    resultsfile.write('Build time'+' : '+str(btime)+"\n")
    resultsfile.write('Maximum_temp'+' : '+str(max(t_max))+"\n")
    resultsfile.write('Average_flux'+' : '+str(np.mean(p_n))+"\n")
    resultsfile.write('Average_area'+' : '+str(np.mean(a_n))+"\n")
    resultsfile.write('Maximum residual stress'+' : '+str(max(s_res))+"\n")
    resultsfile.write('Residual stress distribution'+' : '+','.join(map(str,s_res))+"\n")
    resultsfile.write('---------------------------'+"\n"+'END\n')
    resultsfile.close()

    design_data = [base_ip_results, base_th_results, ip_results, th_results, 
                   min(t_max), np.mean(p_n), np.mean(a_n), max(s_res), U_res[0],
                   float(volume)*density, btime, max(U_p_res_line), max(S_res_line),
                   resiliance_ip_vec_nominal, resiliance_th_vec_nominal,
                   R_volume_ip_vec_nominal, R_volume_th_vec_nominal,
                   capability_ip_vec_nominal, capability_th_vec_nominal,
                   buffer_ip_vec_nominal, buffer_th_vec_nominal,
                   excess_ip_vec_nominal, excess_th_vec_nominal]

    resiliance_data = [resiliance_ip_vec, buffer_ip_vec, excess_ip_vec, 
                       resiliance_th_vec, buffer_th_vec, excess_th_vec, 
                       req_index]

    plt.close('all')

    return design_data, resiliance_data

#------------------------------------------------------------------------------#
# %% MAIN FILE
def main():
    import sys, os
    import numpy as np
    from scipy.io import loadmat
    from pyDOE import lhs
    import pickle
    
    current_path = os.getcwd() # Working directory of file
    
    #============================ PERMUTATIONS ================================#
    
    # one-liner to read a single variable
    P_analysis = loadmat('DOE_permutations.mat')['P_analysis']
    
    #============================= VARIABLES ==================================#
    input_file_name = 'DED_vars_inputs.log'
    
    Inomad_full_name = os.path.join(current_path,'Optimization_studies',input_file_name)
    inputfile=open(Inomad_full_name,'r')
    
    InputText = np.loadtxt(inputfile,
       delimiter = ' ',
       dtype=np.str) # \n is the delimiter
    
    ax_pos =       float(InputText[0])
    st_height =    float(InputText[1])
    st_width =     float(InputText[2])
    laser_power =  float(InputText[3])
    b_thick =      float(InputText[4])
    shroud_width = float(InputText[5])
    
    #============================= PARAMETERS =================================#
    para_file_name = 'DED_parameters.log'
    
    Ipara_full_name = os.path.join(current_path,'Optimization_studies',para_file_name)
    paramfile=open(Ipara_full_name,'r')
    
    paramText = np.loadtxt(paramfile,
       delimiter = ' ',
       dtype=np.str) # \n is the delimiter
    
    index = int(float(paramText[0]))
    
    concept = int(float(paramText[1]))
    permutation_index = []
    for entry in paramText[2:7]:
        permutation_index += [int(float(entry))]
    
    print(paramText)
    
    IP_n =           float(paramText[7])
    T1_n =           float(paramText[8])
    T2_n =           float(paramText[9])
    T3_n =           float(paramText[10])
    T4_n =           float(paramText[11])
    st_thick =       float(paramText[12])
    scanning_speed = float(paramText[13])
    power_density =  float(paramText[14])
    layer_length =   float(paramText[15])
    layer_width =    float(paramText[16])
    layer_thick =    float(paramText[17])
    max_T_pool =     float(paramText[18])
    n_layers =       int(float(paramText[19]))
    n_deposit =      int(float(paramText[20]))
    mesh_size =      float(paramText[21])
    mesh_AM_size =   float(paramText[22])
    melting_T =      float(paramText[23])
    H_subs =         float(paramText[24])
    T_ref =          float(paramText[25])
    
    run_base = 0; run_nominal = 1; new_LHS = False; process_DOE_requirements = False; sampling = 'fullfact'
    new_LHS_MCI = True; plot_index = 3; plot_2D = True

    # %% Sampling
    #========================== REQUIREMENTS SPACE LHS ============================#
        
    # IP loadcase guassian parameters
    mu_lob = np.array([0.250])
    mu_upb = np.array([0.625])
     
    Sigma_lob = np.array([ 0.167 ])
    Sigma_upb = np.array([ 0.375 ])
    
    lob_req = np.append(mu_lob,Sigma_lob)
    upb_req = np.append(mu_upb,Sigma_upb)
    
    # LHS distribution
    if new_LHS:
        points = lhs(len(lob_req), samples=100, criterion='maximin')
        points_us = scaling(points,lob_req,upb_req,2) # unscale latin hypercube points
        
        DOE_full_name = 'req_distribution_IP_LHS'+'.npy'
        DOE_filepath = os.path.join(current_path,'Optimization_studies',DOE_full_name)
        np.save(DOE_filepath, points_us) # save DOE array
        
        DOE_full_name = 'req_distribution_IP_LHS_data'+'.pkl'
        DOE_filepath = os.path.join(current_path,'Optimization_studies',DOE_full_name)
        
        resultsfile=open(DOE_filepath,'wb')
        
        pickle.dump(lob_req, resultsfile)
        pickle.dump(upb_req, resultsfile)
        pickle.dump(points, resultsfile)
        pickle.dump(points_us, resultsfile)
    
        resultsfile.close()
        
    else:
        DOE_full_name = 'req_distribution_IP_LHS'+'.npy'
        DOE_filepath = os.path.join(current_path,'Optimization_studies',DOE_full_name)
        points_us = np.load(DOE_filepath) # load DOE array
    
    # Thermal loadcase guassian parameters
    # mu_lob = np.array([0.375, 0.80, 0.80, 0.625])
    # mu_upb = np.array([0.625, 0.20, 0.20, 0.375])
    mu_lob = np.array([0.15, 0.80, 0.80, 0.85])
    mu_upb = np.array([0.85, 0.20, 0.20, 0.15])
     
    Sigma_lob = np.array([0.1875, 0.125, 0.125, 0.1875]) # sigma^2
    Sigma_upb = np.array([0.375 , 0.250, 0.250, 0.375]) # sigma^2
    
    lob_req = np.append(mu_lob,Sigma_lob)
    upb_req = np.append(mu_upb,Sigma_upb)
    
    # LHS distribution
    if new_LHS:
        points = lhs(len(lob_req), samples=800, criterion='maximin')
        points_us = scaling(points,lob_req,upb_req,2) # unscale latin hypercube points
        
        DOE_full_name = 'req_distribution_TH_LHS'+'.npy'
        DOE_filepath = os.path.join(current_path,'Optimization_studies',DOE_full_name)
        np.save(DOE_filepath, points_us) # save DOE array
        
            
        DOE_full_name = 'req_distribution_TH_LHS_data'+'.pkl'
        DOE_filepath = os.path.join(current_path,'Optimization_studies',DOE_full_name)
        
        resultsfile=open(DOE_filepath,'wb')
        
        pickle.dump(lob_req, resultsfile)
        pickle.dump(upb_req, resultsfile)
        pickle.dump(points, resultsfile)
        pickle.dump(points_us, resultsfile)
    
        resultsfile.close()
        
    else:
        DOE_full_name = 'req_distribution_TH_LHS'+'.npy'
        DOE_filepath = os.path.join(current_path,'Optimization_studies',DOE_full_name)
        points_us = np.load(DOE_filepath) # load DOE array
    
    # %% Processing
    #============================= MAIN EXECUTION =================================#
    
    ##############################
    # Presentation graphics
    design_indices = [109, 110]
    req_indices = [[None,],[None,]]
    
    # design_indices = [109, 110, 146, 163, 164, 167, 168]

    # req_indices = [[1,], # D: 109
    #               [15, 11, 37,], # D: 110
    #               [36,], # D: 146
    #               [50, 1,], # D: 163
    #               [1,], # D: 164
    #               [46, 13,], # D: 167
    #               [31,]] # D: 168

    # design_indices = [146, 163, 164, 167, 168]

    # req_indices = [[36,], # D: 146
    #                [50, 1,], # D: 163
    #                [1,], # D: 164
    #                [46, 13,], # D: 167
    #                [31,]] # D: 168

    ##############################

    ##############################
    # For a general DOE
    # index = 0
    # P_analysis = P_analysis[0:15] # for testing
    # for P_i in P_analysis:
    #     index += 1
    ##############################

    ##############################
    for D_i, reqs in zip(design_indices,req_indices):
        P_i = P_analysis[D_i - 1]
        index = D_i
    ##############################
        print(P_i)
        concept = P_i[0]
        permutation_index = P_i[1::]
        
        print("\n+============================================================+")
        print("|                         LOOP %04d                          |" %(index))
        print("+============================================================+\n")
        


        [design_data,resiliance_data] = DED_blackbox_evaluation(concept, permutation_index, run_base, run_nominal, 
                                              ax_pos, st_thick, st_width, laser_power, scanning_speed, 
                                              power_density, layer_length, layer_width, layer_thick, 
                                              n_layers, n_deposit, mesh_size, mesh_AM_size, 
                                              melting_T, b_thick, process_DOE_requirements, sampling, 
                                              new_LHS_MCI, index, plot_index=plot_index,plot_2D=plot_2D,
                                              req_indices=reqs)
        
        [resiliance_ip_vec, buffer_ip_vec, excess_ip_vec, 
         resiliance_th_vec, buffer_th_vec, excess_th_vec, n_reqs] = resiliance_data

        # %% Postprocessing
        [base_ip_results, base_th_results, ip_results, th_results, 
         temp, h_flux, h_area, S_residual, U_res, weight, build_time, 
         max_res_U, max_res_S, 
         resiliance_ip_vec_nominal, resiliance_th_vec_nominal,
         R_volume_ip_vec_nominal, R_volume_th_vec_nominal,
         capability_ip_vec_nominal, capability_th_vec_nominal,
         buffer_ip_vec_nominal, buffer_th_vec_nominal,
         excess_ip_vec_nominal, excess_th_vec_nominal] = design_data

        [S_ip,N_ip,n_f_ip,U_ip] = ip_results
        [S_th,N_th,n_f_th,U_th] = th_results
        
        [base_S_ip,base_N_ip,base_n_f_ip,base_U_ip] = base_ip_results
        [base_S_th,base_N_th,base_n_f_th,base_U_th] = base_th_results
 
        out_titles = ['base_S_ip', 'base_N_ip', 'base_n_f_ip', 'base_U_ip', 
                      'base_S_th', 'base_N_th', 'base_n_f_th', 'base_U_th',
                      'S_ip', 'N_ip', 'n_f_ip', 'U_ip', 'S_th', 'N_th', 'n_f_th', 'U_th',
                      'weight', 'temp', 'h_flux', 'h_area', 'S_residual', 
                      'U_res', 'max_T_pool', 'build_time', 'max_res_U', 
                      'max_res_S', 
                      'resiliance_ip_uni','resiliance_ip_gau',
                      'R_volume_ip_uni','R_volume_ip_gau',
                      'capability_ip_uni','capability_ip_gau',
                      'buffer_ip_uni','buffer_ip_gau',
                      'excess_ip_uni','excess_ip_gau',
                      'resiliance_th_uni','resiliance_th_gau',
                      'R_volume_th_uni','R_volume_th_gau',
                      'capability_th_uni','capability_th_gau',
                      'buffer_th_uni','buffer_th_gau',
                      'excess_th_uni','excess_th_gau']
        
        req_titles = []
        for r_i in range(n_reqs):
            req_titles += ['req_index_%i' %(r_i+1)]
        
        out_data = [base_S_ip,base_N_ip,base_n_f_ip,base_U_ip,
                    base_S_th,base_N_th,base_n_f_th,base_U_th,
                    S_ip, N_ip, n_f_ip, U_ip, S_th, N_th, n_f_th, U_th,
                    weight, temp, h_flux, h_area, S_residual, 
                    U_res, max_T_pool, build_time, max_res_U, 
                    max_res_S]
    
        #========================== OUTPUT VARIABLES LOG ==============================#
        filename = "varout_opt_log.log"
        filename_ip = "resiliance_ip.log"
        filename_th = "resiliance_th.log"
        filename_buffer_ip = "buffer_ip.log"
        filename_buffer_th = "buffer_th.log"
        filename_excess_ip = "excess_ip.log"
        filename_excess_th = "excess_th.log"
        full_filename = os.path.join(current_path,'Optimization_studies',filename)
        full_filename_ip = os.path.join(current_path,'Optimization_studies',filename_ip)
        full_filename_th = os.path.join(current_path,'Optimization_studies',filename_th)
        full_filename_buffer_ip = os.path.join(current_path,'Optimization_studies',filename_buffer_ip)
        full_filename_buffer_th = os.path.join(current_path,'Optimization_studies',filename_buffer_th)
        full_filename_excess_ip = os.path.join(current_path,'Optimization_studies',filename_excess_ip)
        full_filename_excess_th = os.path.join(current_path,'Optimization_studies',filename_excess_th)
        
        if index == 1:
            resultsfile=open(full_filename,'w')
            resultsfile.write('index'+','+'concept'+','+'i1'+','+'i2'+','+'i3'+','+'i4'+','+'i5'+','
                              +'ax_pos'+','+'st_height'+','+'st_width'+','+'st_thick'+','
                              +'laser_power'+','+'scanning_speed'+','+'power_density'+','
                              +'n_layers'+','+'n_deposits'+','+'layer_thick'+','+'layer_width'+','+'layer_length'+','
                              +','.join(out_titles)+'\n')
            
            resultsfile_ip=open(full_filename_ip,'w')
            resultsfile_ip.write('index'+','+'concept'+','+'i1'+','+'i2'+','+'i3'+','+'i4'+','+'i5'+','
                              +','.join(req_titles)+'\n')
            
            resultsfile_th=open(full_filename_th,'w')
            resultsfile_th.write('index'+','+'concept'+','+'i1'+','+'i2'+','+'i3'+','+'i4'+','+'i5'+','
                              +','.join(req_titles)+'\n')

            resultsfile_buffer_ip=open(full_filename_buffer_ip,'w')
            resultsfile_buffer_ip.write('index'+','+'concept'+','+'i1'+','+'i2'+','+'i3'+','+'i4'+','+'i5'+','
                                        +','.join(req_titles)+'\n')

            resultsfile_buffer_th=open(full_filename_buffer_th,'w')
            resultsfile_buffer_th.write('index'+','+'concept'+','+'i1'+','+'i2'+','+'i3'+','+'i4'+','+'i5'+','
                                        +','.join(req_titles)+'\n')

            resultsfile_excess_ip=open(full_filename_excess_ip,'w')
            resultsfile_excess_ip.write('index'+','+'concept'+','+'i1'+','+'i2'+','+'i3'+','+'i4'+','+'i5'+','
                                        +','.join(req_titles)+'\n')

            resultsfile_excess_th=open(full_filename_excess_th,'w')
            resultsfile_excess_th.write('index'+','+'concept'+','+'i1'+','+'i2'+','+'i3'+','+'i4'+','+'i5'+','
                                        +','.join(req_titles)+'\n')
        
        resultsfile=open(full_filename,'a+')
        resultsfile.write(str(index)+','+str(concept)+','+','.join(map(str,permutation_index))+','
                          +str(ax_pos)+','+str(st_height)+','+str(st_width)+','+str(st_thick)+','
                          +str(laser_power)+','+str(scanning_speed)+','+str(power_density)+','
                          +str(n_layers)+','+str(n_deposit)+','+str(layer_thick)+','+str(layer_width)+','+str(layer_length)+','
                          +','.join(map(str,out_data))+','
                          +','.join(map(str,resiliance_ip_vec_nominal))+','
                          +','.join(map(str,R_volume_ip_vec_nominal))+','
                          +','.join(map(str,capability_ip_vec_nominal))+','
                          +','.join(map(str,buffer_ip_vec_nominal))+','
                          +','.join(map(str,excess_ip_vec_nominal))+','
                          +','.join(map(str,resiliance_th_vec_nominal))+','
                          +','.join(map(str,R_volume_th_vec_nominal))+','
                          +','.join(map(str,capability_th_vec_nominal))+','
                          +','.join(map(str,buffer_th_vec_nominal))+','
                          +','.join(map(str,excess_th_vec_nominal))+'\n')
        resultsfile.close()

        resultsfile_ip=open(full_filename_ip,'a+')
        resultsfile_ip.write(str(index)+','+str(concept)+','+','.join(map(str,permutation_index))+','
                          +','.join(map(str,resiliance_ip_vec))+'\n')
        resultsfile_ip.close()
        
        resultsfile_th=open(full_filename_th,'a+')
        resultsfile_th.write(str(index)+','+str(concept)+','+','.join(map(str,permutation_index))+','
                          +','.join(map(str,resiliance_th_vec))+'\n')
        resultsfile_th.close()

        resultsfile_buffer_ip=open(full_filename_buffer_ip,'a+')
        resultsfile_buffer_ip.write(str(index)+','+str(concept)+','+','.join(map(str,permutation_index))+','
                          +','.join(map(str,buffer_ip_vec))+'\n')
        resultsfile_buffer_ip.close()
        
        resultsfile_buffer_th=open(full_filename_buffer_th,'a+')
        resultsfile_buffer_th.write(str(index)+','+str(concept)+','+','.join(map(str,permutation_index))+','
                          +','.join(map(str,buffer_th_vec))+'\n')
        resultsfile_buffer_th.close()

        resultsfile_excess_ip=open(full_filename_excess_ip,'a+')
        resultsfile_excess_ip.write(str(index)+','+str(concept)+','+','.join(map(str,permutation_index))+','
                          +','.join(map(str,excess_ip_vec))+'\n')
        resultsfile_excess_ip.close()
        
        resultsfile_excess_th=open(full_filename_excess_th,'a+')
        resultsfile_excess_th.write(str(index)+','+str(concept)+','+','.join(map(str,permutation_index))+','
                          +','.join(map(str,excess_th_vec))+'\n')
        resultsfile_excess_th.close()
        
        print("\n--------------------------------------------------------------------------------\n")

if __name__ == '__main__':
    main()
