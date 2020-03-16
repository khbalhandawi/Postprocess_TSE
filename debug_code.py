# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 15:39:18 2019

@author: Khalil
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt

def fatigue_life_calc(sa,sm,plot_out):
    # import scipy.integrate as integrate
    from numpy import pi
    
    smax = sa + sm;
    smin = sm - sa;
    
    sopen = max([smin,0.0]); # to acount for crack closure effects
    smax = max(smax,1.0); # to make sure there is no division by zero or inf

    K1c = 30.0 # MPa/sqrt(m)
    
#    it_integral = 5.0
#    C = 4e-10; # From J. M. Barsom and S. T. Rolfe
#    m = 3; # From J. M. Barsom and S. T. Rolfe
    it_integral = 10.0
    C = 2e-8; # From J. M. Barsom and S. T. Rolfe
    m = 3.4; # From J. M. Barsom and S. T. Rolfe
    
#    it_integral = 100.0
#    C = 2e-7; # From J. M. Barsom and S. T. Rolfe
#    m = 1; # From J. M. Barsom and S. T. Rolfe
    
    ai = 5e-4; # m
    beta = 1.00;
    af = ( 1/pi ) * ( ( K1c / (beta*smax) ) ** 2 );
    # print('critical crack size: %f mm' %(af*1000.0))

    dN_j = (af - ai)*100*500

    a_j = ai; N_j = 0.0; a_j_plot = []; N_j_plot = [];
    
    while a_j < af:
        
        Kmax = beta*smax*np.sqrt(pi*a_j);
        Kmin = beta*sopen*np.sqrt(pi*a_j);
        R = Kmin/Kmax
        
        dK_1 = Kmax - Kmin;
#        da_j = ( ( C * ( (dK_1) ** m ) ) / ( ((1 - R) * K1c) - dK_1) ) * dN_j # Foreman relation
        da_j = ( ( C * ( (dK_1) ** m ) ) / ( 1.0 ) ) * dN_j # Paris law
        dN_j = ((af - ai)/da_j) * it_integral
        a_j += da_j
        N_j += dN_j
        
        a_j_plot += [a_j]
        N_j_plot += [N_j]
        
        # print(a_j*1000)

    # print(N_j)
    
    if plot_out:
        plt.figure()
        plt.plot(N_j_plot, a_j_plot, '-r', label='Nominal')
        plt.show()
    
    # result = integrate.quad(lambda a: 1/((beta*s_eff*np.sqrt(pi*a))**m), ai, af)
    # Nf = (1/C)*result
    
    return N_j

def fatigue_parameters(S_ut,S_y,compression):
    
    a = 4.51;
    b = -0.265;
    ka = a*(S_ut**(b)) # surface condition modification factor
    kb = 1.0; # size modification factor
    kc  = 0.85; # load modification factor
    kd = 0.8; # temperature modification factor
    ke = 0.7; # reliability factor
    kf = 1.0; # miscellaneous-effects modification factor
    
    K_f = 1.0 # Stress concentration factors (SAME SIZE AS HIST REGIONS)
    
    S_e = ka*kb*kc*kd*ke*kf*S_ut;
    
    f = 0.76 # Fig 6-18 fatigue strength fraction
    a = ((f*S_ut)**2)/S_e
    b = (-1/3)*np.log10((f*S_ut)/S_e)
    # print("Se: %f" %(S_e))
    # print("a: %f, b: %f \n" %(a,b))
    
    return f,a,b,S_e

#==============================================================================#
# %% FATIGUE CALCULATION SUBROUTINE
def fatigue_calculation( s1,s2,p1,p2,N_plot ):
    
    import numpy as np
    
    # Find sign of Mises stress (-: comp, +: tension)
    s1 = s1*np.sign(p1)
    s2 = s2*np.sign(p2)
    
    # Find sign of Mises stress (-: comp, +: tension)
    s1 = s1* -1.0; s2 = s2* -1.0; 
    s_m0 = ((s1+s2)/2.0)
    s_a0 = abs(s2-s1)/2.0
    
    K_f = 1.0
    
    s_m = np.asarray(s_m0) * K_f
    s_a = np.asarray(s_a0) * K_f
    
    i = -1;
    N = np.empty(len(s_a))
    n_f = np.empty(len(s_a))
    S_rev = np.empty(len(s_a))
    for s_an,s_mn in zip(s_a,s_m):
        i += 1;
        
        S_ut = 1332.0 # Ultimate strength
        S_y = 1100.0 # Yield strength
        compression = True
        [f,a,b,S_e] = fatigue_parameters(S_ut,S_y,compression)
        
        if i in N_plot:
            plot_out = True
        else:
            plot_out = False
        
        N[i] = fatigue_life_calc(s_an,s_mn,plot_out);
        
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
    
    return N, n_f, S_rev

def load_res_results(fname):
    import numpy as np
    #--------------------------------------------------------------------------#
    # PLOT ELEMENT LINE PATH RESULTS
    
    # Get elemental results at line path location
    resultsfile=open(fname,'rb')
    
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
    
    # Plot stress data
    S_11_line = []; S_22_line = []; S_33_line = []; S_12_line = []; S_13_line = []; S_23_line = [];
    S_line = []; SP1_line = []; SP2_line = []; SP3_line = []; pressure_line = []; pressure_hoop_line = []; stress_hoop_line = [];
    for S_e,SP1_e,SP2_e,SP3_e,S_11_e,S_22_e,S_33_e,S_12_e,S_13_e,S_23_e,Press_e,Press_hoop_e,stress_hoop_e in zip(stress_p,SP1_p,SP2_p,SP3_p,S11_p,S22_p,S33_p,S12_p,S13_p,S23_p,Press_p,Press_hoop_p,stress_hoop_p):
        S_11_line += [S_11_e[-1]] # extract last time entry for data type
        S_22_line += [S_22_e[-1]] # extract last time entry for data type
        S_33_line += [S_33_e[-1]] # extract last time entry for data type
        S_12_line += [S_12_e[-1]] # extract last time entry for data type
        S_13_line += [S_13_e[-1]] # extract last time entry for data type
        S_23_line += [S_23_e[-1]] # extract last time entry for data type
        S_line += [S_e[-1]] # extract last time entry for data type
        SP1_line += [SP1_e[-1]] # extract last time entry for data type
        SP2_line += [SP2_e[-1]] # extract last time entry for data type
        SP3_line += [SP3_e[-1]] # extract last time entry for data type
        pressure_line += [Press_e[-1]]
        pressure_hoop_line += [Press_hoop_e[-1]]
        stress_hoop_line += [stress_hoop_e[-1]]
    
    S_11_line = np.array(S_11_line)
    S_22_line = np.array(S_22_line)
    S_33_line = np.array(S_33_line)
    S_12_line = np.array(S_12_line)
    S_13_line = np.array(S_13_line)
    S_23_line = np.array(S_23_line)
    pressure_line = np.array(pressure_line)
    S_line = np.array(S_line)
    
#    pressure_line = -1/3.0 * ( S_22_line + S_33_line )
#    S_line = ( ( S_22_line ) ** 2 + ( S_33_line ) ** 2  - ( S_22_line * S_33_line ) ) ** (1/2.0)
    
    return stress_hoop_line, pressure_hoop_line

def load_e_results(fname):
        
    resultsfile = open(fname,'rb')
    
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
    S_11_line = []; S_22_line = []; S_33_line = []; S_12_line = []; S_13_line = []; S_23_line = [];
    S_line = []; S_M_line = []; S_A_line = []; pressure_line = []; n_f_line = []; N_line = [];
    pressure_hoop_line = []; stress_hoop_line = [];
    plot_data = zip(S11_p,S22_p,S33_p,S12_p,S13_p,S23_p,stress_p,S_M_p,S_A_p,Press_p,Press_hoop_p,stress_hoop_p,N_p,n_f_p);
    for S_11_e,S_22_e,S_33_e,S_12_e,S_13_e,S_23_e,S_e,S_M_e,S_A_e,Press_e,Press_hoop_e,stress_hoop_e,N_e,n_f_e in plot_data:
        S_11_line += [S_11_e[-1]] # extract last time entry for data type
        S_22_line += [S_22_e[-1]] # extract last time entry for data type
        S_33_line += [S_33_e[-1]] # extract last time entry for data type
        S_12_line += [S_12_e[-1]] # extract last time entry for data type
        S_13_line += [S_13_e[-1]] # extract last time entry for data type
        S_23_line += [S_23_e[-1]] # extract last time entry for data type
        S_line += [S_e[-1]] # extract last time entry for data type
        S_M_line += [S_M_e[-1]] # extract last time entry for data type
        S_A_line += [S_A_e[-1]] # extract last time entry for data type
        pressure_line += [Press_e[-1]]
        pressure_hoop_line += [Press_hoop_e[-1]]
        stress_hoop_line += [stress_hoop_e[-1]]
        N_line += [N_e[-1]] 
        n_f_line += [n_f_e[-1]]
    
#    # Plot stress data
#    S_11_line = []; S_22_line = []; S_33_line = []; S_12_line = []; S_13_line = []; S_23_line = [];
#    S_line = []; S_M_line = []; S_A_line = []; pressure_line = []; n_f_line = []; N_line = [];
#    plot_data = zip(S11_p,S22_p,S33_p,S12_p,S13_p,S23_p,stress_p,S_M_p,S_A_p,Press_p,N_p,n_f_p);
#    for S_11_e,S_22_e,S_33_e,S_12_e,S_13_e,S_23_e,S_e,S_M_e,S_A_e,Press_e,N_e,n_f_e in plot_data:
#        S_11_line += [0.0] # extract last time entry for data type
#        S_22_line += [S_22_e[-1]] # extract last time entry for data type
#        S_33_line += [S_33_e[-1]] # extract last time entry for data type
#        S_12_line += [0.0] # extract last time entry for data type
#        S_13_line += [0.0] # extract last time entry for data type
#        S_23_line += [S_23_e[-1]] # extract last time entry for data type
#        S_M_line += [S_M_e[-1]] # extract last time entry for data type
#        S_A_line += [S_A_e[-1]] # extract last time entry for data type
#        pressure_line += [Press_e[-1]]
    
    S_11_line = np.array(S_11_line)
    S_22_line = np.array(S_22_line)
    S_33_line = np.array(S_33_line)
    S_12_line = np.array(S_12_line)
    S_13_line = np.array(S_13_line)
    S_23_line = np.array(S_23_line)
    pressure_line = np.array(pressure_line)
    S_line = np.array(S_line)
    
#    S_line = np.sqrt( 0.5 * ( ( ( S_11_line - S_22_line ) ** 2 ) + ( ( S_11_line - S_33_line ) ** 2 ) + 
#       ( ( S_22_line - S_33_line ) ** 2 ) + 6 * ( ( S_12_line**2 ) + ( S_13_line**2 ) + ( S_23_line**2 ) ) ) )
    
#    pressure_line = -1/3.0 * ( S_22_line + S_33_line )
#    S_line = ( ( S_22_line ) ** 2 + ( S_33_line ) ** 2  - ( S_22_line * S_33_line ) ) ** (1/2.0)
    return angle_p, n_f_line, N_line, stress_hoop_line, pressure_hoop_line, S_M_line, S_A_line

def main():
    import os
    load_res = True;
    DOEin = np.load(os.path.join(os.getcwd(),'Job_results','Results_log','8_DOE_th_inputs_1.npy'))
    
    fname_res = os.path.join(os.getcwd(),'Job_results','Results_log','8_hline_out_e_file.pkl')
    fname_base = os.path.join(os.getcwd(),'Job_results','Results_log','8_base_thermal_out_e_file.pkl')
    fname = os.path.join(os.getcwd(),'Job_results','Thermal_DOE_results','8_49_thermal_out_e_file.pkl')
        
    N_plot = [1091,];
        
#    N_f = fatigue_life_calc(25.054283142089844,25.054283142089844,N_plot)
#    N_f = fatigue_life_calc(400,100,N_plot)
#    N_f = fatigue_life_calc(2.3998241424560547,210.27264976501465,N_plot)
    #-------------------------------------------------------------------------#
    [angle_p_base, n_f_line_base, N_line_base, S_line_base, pressure_line_base, S_M_line_base, S_A_line_base] =  load_e_results(fname_base)
    
    if load_res:
        [stress_res, Press_res] =  load_res_results(fname_res)
    else:
        stress_res = np.zeros_like(S_line_base)
        Press_res = np.zeros_like(pressure_line_base)

    [N_line_base, n_f_line_base, S_rev_base] = fatigue_calculation( np.zeros_like(S_line_base),S_line_base,np.zeros_like(pressure_line_base),pressure_line_base,N_plot ) # perform fatigue calculation
    
    #-------------------------------------------------------------------------#
    [angle_p, n_f_line, N_line, S_line, pressure_line, S_M_line, S_A_line] =  load_e_results(fname)
    [N_line, n_f_line, S_rev] = fatigue_calculation( stress_res,S_line,Press_res,pressure_line,N_plot ) # perform fatigue calculation
    
    plt.figure()
    plt.title('Pressure')
    plt.plot(angle_p, pressure_line, '-r', label='Nominal')
    plt.plot(angle_p_base, pressure_line_base, '-b', label='Base')
    plt.legend(loc = 1)
    
    plt.figure()
    plt.title('Stress')
    plt.plot(angle_p, S_line, '-r', label='Nominal')
    plt.plot(angle_p_base, S_line_base, '-b', label='Base')
    plt.legend(loc = 1)
          
    # Find sign of Mises stress (-: comp, +: tension)
    s1 = np.zeros_like(S_line_base)
    s2 = S_line_base*np.sign(pressure_line_base)
    
    # Find sign of Mises stress (-: comp, +: tension)
    s1 = s1* -1.0; s2 = s2* -1.0;
    sm = ((s1+s2)/2.0)
    sa = abs(s2-s1)/2.0
    smax = sa + sm;
    smin = sm - sa;
    sopen = np.array([max([smin_n,0.0]) for smin_n in smin]); # to acount for crack closure effects
    smax = np.array([max([smax_n,1.0]) for smax_n in smax]); # to make sure there is no division by zero or inf
    s_eff_base = smax - sopen;
    
    # Find sign of Mises stress (-: comp, +: tension)
    s1 = stress_res*np.sign(Press_res)
    s2 = S_line*np.sign(pressure_line)
    
    # Find sign of Mises stress (-: comp, +: tension)
    s1 = s1* -1.0; s2 = s2* -1.0;
    sm = ((s1+s2)/2.0)
    sa = abs(s2-s1)/2.0
    smax = sa + sm;
    smin = sm - sa;
    sopen = np.array([max([smin_n,0.0]) for smin_n in smin]); # to acount for crack closure effects
    smax = np.array([max([smax_n,1.0]) for smax_n in smax]); # to make sure there is no division by zero or inf
    s_eff = smax - sopen;
    
    plt.figure()
    plt.title('s_eff')
    plt.plot(angle_p, s_eff, '-r', label='Nominal')
    plt.plot(angle_p_base, s_eff_base, '-b', label='Base')
    plt.legend(loc = 1)
    
    if load_res:
        plt.figure()
        plt.title
        plt.title('residuals')
        plt.plot(angle_p, Press_res, '-r', label='press')
        plt.plot(angle_p_base, stress_res, '-b', label='$\sigma_v$')
        plt.legend(loc = 1)

    print('max S: %f' %max(S_line))
    print('min S: %f' %min(S_line))
    print('max S_res: %f' %max(stress_res))
    print('min S_res: %f' %min(stress_res))
    print('max S_base: %f' %max(S_line_base))
    print('min S_base: %f' %min(S_line_base))
    print('avg n_f: %f' %min(n_f_line))
    print('avg n_f base: %f' %min(n_f_line_base))
    print('max s_eff: %f' %max(s_eff))
    print('max s_eff base: %f' %max(s_eff_base))
    print('min N: %f' %min(N_line))
    print('min i N: %i' %(N_line.argmin()))  # N_line.index(min(N_line))
    print('min N base: %f' %min(N_line_base))
    print(len(n_f_line))
    plt.xlabel('Angle ($^o$)') # x Axis labels
    plt.ylabel('Stress (MPa)') # y Axis label
    plt.tight_layout()
    
if __name__ == '__main__':
    main()