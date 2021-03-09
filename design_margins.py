# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 01:34:24 2017

@author: Khalil
"""

from visualization import parallel_plots, define_SGTE_model, hyperplane_SGTE_vis_norm, gridsamp
from SGTE_library import SGTE_server
import pickle, os, sys
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import rc
import itertools
from pyDOE import lhs
from scipy.io import loadmat
import copy 

#==============================================================================#
# SCALING BY A RANGE
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
# POSTPROCESS DOE DATA TO GET CAPABILITY, RESILIANCE AND EXCESS
        
def Rspace_calculation(server,bounds_req,mu,Sigma,req_type,bounds,res,threshold,
                           new_LHS_MCI,LHS_MCI_file):

    lob = bounds[:,0]
    upb = bounds[:,1]
    
    lob_req = bounds_req[:,0]
    upb_req = bounds_req[:,1]

    if req_type == "guassian":

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

    return R_volume

def resiliance_calculation(server,bounds_req,mu,Sigma,req_type,bounds,res,threshold,
                           new_LHS_MCI,LHS_MCI_file):
    
    lob = bounds[:,0]
    upb = bounds[:,1]
    
    lob_req = bounds_req[:,0]
    upb_req = bounds_req[:,1]

    # LHS distribution
    DOE_full_name = LHS_MCI_file +'.pkl'
    DOE_filepath = os.path.join(os.getcwd(),'design_margins',DOE_full_name)

    # LHS distribution
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

        resiliance = len(cond_req_feas[cond_req_feas])/np.shape(dFF)[0]
        
    return resiliance

def capability_calculation(server,bounds_req,bounds,res,threshold,
                           new_LHS_MCI,LHS_MCI_file):
    
    lob = bounds[:,0]
    upb = bounds[:,1]
    
    lob_req = bounds_req[:,0]
    upb_req = bounds_req[:,1]

    # LHS distribution
    DOE_full_name = LHS_MCI_file +'.pkl'
    DOE_filepath = os.path.join(os.getcwd(),'design_margins',DOE_full_name)

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

    cond_req_feas = (YX - threshold) > 0
    
    # Design space volume
    capability = len(cond_req_feas[cond_req_feas])/np.shape(dFF_Pspace)[0]
        
    return capability

def multivariate_gaussian(pos, mu, Sigma):
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
# POSTPROCESS DOE DATA
def postprocess_DOE(index,base_name,current_path,bounds,variable_lbls_pc,
                    permutation_index,concept,metric,plot_para_coords):

    # Import DOE outputs from pickle file
    backup_file = '%i_%s_DOE_out.pkl' %(index,base_name[1])
    backup_filepath = os.path.join(os.getcwd(),'Job_results','Results_log_EX',backup_file) 
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

    print(np.shape(S_n)); print(np.shape(Y))
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
        parallel_filepath = os.path.join(current_path,'design_margins',parallel_file)
        parallel_plots(DOE_inputs,outputs,variable_lbls_pc,output_lbls,parallel_filepath)
    #===========================================================================  

    return server, DOE_inputs, outputs

def process_requirements(index,base_name,current_path,bounds,mu,Sigma,req_type,variable_lbls,
                         threshold,LHS_MCI_file,req_index,server,DOE_inputs,outputs,
                         plt,resolution=20,plot_R_space=False,new_LHS_MCI=False,compute_margins=True,
                         plot_index=4,plot_2D=False):
    
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
        
        fig_name = '%i_req_%i_%s_RS_pi_%i.png' %(index,req_index,base_name[1],plot_index)
        fig_file_name = os.path.join(current_path,'design_margins',fig_name)
        fig.savefig(fig_file_name, bbox_inches='tight')
    
    if compute_margins:
        return resiliance, R_volume, capability, Buffer, Excess
    else:
        return resiliance 

#==============================================================================#
# MAIN FILE
def Design_margin_evaluation(concept, permutation_index, run_base, run_nominal,
                            new_LHS_MCI, index, plot_index=4,plot_2D=False,req_indices=None):
    
    # plt.rc('text', usetex=True)

    current_path = os.getcwd() # Working directory of file
    start_time = time.time()
    
    design_variables = []; pi_file = []
    for p_i in permutation_index:
        design_variables += [p_i]
        if p_i != -1:
            pi_file += [p_i]

    LHS_MCI_file = 'LHS_MCI_TH'
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
    
    resolution = 10000 # sampling resolution for capability calculation (must be a square number)!
    threshold = 2.8 # cutoff threshold for capability calculation
    
    DOE_folder = 'Thermal_DOE_results'; base_name = ['DOE_th_inputs','thermal_out']
    variable_lbls_pc = ['T1','T2','T3','T4']
    variable_lbls = ['$T_1$ ($^o$C)','$T_2$ ($^o$C)','$T_3$ ($^o$C)','$T_4$ ($^o$C)']
    
    # choose type of output [s,N,n_f,U]
    # location [outer casing, outer weld roots, inner weld roots]
    metric = [0,2]
    
    # Train surrogate model for use in subsequent capability calculation and plotting
    [server,DOE_inputs,outputs] = postprocess_DOE(index,base_name,current_path,bounds_th,variable_lbls_pc,design_variables,concept,metric,True)
    
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
        
        resiliance_th,R_volume_th,capability_th,buffer_th, excess_th = process_requirements(
            index, ['DOE_th_inputs','thermal_out_nominal'], current_path,bounds_th,
            mu,Sigma,req_type,variable_lbls,threshold,LHS_MCI_file,req_index,server,DOE_inputs,outputs,plt,
            new_LHS_MCI=new_LHS_MCI,resolution=resolution,plot_R_space=True,plot_index=plot_index,plot_2D=plot_2D)

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
    
    plt.show()

    #--------------------------------------------------------------------------#
    # Get volume results
    print("------------------------- %s -------------------------\n" %('VOLUME'))
    filename = "%i_volume_out_file.log" %(index)
    body_full_name = os.path.join(current_path,'Job_results','Results_log_EX',filename)
     
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
    # Write results summary

    design_data = [volume*density, resiliance_th_vec_nominal, R_volume_th_vec_nominal, 
        capability_th_vec_nominal, buffer_th_vec_nominal, excess_th_vec_nominal]

    return design_data

#------------------------------------------------------------------------------#
# %% MAIN FILE
def main():
    
    current_path = os.getcwd() # Working directory of file
    
    #============================ PERMUTATIONS ================================#
    
    # one-liner to read a single variable
    P_analysis = loadmat('DOE_permutations.mat')['P_analysis']
    
    run_base = 0; run_nominal = 1; new_LHS = False; process_DOE_requirements = False; sampling = 'fullfact'
    new_LHS_MCI = True; plot_index = 4

    # %% Processing
    #============================= MAIN EXECUTION =================================#

    ##############################
    P_i = P_analysis[109 - 1]
    index = 109
    ##############################
    print(P_i)
    concept = P_i[0]
    permutation_index = P_i[1::]
    
    design_data = Design_margin_evaluation(concept, permutation_index, run_base, run_nominal, new_LHS_MCI, index, plot_index=plot_index, plot_2D=False)

    # %% Postprocessing
    [weight, resiliance_th_vec_nominal, R_volume_th_vec_nominal, 
        capability_th_vec_nominal, buffer_th_vec_nominal, excess_th_vec_nominal] = design_data

    out_titles = ['weight','resiliance_th_uni','resiliance_th_gau','R_volume_th_uni','R_volume_th_gau',
                    'capability_th_uni','capability_th_gau','buffer_th_uni','buffer_th_gau','excess_th_uni','excess_th_gau']
    out_data = [weight,]

    #========================== OUTPUT VARIABLES LOG ==============================#
    filename = "varout_opt_log.log"
    full_filename = os.path.join(current_path,'design_margins',filename)

    resultsfile=open(full_filename,'w')
    resultsfile.write('index'+','+'concept'+','+'i1'+','+'i2'+','+'i3'+','+'i4'+','+'i5'+','
                        +','.join(out_titles)+'\n')
    
    resultsfile=open(full_filename,'a+')
    resultsfile.write(str(index)+','+str(concept)+','+','.join(map(str,permutation_index))+','
                        +','.join(map(str,out_data))+','
                        +','.join(map(str,resiliance_th_vec_nominal))+','
                        +','.join(map(str,R_volume_th_vec_nominal))+','
                        +','.join(map(str,capability_th_vec_nominal))+','
                        +','.join(map(str,buffer_th_vec_nominal))+','
                        +','.join(map(str,excess_th_vec_nominal))+'\n')
    resultsfile.close()
    
    print("\n--------------------------------------------------------------------------------\n")

if __name__ == '__main__':
    main()