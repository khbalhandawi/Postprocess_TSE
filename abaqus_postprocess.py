# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 12:35:50 2017

@author: Khalil
"""
#------------------------------------------------------------------------------#
# %% TEXT FILE READ ELEMENT RESULTS MACRO (AM Case)
def text_read_element_AM( filename ):
	import numpy as np
	# Read data from history output results
	fileID = open(filename,'r'); # Open file
	InputText = np.loadtxt(fileID,
						   delimiter = '\n',
						   dtype=np.str); # \n is the delimiter
	#Header separator mark
	header_border = '---------------------------';
	HeaderLines = np.array([], dtype=str) # Initialize empty header titles object
	HeaderIndex = np.array([], dtype=int) # Initialize empty header index object
	for n,line in enumerate(InputText): # Read line by line
		#Look for title after header mark
		if line == header_border:
			s = np.array([InputText[n+1]]) # Title object
			s2 = np.array(n, dtype=int) # Line number of title
			# Concat final result to Headerlines object
			HeaderLines = np.append(HeaderLines, s, axis=0)
			HeaderIndex = np.append(HeaderIndex, [s2], axis=0)
	
	set_name = []
	time_table = [] 
	stress_table_max = []; stress_table_min = []; stress_table_avg = []
	temp_table_max = []; temp_table_min = []; temp_table_avg = []
	S11_table_max = []; S11_table_min = []; S11_table_avg = []
	S33_table_max = []; S33_table_min = []; S33_table_avg = []
			
	for H in range(len(HeaderLines)): # Blocks to read and tabulate (all selected)
		T = 0; # line counter (inside header block)
		# Read lines in range between two header blocks +1 and -1 from end
		#-------------------------------------------------- >>> n starts here
		table_T = np.array([], dtype=float) # Initialize empty TIME table (block)
		table_S_max = np.array([], dtype=float) # Initialize empty STRESS table (block)
		table_S_min = np.array([], dtype=float) # Initialize empty STRESS table (block)
		table_S_avg = np.array([], dtype=float) # Initialize empty STRESS table (block)
		table_TEMP_max = np.array([], dtype=float) # Initialize empty TEMPERATURE table (block)
		table_TEMP_min = np.array([], dtype=float) # Initialize empty TEMPERATURE table (block)
		table_TEMP_avg = np.array([], dtype=float) # Initialize empty TEMPERATURE table (block)
		table_S11_max = np.array([], dtype=float) # Initialize empty S11 table (block)
		table_S11_min = np.array([], dtype=float) # Initialize empty S11 table (block)
		table_S11_avg = np.array([], dtype=float) # Initialize empty S11 table (block)
		table_S33_max = np.array([], dtype=float) # Initialize empty S33 table (block)
		table_S33_min = np.array([], dtype=float) # Initialize empty S33 table (block)
		table_S33_avg = np.array([], dtype=float) # Initialize empty S33 table (block)
		if H < len(HeaderLines)-2: # Dont exceed index range (-1: normalized time)
			for nn in range(int(HeaderIndex[H]+1), int(HeaderIndex[H+1]), 1):
				T += 1
				line = InputText[nn].split(",")
				if len(line) > 1:
					# time,stress,temperature,SP1,SP3
					table_T = np.append(table_T, float(line[0])) # Return TIME
					table_S_max = np.append(table_S_max, float(line[1])) # Return STRESS
					table_S_min = np.append(table_S_min, float(line[2])) # Return STRESS
					table_S_avg = np.append(table_S_avg, float(line[3])) # Return STRESS
					table_TEMP_max = np.append(table_TEMP_max, float(line[4])) # Return TEMP
					table_TEMP_min = np.append(table_TEMP_min, float(line[5])) # Return TEMP
					table_TEMP_avg = np.append(table_TEMP_avg, float(line[6])) # Return TEMP
					table_S33_max = np.append(table_S33_max, float(line[7])) # Return S33
					table_S33_min = np.append(table_S33_min, float(line[8])) # Return S33
					table_S33_avg = np.append(table_S33_avg, float(line[9])) # Return S33
					table_S11_max = np.append(table_S11_max, float(line[10])) # Return S11
					table_S11_min = np.append(table_S11_min, float(line[11])) # Return S11
					table_S11_avg = np.append(table_S11_avg, float(line[12])) # Return S11
		elif H < len(HeaderLines)-1:
			for nn in range(int(HeaderIndex[H]+1), int(HeaderIndex[H+1]), 1):
				time_norm = InputText[nn]

		# Append each element set to multi-dimensional array 
		set_name += [HeaderLines[H]]
		time_table += [table_T]
		stress_table_max += [table_S_max]; stress_table_min += [table_S_min]; stress_table_avg += [table_S_avg]
		temp_table_max += [table_TEMP_max]; temp_table_min += [table_TEMP_min]; temp_table_avg += [table_TEMP_avg]
		S11_table_max += [table_S11_max]; S11_table_min += [table_S11_min]; S11_table_avg += [table_S11_avg]
		S33_table_max += [table_S33_max]; S33_table_min += [table_S33_min]; S33_table_avg += [table_S33_avg]
			
	fileID.close() #close file
	
	output = [time_table,stress_table_max,stress_table_min,stress_table_avg,
			temp_table_max,temp_table_min,temp_table_avg,
			S11_table_max,S11_table_min,S11_table_avg,
			S33_table_max,S33_table_min,S33_table_avg,
			set_name,time_norm]
	
	return output

#------------------------------------------------------------------------------#
# %% TEXT FILE READ NODAL RESULTS MACRO (AM Case)
#------------------------------------------------------------------------------#
# %% TEXT FILE READ NODAL RESULTS MACRO (AM Case)
def text_read_nodal( filename ):
	import numpy as np
	# Read data from history output results
	fileID = open(filename,'r'); # Open file
	InputText = np.loadtxt(fileID,
						   delimiter = '\n',
						   dtype=np.str); # \n is the delimiter
	#Header separator mark
	header_border = '---------------------------';
	HeaderLines = np.array([], dtype=str) # Initialize empty header titles object
	HeaderIndex = np.array([], dtype=int) # Initialize empty header index object
	for n,line in enumerate(InputText): # Read line by line
		#Look for title after header mark
		if line == header_border:
			s = np.array([InputText[n+1]]) # Title object
			s2 = np.array(n, dtype=int) # Line number of title
			# Concat final result to Headerlines object
			HeaderLines = np.append(HeaderLines, s, axis=0)
			HeaderIndex = np.append(HeaderIndex, [s2], axis=0)
	
	set_name = []
	time_table = [] 
	U_table_max = []; U_table_min = []; U_table_avg = []
			
	for H in range(len(HeaderLines)): # Blocks to read and tabulate (all selected)
		T = 0; # line counter (inside header block)
		# Read lines in range between two header blocks +1 and -1 from end
		#-------------------------------------------------- >>> n starts here
		table_T = np.array([], dtype=float) # Initialize empty TIME table (block)
		table_U_max = np.array([], dtype=float) # Initialize empty U table (block)
		table_U_min = np.array([], dtype=float) # Initialize empty U table (block)
		table_U_avg = np.array([], dtype=float) # Initialize empty U table (block)
		if H < len(HeaderLines)-2: # Dont exceed index range (-1: normalized time)
			for nn in range(int(HeaderIndex[H]+1), int(HeaderIndex[H+1]), 1):
				T += 1
				line = InputText[nn].split(",")
				if len(line) > 1:
					# time,U
					table_T = np.append(table_T, float(line[0])) # Return TIME
					table_U_max = np.append(table_U_max, float(line[1])) # Return U
					table_U_min = np.append(table_U_min, float(line[2])) # Return U
					table_U_avg = np.append(table_U_avg, float(line[3])) # Return U
		elif H < len(HeaderLines)-1:
			for nn in range(int(HeaderIndex[H]+1), int(HeaderIndex[H+1]), 1):
				time_norm = InputText[nn]

		# Append each element set to multi-dimensional array 
		set_name += [HeaderLines[H]]
		time_table += [table_T]
		U_table_max += [table_U_max]; U_table_min += [table_U_min]; U_table_avg += [table_U_avg]
			
	fileID.close() #close file
	
	outputs = [time_table,U_table_max,U_table_min,U_table_avg,time_norm]
	
	return outputs

def gridsamp(bounds, q):
	import numpy as np
	#===========================================================================
	# GRIDSAMP  n-dimensional grid over given range
	# 
	#  Call:    S = gridsamp(bounds, q)
	# 
	#  bounds:  2*n matrix with lower and upper limits
	#  q     :  n-vector, q(j) is the number of points
	#           in the j'th direction.
	#           If q is a scalar, then all q(j) = q
	#  S     :  m*n array with points, m = prod(q)
	# 
	#  hbn@imm.dtu.dk  
	#  Last update June 25, 2002
	#===========================================================================
	
	[mr,n] = np.shape(bounds);    dr = np.diff(bounds, axis=0)[0]; # difference across rows
	if  mr != 2 or any([item < 0 for item in dr]):
	  raise Exception('bounds must be an array with two rows and bounds(1,:) <= bounds(2,:)')
	 
	if  q.ndim > 1 or any([item <= 0 for item in q]):
	  raise Exception('q must be a vector with non-negative elements')
	
	p = len(q);   
	if  p == 1:
		q = np.tile(q, (1, n))[0]; 
	elif  p != n:
	  raise Exception(sprintf('length of q must be either 1 or %d',n))
	 
	
	# Check for degenerate intervals
	i = np.where(dr == 0)[0]
	if  i.size > 0:
		q[i] = 0*q[i]; 
	
	# Recursive computation
	if  n > 1:
		A = gridsamp(bounds[:,1::], q[1::]);  # Recursive call
		[m,p] = np.shape(A);
		q = q[0];
		S = np.concatenate((np.zeros((m*q,1)), np.tile(A, (q, 1))),axis=1);
		y = np.linspace(bounds[0,0],bounds[1,0], q);
		
		k = range(m);
		for i in range(q):
			aug = np.tile(y[i], (m, 1))
			aug = np.reshape(aug, S[k,0].shape)
			
			S[k,0] = aug;
			k = [item + m for item in k];
	else:
		S = np.linspace(bounds[0,0],bounds[1,0],q[-1])
		S = np.transpose([S])
		
	return S

def get_SGTE_model(out_file):
	import os
	import numpy as np
	
	current_dir = os.getcwd()
	filepath = os.path.join(current_dir,'SGTE_matlab_server',out_file)
	
	# Get matrices names from the file
	NAMES = [];
	fileID = open(filepath,'r'); # Open file
	InputText = np.loadtxt(fileID,
					   delimiter = '\n',
					   dtype=np.str); # \n is the delimiter

	for n,line in enumerate(InputText): # Read line by line
		#Look for object
		if line.find('Surrogate: ') != -1:
			i = line.find(': ')
			line = line[i+2::];
			NAMES += [line];
	
	fileID.close()
	model = NAMES[0]
	
	return model

def define_SGTE_model(fit_type,run_type):
	import os
	fitting_types = [0,1,2,3,4,5]
	fitting_names = ['KRIGING','LOWESS','KS','RBF','PRS','ENSEMBLE'];
	out_file = '%s.sgt' %fitting_names[fit_type]
	if os.path.exists(out_file):
		os.remove(out_file)
    	
	budget = 200;
	
	if run_type == 1: # optimize fitting parameters
		if fit_type == fitting_types[0]:
				model = ("TYPE KRIGING RIDGE OPTIM DISTANCE_TYPE OPTIM METRIC "
						 "OECV BUDGET %i OUTPUT %s" %(budget, out_file))
		elif fit_type == fitting_types[1]:
				model = ("TYPE LOWESS DEGREE OPTIM RIDGE OPTIM "
						 "KERNEL_TYPE OPTIM KERNEL_COEF OPTIM "
						 "DISTANCE_TYPE OPTIM METRIC OECV BUDGET %i OUTPUT %s" %(budget, out_file))
		elif fit_type == fitting_types[2]:
				model = ("TYPE KS KERNEL_TYPE OPTIM KERNEL_COEF OPTIM "
						 "DISTANCE_TYPE OPTIM METRIC OECV BUDGET %i OUTPUT %s" %(budget, out_file))
		elif fit_type == fitting_types[3]:
				model = ("TYPE RBF KERNEL_TYPE OPTIM KERNEL_COEF OPTIM "
						 "DISTANCE_TYPE OPTIM RIDGE OPTIM METRIC OECV BUDGET %i OUTPUT %s" %(budget, out_file))
		elif fit_type == fitting_types[4]:
				model = ("TYPE PRS DEGREE OPTIM RIDGE OPTIM "
						 "METRIC OECV BUDGET %i OUTPUT %s" %(budget, out_file))
		elif fit_type == fitting_types[5]:
				model = ("TYPE ENSEMBLE WEIGHT OPTIM METRIC OECV "
						 "DISTANCE_TYPE OPTIM BUDGET %i OUTPUT %s" %(budget, out_file))
	elif run_type == 2: # Run existing SGTE model
		model = get_SGTE_model(out_file);
		
	return model,out_file

def plot_countour_code(q,bounds,bounds_n,bounds_req,mu,Sigma,req_type,lob,upb,LHS_f,d,
					   nominal,threshold,nn,S,server,variable_lbls,gs,plt,fig):
	
	from DED_Blackbox_opt import scaling
	from DED_Blackbox_opt import multivariate_gaussian
	import matplotlib.patches as patches
	import numpy as np
	
	iteraton = -1;
	for par in q:
		iteraton += 1;
		# Plot points
		i = par; # plot variable indices
		bounds_p = np.zeros(bounds.shape);
		bounds_p_n = np.zeros(bounds_n.shape);
		nn_vec = nn*np.ones(len(LHS_f[0,:]),dtype=int);
		fixed_value_lc = np.zeros((d,));
		for n in range(len(bounds)):
		    if n not in i:
		        lm = nominal[n];
		        fixed_value = scaling(lm,lob[n],upb[n],2); # Assign lambdas
		        
		        bounds_p[n,0] = fixed_value-0.0000001; # Set bounds equal to each other
		        bounds_p[n,1] = fixed_value+0.0000001; # Set bounds equal to each other
		        bounds_p_n[n,0] = lm; # Nomalized bounds
		        bounds_p_n[n,1] = lm+0.01; # Nomalized bounds
		        nn_vec[n] = 1;
		        
		        fixed_value_lc[n] = scaling(lm,lob[n],upb[n],2); # Assign lambdas
		    else:
		        bounds_p[n,0] = bounds[n,0]
		        bounds_p[n,1] = bounds[n,1]
		        bounds_p_n[n,0] = bounds_n[n,0]
		        bounds_p_n[n,1] = bounds_n[n,1]
		
		X = gridsamp(bounds_p_n.T, nn_vec);
		# Prediction
		# YX = sm.predict_values(X)
		[YX, std, ei, cdf] = server.sgtelib_server_predict(X)
		
	#========================= DATA VISUALIZATION =============================#
	# %% Sensitivity plots
		YX_obj = YX[:,0];
		X = X[:,i];
		X1_norm = np.reshape(X[:,0],(nn,nn)); X2_norm = np.reshape(X[:,1],(nn,nn));
		X1 = scaling(X1_norm, lob[i[0]], upb[i[0]], 2); # Scale up plot variable
		X2 = scaling(X2_norm, lob[i[1]], upb[i[1]], 2); # Scale up plot variable
		YX_obj = np.reshape(YX_obj, np.shape(X1));
		
		cmax = 100000; cmin = 0; # set colorbar limits
		# cmax = 4.4; cmin = 1.2; # set colorbar limits
		
		ax = fig.add_subplot(gs[iteraton]); # subplot
		cf = ax.contourf( X1, X2, YX_obj, cmap=plt.cm.jet); # plot contour
		# cf = ax.contourf( X1, X2, YX_obj, vmin = cmin, vmax = cmax, cmap=plt.cm.jet); # plot contour
		ax.contour(cf, colors='k')
		
		cbar = plt.cm.ScalarMappable(cmap=plt.cm.jet)
		cbar.set_array(YX_obj)
		# cbar.set_clim(cmin, cmax)
		fig.colorbar(cbar)
		# fig.colorbar(cbar, boundaries=np.linspace(cmin, cmax, 10))
		
		artists, labels = cf.legend_elements()
		af = artists[0];
		
	#======================== NONLINEAR CONSTRAINTS ============================#	
	# %% Nonlinear constraints
		YX_cstr = YX - threshold;
		YX_cstr = np.reshape(YX_cstr, np.shape(X1));
		c1 = ax.contourf( X1, X2, YX_cstr, levels=[-20, 0, 20], colors=['#FF0000','None'], 
						  hatches=['//', None], alpha=0.0);
		ax.contour(c1, colors='r', linewidths = 2.0)
		a1 = patches.Rectangle((20,20), 20, 20, linewidth=0, color='#FF0000', fill='#FF0000', hatch='//')
		
	#====================== REQUIREMENTS CONSTRAINTS ==========================#	
	# %% Requirements
		
		if req_type == 'guassian':
			
			# Mean vector and covariance matrix
			mu_sp = np.array([ mu[i[0]], mu[i[1]] ])
			Sigma_sp = Sigma[np.ix_([i[0],i[1]],[i[0],i[1]])] # pick corresponding row and column
			Sigma_sp = Sigma_sp
			
			# Pack X1 and X2 into a single 3-dimensional array
			pos = np.empty(X1_norm.shape + (2,))
			pos[:, :, 0] = X1_norm
			pos[:, :, 1] = X2_norm
			
			Z = multivariate_gaussian(pos, mu_sp, Sigma_sp)
			
			# 1,2,3 Sigma level contour
			L = [];
			for n in range(3):
				# Pack X and Y into a single 3-dimensional array
				pos = np.empty((1,1) + (2,))
				x_l = [ mu_sp[0] + ((n+1) * np.sqrt(Sigma_sp[0,0])), # evaluate at Sigma not Sigma^2
						mu_sp[1]                                   ]
				
				level_index = 0;
				for value in x_l:
					pos[:, :, level_index] = value
					level_index += 1
					
				LN = multivariate_gaussian(pos, mu_sp, Sigma_sp)
				L += [LN]

			# c2 = ax.contourf( X1, X2, Z, colors=['#39FFF2'], alpha=0.0);
			c2 = ax.contourf( X1, X2, Z, alpha=0.55, cmap=plt.cm.Blues);
			
			ax.contour(c2, colors='#39FFF2', levels=L, linewidths = 1.0)
			a2 = patches.Rectangle((20,20), 20, 20, linewidth=0, color='#39FFF2', fill='#39FFF2', hatch='..')
			
		elif req_type == 'uniform':	
		
			YX_req = np.zeros(X1.shape)
			
			cond1 = (X1 >= bounds_req[i[0],0]) & (X1 <= bounds_req[i[0],1]) # x-axis
			cond2 = (X2 >= bounds_req[i[1],0]) & (X2 <= bounds_req[i[1],1]) # y-axis
			cond = (cond1) & (cond2)
			
			YX_req[cond] = 1
			
			c2 = ax.contourf( X1, X2, YX_req, levels=[-10, 0, 10], colors=['None','#39FFF2'], 
							  hatches=[None, '..'], alpha=0.0);
			ax.contour(c2, colors='#39FFF2', linewidths = 2.0)
			a2 = patches.Rectangle((20,20), 20, 20, linewidth=0, color='#39FFF2', fill='#39FFF2', hatch='..')
			
	#========================= REGION OF INTEREST =============================#	
	# %% Requirements
		YX_req = np.zeros(X1.shape)

		cond1 = (X1 >= bounds_req[i[0],0]) & (X1 <= bounds_req[i[0],1]) # x-axis
		cond2 = (X2 >= bounds_req[i[1],0]) & (X2 <= bounds_req[i[1],1]) # y-axis
		cond3 = (YX_cstr >= 0) # z-axis
		cond = (cond1) & (cond2) & (cond3)
		
		YX_req[cond] = 1
		
		c3 = ax.contourf( X1, X2, YX_req, levels=[-10, 0, 10], colors=['None','#6AFF39'], 
						  hatches=[None, '+'], alpha=0.0);
		ax.contour(c3, colors='#6AFF39', linewidths = 2.0)
		
		#=======================================================================
		# artists, labels = c3.legend_elements()
		# a3 = artists[1];
		#=======================================================================
		a3 = patches.Rectangle((20,20), 20, 20, linewidth=0, color='#6AFF39', fill='#6AFF39', hatch='+')
		
	#============================ AXIS LABELS =================================#	
		ax.axis([lob[i[0]],upb[i[0]],lob[i[1]],upb[i[1]]]) # fix the axis limits
		
		ax.plot(S[:,i[0]],S[:,i[1]],'.k')
		ax.set_xlabel(variable_lbls[i[0]])
		ax.set_ylabel(variable_lbls[i[1]])
	
	handles, labels = [[af,a1,a2,a3], ["$F(\mathbf{T})$", 
									   "$\{C^\prime:F(\mathbf{T})<F_{th}\}$", 
								   	   "$R$","${C}\cap{R}$"]]
	fig.legend(handles, labels, loc='upper center', ncol=4, fontsize = 14)
		
def plot_line_code(bounds,bounds_n,bounds_req,mu,Sigma,req_type,lob,upb,LHS_f,d,threshold,
				   nn,S,Y,server,variable_lbls,gs,plt,fig):
	
	from DED_Blackbox_opt import scaling
	from DED_Blackbox_opt import multivariate_gaussian
	import numpy as np
	import copy
	
	# Plot points
	bounds_p = np.zeros(bounds.shape);
	bounds_p_n = np.zeros(bounds_n.shape);
	nn_vec = (nn**2)*np.ones(len(LHS_f[0,:]),dtype=int);
	fixed_value_lc = np.zeros((d,));
	n = 0;
	bounds_p[n,0] = bounds[n,0]
	bounds_p[n,1] = bounds[n,1]
	bounds_p_n[n,0] = bounds_n[n,0]
	bounds_p_n[n,1] = bounds_n[n,1]
	
	X = gridsamp(bounds_p_n.T, nn_vec);
	
	# Prediction
	[YX, std, ei, cdf] = server.sgtelib_server_predict(X)
	#========================= DATA VISUALIZATION =============================#
	# %% Sensitivity plots
	
	YX_obj = YX[:,0];
	X_norm = X[:,0];
	X1 = scaling(X_norm, lob[0], upb[0], 2); # Scale up plot variable
	YX_obj = np.reshape(YX_obj, np.shape(X1));
	
	ax = fig.add_subplot(gs[0]); # subplot
	cf = ax.plot( X1, YX_obj, '-g' ); # plot line
	ax.plot( S[:,0], Y[:,0], '.k' )
	
	#====================== REQUIREMENTS CONSTRAINTS ==========================#	
	# %% Requirements

	if req_type == 'uniform':
		cond = (X1 >= bounds_req[0,0]) & (X1 <= bounds_req[0,1]) # x-axis
		YX_req = np.ones(X1.shape) * min(YX_obj)
		YX_req[cond] = max(YX_obj);
		
	elif req_type == 'guassian':
		
		#=======================================================================
		# mu = mu[0]
		# Sigma = Sigma[0][0];
		# YX_req = ( 1 / ( Sigma * np.sqrt(2*np.pi) ) ) * np.exp(-0.5 * ( ( (X - mu) / (Sigma) ) ** 2 ) )
		# YX_req = scaling( YX_req / max(YX_req), min(YX_obj), max(YX_obj), 2)
		#=======================================================================
		res_sq = np.ceil(nn).astype(int) # size of equivalent square matrix
		pos = np.empty((res_sq,res_sq) + (len(lob),))
		  
		for i in range(len(lob)):
			X_norm = np.reshape(X[:,i],(res_sq,res_sq));
			# Pack X1, X2 ... Xk into a single 3-dimensional array
			pos[:, :, i] = X_norm
 
		Z = multivariate_gaussian(pos, mu, Sigma)
		Z = np.reshape(Z, np.shape(X)[0]);
		YX_req = scaling( Z / max(Z), min(YX_obj), max(YX_obj), 2)
		
	ax.plot( X1, YX_req, '-b', linewidth=2.0 )
	# ax.fill_between(X1, min(YX_obj), YX_req, facecolor="none", hatch="//", edgecolor="b", linewidth=0.0)
	
	#======================== NONLINEAR CONSTRAINTS ============================#	
	# %% Nonlinear constraints
	
	cond = YX_obj - threshold < 0
	YX_cstr = copy.deepcopy(YX_obj);
	YX_cstr[cond] = 0.0;
	
	ax.fill_between(X1[cond], min(YX_obj), YX_req[cond], facecolor="none", hatch="XX", edgecolor="r", linewidth=0.0)
	
	#============================ AXIS LABELS =================================#
	# ax.set_ylim([min(YX_obj),max(YX_obj)]) # fix the axis limits
	# ax.set_ylim([0,100000]) # fix the axis limits
	# ax.set_ylim([2.80,3.75]) # fix the axis limits
	
	ax.set_xlabel(variable_lbls[0])
	ax.set_ylabel('$n_{safety}$')

def hyperplane_SGTE_vis_norm(server,LHS_f,bounds,bounds_req,mu,Sigma,req_type,variable_lbls,
							 nominal,threshold,objs,nn,fig,plt):
	import numpy as np
	from scipy.special import comb
	from DED_Blackbox_opt import scaling
	from itertools import combinations
	import matplotlib.pyplot as plt
	import matplotlib.gridspec as gridspec
	from matplotlib import cm
	from shutil import copyfile
	import os
	
	lob = bounds[:,0]
	upb = bounds[:,1]
	
	Y = objs; S = LHS_f;
	
	lob_n = np.zeros(np.size(lob)); upb_n = np.ones(np.size(upb)); 
	bounds_n = np.zeros(np.shape(bounds))
	bounds_n[:,0] = lob_n; bounds_n[:,1] = upb_n; 
	
	#======================== 2D GRID CONSTRUCTION ============================#
	# %% Activate 2 out 4 variables up to 4 variables
	d = len(LHS_f[0,:]); #<-------- Number of variables
	if d > 4: # maximum of four variables allowed
	    d = 4;
	    
	if d == 1:
		sp_shape = [1,1]; ax_h = -0.08; ax_bot = 0; ax_left = 0.0; ann_pos = 0.45;    #<-------- Edit as necessary to fit figure properl
		fig.set_figheight(3); fig.set_figwidth(4.67)
	elif d == 2:
		sp_shape = [1,1]; ax_h = -0.08; ax_bot = 0; ax_left = 0.0; ann_pos = 0.45;    #<-------- Edit as necessary to fit figure properl
		fig.set_figheight(3); fig.set_figwidth(4.67)
	elif d == 3:
		sp_shape = [1,3]; ax_h = -0.08; ax_bot = 0; ax_left = 0.0; ann_pos = 0.45;    #<-------- Edit as necessary to fit figure properl
		fig.set_figheight(2.9); fig.set_figwidth(15)
	elif d == 4:
		sp_shape = [2,3]; ax_h = 0.1; ax_bot = 0.08; ax_left = 0.0; ann_pos = 0.45;   #<-------- Edit as necessary to fit figure properly
		fig.set_figheight(5.8); fig.set_figwidth(15)
	
	gs = gridspec.GridSpec(sp_shape[0],sp_shape[1], 
						width_ratios = np.ones(sp_shape[1],dtype=int), 
						height_ratios = np.ones(sp_shape[0],dtype=int))
	
	
	q = combinations(range(d),2); # choose 2 out d variables
	ss = comb(d,2,exact=True)
	
	if d != 1:
		plot_countour_code(q,bounds,bounds_n,bounds_req,mu,Sigma,req_type,lob,upb,LHS_f,d,
						   nominal,threshold,nn,S,server,variable_lbls,gs,plt,fig)
	else:
		plot_line_code(bounds,bounds_n,bounds_req,mu,Sigma,req_type,lob,upb,LHS_f,d,
					   threshold,nn,S,Y,server,variable_lbls,gs,plt,fig)

	#===========================================================================
	# copyfile(sgt_file, os.path.join(os.getcwd(),'SGTE_matlab_server',sgt_file)) # backup hyperparameters
	#===========================================================================

def parallel_plots(input_data,output_data,in_data_labels,out_data_labels,plot_filename):
	import plotly.graph_objs as go # Used to create data structure for plotly
	from plotly.offline import plot, iplot, init_notebook_mode # Import offline plot functions from plotly
	
	#=============================================================================#
	#Create Data Structure
	
	data_list = []
	for in_data,in_label in zip(input_data.T,in_data_labels):
		data_list += [dict(label = in_label, values = in_data, tickformat=".2f%")]
	
	for out_data,out_label in zip(output_data.T,out_data_labels):
		data_list += [dict(label = out_label, values = out_data, tickformat=".2f%")]

	data_pd = [
	    go.Parcoords(
	        line = dict(color = input_data[:,0],
						colorscale = 'Jet',#'[[0,'blue'],[0.5,'green'],[1,'red']],
				   		showscale = True),
	        dimensions = data_list
	    )
	]
	#fig = go.Figure(data = data_pd) 
	#fig.show()
	plot(data_pd, show_link=False, filename = plot_filename, auto_open=False)
	
#------------------------------------------------------------------------------#
# %% TEXT FILE READ LOAD CASE MACRO
#------------------------------------------------------------------------------#
# %% TEXT FILE READ LOAD CASE MACRO
def text_read_loadcase( filename ):
	import numpy as np
	# Read data from history output results
	fileID = open(filename,'r'); # Open file
	InputText = np.loadtxt(fileID,
						   delimiter = '\n',
						   dtype=np.str); # \n is the delimiter
	#Header separator mark
	header_border = '---------------------------';
	HeaderLines = np.array([], dtype=str) # Initialize empty header titles object
	HeaderIndex = np.array([], dtype=int) # Initialize empty header index object
	for n,line in enumerate(InputText): # Read line by line
		#Look for title after header mark
		if line == header_border:
			s = np.array([InputText[n+1]]) # Title object
			s2 = np.array(n, dtype=int) # Line number of title
			# Concat final result to Headerlines object
			HeaderLines = np.append(HeaderLines, s, axis=0)
			HeaderIndex = np.append(HeaderIndex, [s2], axis=0)
	
	set_name = []
	time_table = [] 
	stress_table_max = []; stress_table_min = []; stress_table_avg = []
	temp_table_max = []; temp_table_min = []; temp_table_avg = []
	S_hoop_table_max = []; S_hoop_table_min = []; S_hoop_table_avg = []
	S_M_table_max = []; S_M_table_min = []; S_M_table_avg = []
	S_A_table_max = []; S_A_table_min = []; S_A_table_avg = []
	N_table_max = []; N_table_min = []; N_table_avg = []
	n_f_table_max = []; n_f_table_min = []; n_f_table_avg = []
			
	for H in range(len(HeaderLines)): # Blocks to read and tabulate (all selected)
		T = 0; # line counter (inside header block)
		# Read lines in range between two header blocks +1 and -1 from end
		#-------------------------------------------------- >>> n starts here
		table_T = np.array([], dtype=float) # Initialize empty TIME table (block)
		table_S_max = np.array([], dtype=float) # Initialize empty STRESS table (block)
		table_S_min = np.array([], dtype=float) # Initialize empty STRESS table (block)
		table_S_avg = np.array([], dtype=float) # Initialize empty STRESS table (block)
		table_TEMP_max = np.array([], dtype=float) # Initialize empty TEMPERATURE table (block)
		table_TEMP_min = np.array([], dtype=float) # Initialize empty TEMPERATURE table (block)
		table_TEMP_avg = np.array([], dtype=float) # Initialize empty TEMPERATURE table (block)
		table_S_hoop_max = np.array([], dtype=float) # Initialize empty S_hoop table (block)
		table_S_hoop_min = np.array([], dtype=float) # Initialize empty S_hoop table (block)
		table_S_hoop_avg = np.array([], dtype=float) # Initialize empty S_hoop table (block)
		table_S_M_max = np.array([], dtype=float) # Initialize empty S_M table (block)
		table_S_M_min = np.array([], dtype=float) # Initialize empty S_M table (block)
		table_S_M_avg = np.array([], dtype=float) # Initialize empty S_M table (block)
		table_S_A_max = np.array([], dtype=float) # Initialize empty S_A table (block)
		table_S_A_min = np.array([], dtype=float) # Initialize empty S_A table (block)
		table_S_A_avg = np.array([], dtype=float) # Initialize empty S_A table (block)
		table_N_max = np.array([], dtype=float) # Initialize empty N table (block)
		table_N_min = np.array([], dtype=float) # Initialize empty N table (block)
		table_N_avg = np.array([], dtype=float) # Initialize empty N table (block)
		table_n_f_max = np.array([], dtype=float) # Initialize empty n_f table (block)
		table_n_f_min = np.array([], dtype=float) # Initialize empty n_f table (block)
		table_n_f_avg = np.array([], dtype=float) # Initialize empty n_f table (block)
		if H < len(HeaderLines)-2: # Dont exceed index range (-1: normalized time)
			for nn in range(int(HeaderIndex[H]+1), int(HeaderIndex[H+1]), 1):
				T += 1
				line = InputText[nn].split(",")
				if len(line) > 1:
					# time,stress,temperature,SP1,SP3
					table_T = np.append(table_T, float(line[0])) # Return TIME
					table_S_max = np.append(table_S_max, float(line[1])) # Return STRESS
					table_S_min = np.append(table_S_min, float(line[2])) # Return STRESS
					table_S_avg = np.append(table_S_avg, float(line[3])) # Return STRESS
					table_TEMP_max = np.append(table_TEMP_max, float(line[4])) # Return TEMP
					table_TEMP_min = np.append(table_TEMP_min, float(line[5])) # Return TEMP
					table_TEMP_avg = np.append(table_TEMP_avg, float(line[6])) # Return TEMP
					table_S_hoop_max = np.append(table_S_hoop_max, float(line[7])) # Return S_hoop
					table_S_hoop_min = np.append(table_S_hoop_min, float(line[8])) # Return S_hoop
					table_S_hoop_avg = np.append(table_S_hoop_avg, float(line[9])) # Return S_hoop
					table_S_M_max = np.append(table_S_M_max, float(line[10])) # Return S_M
					table_S_M_min = np.append(table_S_M_min, float(line[11])) # Return S_M
					table_S_M_avg = np.append(table_S_M_avg, float(line[12])) # Return S_M
					table_S_A_max = np.append(table_S_A_max, float(line[13])) # Return S_A
					table_S_A_min = np.append(table_S_A_min, float(line[14])) # Return S_A
					table_S_A_avg = np.append(table_S_A_avg, float(line[15])) # Return S_A
					table_N_max = np.append(table_N_max, float(line[16])) # Return N
					table_N_min = np.append(table_N_min, float(line[17])) # Return N
					table_N_avg = np.append(table_N_avg, float(line[18])) # Return N
					table_n_f_max = np.append(table_n_f_max, float(line[19])) # Return n_f
					table_n_f_min = np.append(table_n_f_min, float(line[20])) # Return n_f
					table_n_f_avg = np.append(table_n_f_avg, float(line[21])) # Return n_f
		elif H < len(HeaderLines)-1:
			for nn in range(int(HeaderIndex[H]+1), int(HeaderIndex[H+1]), 1):
				time_norm = InputText[nn]

		# Append each element set to multi-dimensional array 
		set_name += [HeaderLines[H]]
		time_table += [table_T]
		stress_table_max += [table_S_max]; stress_table_min += [table_S_min]; stress_table_avg += [table_S_avg]
		temp_table_max += [table_TEMP_max]; temp_table_min += [table_TEMP_min]; temp_table_avg += [table_TEMP_avg]
		S_hoop_table_max += [table_S_hoop_max]; S_hoop_table_min += [table_S_hoop_min]; S_hoop_table_avg += [table_S_hoop_avg]
		S_M_table_max += [table_S_M_max]; S_M_table_min += [table_S_M_min]; S_M_table_avg += [table_S_M_avg]
		S_A_table_max += [table_S_A_max]; S_A_table_min += [table_S_A_min]; S_A_table_avg += [table_S_A_avg]
		N_table_max += [table_N_max]; N_table_min += [table_N_min]; N_table_avg += [table_N_avg]
		n_f_table_max += [table_n_f_max]; n_f_table_min += [table_n_f_min]; n_f_table_avg += [table_n_f_avg]
		
	fileID.close() #close file
	
	output = [time_table,stress_table_max,stress_table_min,stress_table_avg,
			temp_table_max,temp_table_min,temp_table_avg,
			S_hoop_table_max,S_hoop_table_min,S_hoop_table_avg,
			S_M_table_max,S_M_table_min,S_M_table_avg,
			S_A_table_max,S_A_table_min,S_A_table_avg,
			N_table_max,N_table_min,N_table_avg,
			n_f_table_max,n_f_table_min,n_f_table_avg,
			set_name,time_norm]
	
	return output

#------------------------------------------------------------------------------#
# %% Figure plot macro
def fig_plot( time_table,stress_table,temp_table,S11_table,S33Nble,fig_title,time_norm ):
	import numpy as np
	import matplotlib.pyplot as plt

	time_table = time_table/float(time_norm)

	fig, ax1 = plt.subplots() # Fig definition
	fig.suptitle(fig_title, y=1.08)
	ax1.plot(time_table, S11_table, 'b-', label='Max. Principle Stress ($\sigma_{11}$)') # Axis data
	ax1.hold(True)
	ax1.plot(time_table, S33_table, 'm-', label='Min. Principle Stress ($\sigma_{33}$)') # Axis data
	ax1.plot(time_table, stress_table, 'g--', label='Mises') # Axis data

	ax1.set_xlabel('Normalized Time ${t}/{t_c}$') # x Axis labels
	# Make the y-axis label, ticks and tick labels match the line color.
	ax1.set_ylabel('Stress (MPa)', color='b') # y Axis label
	ax1.tick_params('y', colors='b') # y Axis ticks

	ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.135),
			   ncol=3)

	ax2 = ax1.twinx() # x Axis data
	ax2.plot(time_table, temp_table, 'r-') # Axis data
	ax2.set_ylabel('Temperature ($^o$C)', color='r') # y Axis label
	ax2.tick_params('y', colors='r') # y Axis ticks

	fig.tight_layout()
	#    plt.legend([S33_h, mises_h], ['S33', 'Mises']) # Legend labels

	plt.show()
	title = str.split(fig_title,':') # \n is the delimiter

	eps_name = '%s_%s.eps' %(title[0],title[1])
	print('------------------------------')
	print(eps_name)
	fig.savefig(eps_name, format='eps', dpi=1000,bbox_inches='tight')
    
#------------------------------------------------------------------------------#
# %% MAIN FILE
def main():
	import os
	import numpy as np

	current_path = os.getcwd() # Working directory of file
	wokring_directory = os.path.join(current_path,'Job_results','BACKUP','Vf','Results_log') # print variables as space demilited string
	os.chdir(wokring_directory)

	ext = '1'
	filename = '%s_hist_out_file.log' %(ext)

	[time_table,stress_table,temp_table,S11_table,S33_table,fig_title,time_norm] = text_read_element( filename )

	s_res = np.array([])
	for n in range(len(time_table)-2):
		fig_plot( time_table[n],stress_table[n],temp_table[n],S11_table[n],S33_table[n],fig_title[n],time_norm )
		s_res = np.append(s_res,stress_table[n][-1])
		print(stress_table[n][-1])
	print('The maximum residual stress is: %f' %max(s_res))

	filename = '%s_static_out_file.log' %(ext)
	[time_table,stress_table,temp_table,S11_table,S33_table,fig_title,time_norm] = text_read_static( filename )
	s_res = np.array([])
	for n in range(len(time_table)-2):
		s_res = np.append(s_res,stress_table[n][-1])
	print(s_res)

	filename = '%s_modal_out_file.log' %(ext)
	[mode_table,freq_table,fig_title] = text_read_modal( filename )
	print(freq_table)
    
# Stuff to run when not called in import
if __name__ == "__main__":
	main()