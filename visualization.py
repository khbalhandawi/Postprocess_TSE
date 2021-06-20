# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 12:35:50 2017

@author: Khalil
"""

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

def plot_countour_code(q,bounds,bounds_n,bounds_req,LHS_MCI_file,mu,Sigma,req_type,lob,upb,LHS_f,d,
					   nominal,threshold,nn,S,server,variable_lbls,gs,plt,fig,plot_index=4):
	
	from design_margins import scaling
	from design_margins import multivariate_gaussian
	import matplotlib.patches as patches
	import numpy as np
	import os, pickle
	
	# get LHS_MCI points distribution
	DOE_full_name = LHS_MCI_file + '.pkl'
	DOE_filepath = os.path.join(os.getcwd(),'design_margins',DOE_full_name)
	resultsfile=open(DOE_filepath,'rb')
	dFF = pickle.load(resultsfile)
	dFF_n = pickle.load(resultsfile)
	resultsfile.close()

	iteraton = -1
	for par in q:
		iteraton += 1
		# Plot points
		i = par; # plot variable indices
		bounds_p = np.zeros(bounds.shape)
		bounds_p_n = np.zeros(bounds_n.shape)
		nn_vec = nn*np.ones(len(LHS_f[0,:]),dtype=int)
		fixed_value_lc = np.zeros((d,))
		for n in range(len(bounds)):
		    if n not in i:
		        lm = nominal[n]
		        fixed_value = scaling(lm,lob[n],upb[n],2) # Assign lambdas
		        
		        bounds_p[n,0] = fixed_value-0.0000001 # Set bounds equal to each other
		        bounds_p[n,1] = fixed_value+0.0000001 # Set bounds equal to each other
		        bounds_p_n[n,0] = lm # Nomalized bounds
		        bounds_p_n[n,1] = lm+0.01 # Nomalized bounds
		        nn_vec[n] = 1
		        
		        fixed_value_lc[n] = scaling(lm,lob[n],upb[n],2); # Assign lambdas
		    else:
		        bounds_p[n,0] = bounds[n,0]
		        bounds_p[n,1] = bounds[n,1]
		        bounds_p_n[n,0] = bounds_n[n,0]
		        bounds_p_n[n,1] = bounds_n[n,1]
		
		X = gridsamp(bounds_p_n.T, nn_vec)
		# Prediction
		# YX = sm.predict_values(X)
		[YX, std, ei, cdf] = server.sgtelib_server_predict(X)
		
	#========================= DATA VISUALIZATION =============================#
	# %% Sensitivity plots
		YX_obj = YX[:,0]
		X = X[:,i]
		X1_norm = np.reshape(X[:,0],(nn,nn)); X2_norm = np.reshape(X[:,1],(nn,nn))
		X1 = scaling(X1_norm, lob[i[0]], upb[i[0]], 2) # Scale up plot variable
		X2 = scaling(X2_norm, lob[i[1]], upb[i[1]], 2) # Scale up plot variable
		YX_obj = np.reshape(YX_obj, np.shape(X1))
		
		cmax = 6; cmin = 1 # set colorbar limits
		# cmax = 3.6; cmin = 2.2 # set colorbar limits (for plotting lambda = 109)
		# cmax = 100000; cmin = 0 # set colorbar limits
		# cmax = 4.4; cmin = 1.2; # set colorbar limits
		
		ax = fig.add_subplot(gs[iteraton]) # subplot
		cf = ax.contourf( X1, X2, YX_obj, cmap=plt.cm.jet) # plot contour
		# cf = ax.contourf( X1, X2, YX_obj, vmin = cmin, vmax = cmax, cmap=plt.cm.jet); # plot contour
		ax.contour(cf, colors='k', zorder=1)
		
		cbar = plt.cm.ScalarMappable(cmap=plt.cm.jet)
		cbar.set_array(YX_obj)

		boundaries = np.linspace(cmin, cmax, 51)
		# boundaries = np.linspace(cmin, cmax, int(np.round(((cmax-cmin)/0.2)*6))-1) # (for plotting lambda = 109)
		# cbar_h = fig.colorbar(cbar)
		cbar_h = fig.colorbar(cbar, boundaries=boundaries)
		cbar_h.set_label('$g_{f1}(\mathbf{p})$', rotation=90, labelpad=3)
		# cbar_h.set_label('$n_{safety}(\mathbf{T})$', rotation=90, labelpad=3)
		
		artists, labels = cf.legend_elements()
		af = artists[0]
		
	#======================== NONLINEAR CONSTRAINTS ============================#	
	# %% Nonlinear constraints
		YX_cstr = YX - threshold
		YX_cstr = np.reshape(YX_cstr, np.shape(X1))
		if plot_index >= 1:
			c1 = ax.contourf( X1, X2, YX_cstr, alpha=0.0, levels=[-20, 0, 20], colors=['#FF0000','#FF0000'], 
							hatches=['//', None])
			ax.contour(c1, colors='#FF0000', linewidths = 2.0, zorder=2)
			a1 = patches.Rectangle((20,20), 20, 20, linewidth=2, edgecolor='#FF0000', facecolor='none', fill='None', hatch='///')
		else:
			a1 = patches.Rectangle((20,20), 20, 20, linewidth=1.5, edgecolor='#FFFFFF', facecolor='#FFFFFF', fill='#FFFFFF', hatch=None)
	#====================== REQUIREMENTS CONSTRAINTS ==========================#	
	# %% Requirements bounds
		
		if req_type == 'uniform':	

			YX_req = np.zeros(X1.shape)
			
			cond1 = (X1 >= bounds_req[i[0],0]) & (X1 <= bounds_req[i[0],1]) # x-axis
			cond2 = (X2 >= bounds_req[i[1],0]) & (X2 <= bounds_req[i[1],1]) # y-axis
			cond = (cond1) & (cond2)
			
			YX_req[cond] = 1
			if plot_index >= 2:
				c2 = ax.contourf( X1, X2, YX_req, alpha=0.1, levels=[-10, 0, 10], colors=['#1EAA37','#1EAA37'], 
									hatches=[None, None])
				ax.contour(c2, colors='#1EAA37', linewidths = 3.0, zorder=3)
				a2 = patches.Rectangle((20,20), 20, 20, linewidth=2, edgecolor='#1EAA37', facecolor='none', fill='None', hatch=None)
			else:
				a2 = patches.Rectangle((20,20), 20, 20, linewidth=1.5, edgecolor='#FFFFFF', facecolor='#FFFFFF', fill='#FFFFFF', hatch=None)
	
		elif req_type == 'guassian':
			
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
			L = []
			for n in range(3):
				# Pack X and Y into a single 3-dimensional array
				pos = np.empty((1,1) + (2,))
				x_l = [ mu_sp[0] + ((n+1) * np.sqrt(Sigma_sp[0,0])), # evaluate at Sigma not Sigma^2
						mu_sp[1]                                   ]
				
				level_index = 0
				for value in x_l:
					pos[:, :, level_index] = value
					level_index += 1
					
				LN = multivariate_gaussian(pos, mu_sp, Sigma_sp)
				L += [LN]

			if plot_index >= 2:
				# c2 = ax.contourf( X1, X2, Z, colors=['#1EAA37'], alpha=0.0);
				c2 = ax.contourf( X1, X2, Z, alpha=0.25, cmap=plt.cm.Blues)
				
				ax.contour(c2, colors='#1EAA37', levels=L, linewidths = 4.5)
				a2 = patches.Rectangle((20,20), 20, 20, linewidth=2, edgecolor='#1EAA37', facecolor='none', fill='None', hatch=None)
			else:
				a2 = patches.Rectangle((20,20), 20, 20, linewidth=1.5, edgecolor='#FFFFFF', facecolor='#FFFFFF', fill='#FFFFFF', hatch=None)
	
	#========================= MONTE CARLO POINTS =============================#

		import matplotlib.lines as mlines

		ax.axis([lob[i[0]],upb[i[0]],lob[i[1]],upb[i[1]]]) # fix the axis limits
		
		if plot_index >= 3:
			ax.plot(dFF[:50,i[0]],dFF[:50,i[1]],'.k', markersize = 3) # plot DOE points for surrogate (first 50 only)
			a_MCI = mlines.Line2D([], [], color='black', marker='.', markersize=5, linestyle='')
			# ax.plot(S[:,i[0]],S[:,i[1]],'.k') # plot DOE points for surrogate
		else:
			a_MCI = patches.Rectangle((20,20), 20, 20, linewidth=1.5, edgecolor='#FFFFFF', facecolor='#FFFFFF', fill='#FFFFFF', hatch=None)
	
	#============================ AXIS LABELS =================================#	

		ax.set_xlabel(variable_lbls[i[0]], labelpad=-1)
		ax.set_ylabel(variable_lbls[i[1]], labelpad=-16)
	
	#============================= SET LABELS =================================#
		# Set labels (OPTIONAL)
		if iteraton == 0:

			ax.annotate("$\mathrm{Buffer}~B$", fontsize=12, xy =(25, 0), xytext =(50, 75), 
				arrowprops=dict(arrowstyle="simple,tail_width=0.2,head_width=0.5,head_length=0.7",
                                facecolor='#d9d9d9',
								edgecolor='#d9d9d9',
								color='#d9d9d9',
								shrinkA=5,
                                shrinkB=5,
                                fc="k", ec="k",
                                connectionstyle="arc3,rad=-0.05",
                                ),
				bbox=dict(edgecolor='white', facecolor='white', alpha=1.0),)

			ax.annotate("$\mathrm{Danger~zone}~D$", fontsize=12, xy =(-75, 0), xytext =(-70, 75), 
				arrowprops=dict(arrowstyle="simple,tail_width=0.2,head_width=0.5,head_length=0.7",
                                facecolor='#d9d9d9',
								edgecolor='#d9d9d9',
								color='#d9d9d9',
								shrinkA=5,
                                shrinkB=5,
                                fc="k", ec="k",
                                connectionstyle="arc3,rad=-0.05",
                                ),
				bbox=dict(edgecolor='white', facecolor='white', alpha=1.0),)

			ax.text(25, -75, "$\mathrm{Excess}~E$", fontsize=12, 
				bbox=dict(edgecolor='white', facecolor='white', alpha=1.0), zorder = 21) # Danger zone

	# handles, labels = [[a1], ["C':~$\hat{g}_{f1}(\mathbf{p}) - t_1 > 0$", ]]
	# fig.legend(handles, labels, loc='upper center', ncol=1, fontsize = 14)

	# handles, labels = [[a1,a2], ["C':~$\hat{g}_{f1}(\mathbf{p}) - t_1 > 0$", 
	# 							 "Joint PDF:~$F_{\mathbf{X}}(\mathbf{p})$"]]
	# fig.legend(handles, labels, loc='upper center', ncol=2, fontsize = 14)

	# handles, labels = [[a1,a2,a3], ["C':~$\hat{g}_{f1}(\mathbf{p}) - t_1 > 0$", 
	# 								"Joint PDF:~$F_{\mathbf{X}}(\mathbf{p})$",
	# 								"$F_{\mathbf{X}}(\mathbf{p})$ | \mathbf{p} \in C$"]]
	# fig.legend(handles, labels, loc='upper center', ncol=3, fontsize = 14)

	# handles, labels = [[a1,a2,a3,a_MCI], ["$C'$: $t_1 - \hat{g}_{f1}(\mathbf{p}) > 0$", 
	# 									  "Joint PDF: $F_{\mathbf{X}}(\mathbf{p})$",
	# 									  "$F_{\mathbf{X}}(\mathbf{p}) | \mathbf{p} \in C$",
	# 									  "Monte Carlo samples" ]] # for mathematical notation

	handles, labels = [[a1,a2,a_MCI], ["$C'$: Compliment of capability", 
										"R: Requirement set",
										"Monte Carlo samples" ]] # for presentation
	lx = fig.legend(handles, labels, loc='upper center', ncol=3, fontsize = 14)

	l_index = 0
	for item in lx.get_texts():
		if (l_index + 1) > plot_index:
			item.set_color("white")
		l_index += 1

def plot_countour_code_2D(bounds,bounds_n,bounds_req,LHS_MCI_file,mu,Sigma,req_type,lob,upb,LHS_f,d,
					   	  nominal,threshold,nn,S,server,variable_lbls,plt,fig,plot_index=4,reliability=None):
	
	from design_margins import scaling
	from design_margins import multivariate_gaussian
	import matplotlib.patches as patches
	import numpy as np
	import os, pickle
	
	# get LHS_MCI points distribution
	DOE_full_name = LHS_MCI_file + '.pkl'
	DOE_filepath = os.path.join(os.getcwd(),'Optimization_studies',DOE_full_name)
	resultsfile=open(DOE_filepath,'rb')
	dFF = pickle.load(resultsfile)
	dFF_n = pickle.load(resultsfile)
	resultsfile.close()
	
	par = (0,3) # Choose T1 vs T4 2D projection
	# Plot points
	i = par; # plot variable indices
	bounds_p = np.zeros(bounds.shape)
	bounds_p_n = np.zeros(bounds_n.shape)
	nn_vec = nn*np.ones(len(LHS_f[0,:]),dtype=int)
	fixed_value_lc = np.zeros((d,))
	for n in range(len(bounds)):
		if n not in i:
			lm = nominal[n]
			fixed_value = scaling(lm,lob[n],upb[n],2) # Assign lambdas
			
			bounds_p[n,0] = fixed_value-0.0000001 # Set bounds equal to each other
			bounds_p[n,1] = fixed_value+0.0000001 # Set bounds equal to each other
			bounds_p_n[n,0] = lm # Nomalized bounds
			bounds_p_n[n,1] = lm+0.01 # Nomalized bounds
			nn_vec[n] = 1
			
			fixed_value_lc[n] = scaling(lm,lob[n],upb[n],2); # Assign lambdas
		else:
			bounds_p[n,0] = bounds[n,0]
			bounds_p[n,1] = bounds[n,1]
			bounds_p_n[n,0] = bounds_n[n,0]
			bounds_p_n[n,1] = bounds_n[n,1]
	
	X = gridsamp(bounds_p_n.T, nn_vec)
	# Prediction
	# YX = sm.predict_values(X)
	[YX, std, ei, cdf] = server.sgtelib_server_predict(X)
		
	#========================= DATA VISUALIZATION =============================#
	# %% Sensitivity plots
	YX_obj = YX[:,0]
	X = X[:,i]
	X1_norm = np.reshape(X[:,0],(nn,nn)); X2_norm = np.reshape(X[:,1],(nn,nn))
	X1 = scaling(X1_norm, lob[i[0]], upb[i[0]], 2) # Scale up plot variable
	X2 = scaling(X2_norm, lob[i[1]], upb[i[1]], 2) # Scale up plot variable
	YX_obj = np.reshape(YX_obj, np.shape(X1))
	
	cmax = 6; cmin = 1 # set colorbar limits
	# cmax = 3.6; cmin = 2.2 # set colorbar limits (for plotting lambda = 109)
	# cmax = 100000; cmin = 0 # set colorbar limits
	# cmax = 4.4; cmin = 1.2; # set colorbar limits
	
	ax = fig.gca() # plot axis
	cf = ax.contourf( X1, X2, YX_obj, cmap=plt.cm.jet) # plot contour
	# cf = ax.contourf( X1, X2, YX_obj, vmin = cmin, vmax = cmax, cmap=plt.cm.jet); # plot contour
	ax.contour(cf, colors='k', zorder=1)
	
	cbar = plt.cm.ScalarMappable(cmap=plt.cm.jet)
	cbar.set_array(YX_obj)

	boundaries = np.linspace(cmin, cmax, 51)
	# boundaries = np.linspace(cmin, cmax, int(np.round(((cmax-cmin)/0.2)*6))-1) # (for plotting lambda = 109)
	# cbar_h = fig.colorbar(cbar)
	cbar_h = fig.colorbar(cbar, boundaries=boundaries)
	cbar_h.set_label('$g_{f1}(\mathbf{p})$', rotation=90, labelpad=3, fontsize=18)
	cbar_h.ax.tick_params(labelsize=16) 
	# cbar_h.set_label('$n_{safety}(\mathbf{T})$', rotation=90, labelpad=3)
	
	artists, labels = cf.legend_elements()
	af = artists[0]
		
	#======================== NONLINEAR CONSTRAINTS ============================#	
	# %% Nonlinear constraints
	YX_cstr = YX - threshold
	YX_cstr = np.reshape(YX_cstr, np.shape(X1))
	if plot_index >= 1:
		c1 = ax.contourf( X1, X2, YX_cstr, alpha=0.0, levels=[-20, 0, 20], colors=['#FF0000','#FF0000'], 
						hatches=['//', None])
		ax.contour(c1, colors='#FF0000', linewidths = 2.0, zorder=2)
		a1 = patches.Rectangle((20,20), 20, 20, linewidth=2, edgecolor='#FF0000', facecolor='none', fill='None', hatch='///')
	else:
		a1 = patches.Rectangle((20,20), 20, 20, linewidth=1.5, edgecolor='#FFFFFF', facecolor='#FFFFFF', fill='#FFFFFF', hatch=None)
	#====================== REQUIREMENTS CONSTRAINTS ==========================#	
	# %% Requirements bounds
	
	if req_type == 'uniform':	

		YX_req = np.zeros(X1.shape)
		
		cond1 = (X1 >= bounds_req[i[0],0]) & (X1 <= bounds_req[i[0],1]) # x-axis
		cond2 = (X2 >= bounds_req[i[1],0]) & (X2 <= bounds_req[i[1],1]) # y-axis
		cond = (cond1) & (cond2)
		
		YX_req[cond] = 1
		if plot_index >= 2:
			c2 = ax.contourf( X1, X2, YX_req, alpha=0.1, levels=[-10, 0, 10], colors=['#1EAA37','#1EAA37'], 
								hatches=[None, None])
			ax.contour(c2, colors='#1EAA37', linewidths = 3.0, zorder=3)
			a2 = patches.Rectangle((20,20), 20, 20, linewidth=2, edgecolor='#1EAA37', facecolor='none', fill='None', hatch=None)
		else:
			a2 = patches.Rectangle((20,20), 20, 20, linewidth=1.5, edgecolor='#FFFFFF', facecolor='#FFFFFF', fill='#FFFFFF', hatch=None)

	elif req_type == 'guassian':
		
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
		L = []
		for n in range(3):
			# Pack X and Y into a single 3-dimensional array
			pos = np.empty((1,1) + (2,))
			x_l = [ mu_sp[0] + ((n+1) * np.sqrt(Sigma_sp[0,0])), # evaluate at Sigma not Sigma^2
					mu_sp[1]                                   ]
			
			level_index = 0
			for value in x_l:
				pos[:, :, level_index] = value
				level_index += 1
				
			LN = multivariate_gaussian(pos, mu_sp, Sigma_sp)
			L += [LN]

		if plot_index >= 2:
			# c2 = ax.contourf( X1, X2, Z, colors=['#1EAA37'], alpha=0.0);
			c2 = ax.contourf( X1, X2, Z, alpha=0.25, cmap=plt.cm.Blues)
			
			ax.contour(c2, colors='#1EAA37', levels=L, linewidths = 4.5)
			a2 = patches.Rectangle((20,20), 20, 20, linewidth=2, edgecolor='#1EAA37', facecolor='none', fill='None', hatch=None)
		else:
			a2 = patches.Rectangle((20,20), 20, 20, linewidth=1.5, edgecolor='#FFFFFF', facecolor='#FFFFFF', fill='#FFFFFF', hatch=None)
	
	#========================= MONTE CARLO POINTS =============================#

	import matplotlib.lines as mlines

	ax.axis([lob[i[0]],upb[i[0]],lob[i[1]],upb[i[1]]]) # fix the axis limits
	
	if plot_index >= 3:
		ax.plot(dFF[:50,i[0]],dFF[:50,i[1]],'.k', markersize = 10) # plot DOE points for surrogate (first 50 only)
		a_MCI = mlines.Line2D([], [], color='black', marker='.', markersize=5, linestyle='')
		# ax.plot(S[:,i[0]],S[:,i[1]],'.k') # plot DOE points for surrogate
	else:
		a_MCI = patches.Rectangle((20,20), 20, 20, linewidth=1.5, edgecolor='#FFFFFF', facecolor='#FFFFFF', fill='#FFFFFF', hatch=None)
	
	#============================ AXIS LABELS =================================#	
	ax.set_xlabel(variable_lbls[i[0]], labelpad=-1, fontsize=18)
	ax.set_ylabel(variable_lbls[i[1]], labelpad=-16, fontsize=18)

	ax.tick_params(axis='x', labelsize=16)
	ax.tick_params(axis='y', labelsize=16)

	#============================= SET LABELS =================================#
	# Set labels (OPTIONAL)

	if reliability is not None:

		if reliability >= 0.001:

			buffer_coordinates = (bounds_req[i[0],1]-10,bounds_req[i[1],0]+10)

			ax.annotate("$\mathrm{Buffer}~B$", fontsize=18, xy = buffer_coordinates, xytext =(12.5, 85), 
				arrowprops=dict(arrowstyle="simple,tail_width=0.2,head_width=0.5,head_length=0.7",
								facecolor='#d9d9d9',
								edgecolor='#d9d9d9',
								color='#d9d9d9',
								shrinkA=5,
								shrinkB=5,
								fc="k", ec="k",
								connectionstyle="arc3,rad=-0.05",
								),
				bbox=dict(edgecolor='white', facecolor='white', alpha=1.0),)
			
		else:
			ax.text(12.5, 85, "$\mathrm{No~buffer~left}$", fontsize=18, 
				bbox=dict(edgecolor='white', facecolor='white', alpha=1.0), zorder = 21) # Danger zone

		if reliability <= 0.999:

			Danger_coordinates = (bounds_req[i[0],0]+10,bounds_req[i[1],1]-10)

			ax.annotate("$\mathrm{Danger~zone}~D$", fontsize=18, xy = Danger_coordinates, xytext =(-90, -90), 
				arrowprops=dict(arrowstyle="simple,tail_width=0.2,head_width=0.5,head_length=0.7",
								facecolor='#d9d9d9',
								edgecolor='#d9d9d9',
								color='#d9d9d9',
								shrinkA=5,
								shrinkB=5,
								fc="k", ec="k",
								connectionstyle="arc3,rad=-0.05",
								),
				bbox=dict(edgecolor='white', facecolor='white', alpha=1.0),)
		
		else:
			ax.text(-90, -90, "$\mathrm{No~danger}$", fontsize=18, 
				bbox=dict(edgecolor='white', facecolor='white', alpha=1.0), zorder = 21) # Danger zone


		ax.text(62.5, 85, "$\mathrm{Excess}~E$", fontsize=18, 
			bbox=dict(edgecolor='white', facecolor='white', alpha=1.0), zorder = 21) # Danger zone

	# handles, labels = [[a1], ["C':~$\hat{g}_{f1}(\mathbf{p}) - t_1 > 0$", ]]
	# fig.legend(handles, labels, loc='upper center', ncol=1, fontsize = 14)

	# handles, labels = [[a1,a2], ["C':~$\hat{g}_{f1}(\mathbf{p}) - t_1 > 0$", 
	# 							 "Joint PDF:~$F_{\mathbf{X}}(\mathbf{p})$"]]
	# fig.legend(handles, labels, loc='upper center', ncol=2, fontsize = 14)

	# handles, labels = [[a1,a2,a3], ["C':~$\hat{g}_{f1}(\mathbf{p}) - t_1 > 0$", 
	# 								"Joint PDF:~$F_{\mathbf{X}}(\mathbf{p})$",
	# 								"$F_{\mathbf{X}}(\mathbf{p})$ | \mathbf{p} \in C$"]]
	# fig.legend(handles, labels, loc='upper center', ncol=3, fontsize = 14)

	# handles, labels = [[a1,a2,a3,a_MCI], ["$C'$: $t_1 - \hat{g}_{f1}(\mathbf{p}) > 0$", 
	# 									  "Joint PDF: $F_{\mathbf{X}}(\mathbf{p})$",
	# 									  "$F_{\mathbf{X}}(\mathbf{p}) | \mathbf{p} \in C$",
	# 									  "Monte Carlo samples" ]] # for mathematical notation

	handles, labels = [[a1,a2,a_MCI], ["$C'$: Compliment of capability", 
										"R: Requirement set",
										"Monte Carlo samples" ]] # for presentation
	lx = fig.legend(handles, labels, loc='upper center', ncol=3, fontsize = 16)

	l_index = 0
	for item in lx.get_texts():
		if (l_index + 1) > plot_index:
			item.set_color("white")
		l_index += 1

def hyperplane_SGTE_vis_norm(server,LHS_f,bounds,bounds_req,LHS_MCI_file,mu,Sigma,req_type,variable_lbls,
							 nominal,threshold,objs,nn,fig,plt,plot_index=4,plot_2D=False,fig_2D=None,reliability=None):
	import numpy as np
	from scipy.special import comb
	from design_margins import scaling
	from itertools import combinations
	import matplotlib.gridspec as gridspec
	from matplotlib import cm
	from shutil import copyfile
	import os
	
	lob = bounds[:,0]
	upb = bounds[:,1]
	
	Y = objs; S = LHS_f
	
	lob_n = np.zeros(np.size(lob)); upb_n = np.ones(np.size(upb)); 
	bounds_n = np.zeros(np.shape(bounds))
	bounds_n[:,0] = lob_n; bounds_n[:,1] = upb_n; 
	
	#======================== 2D GRID CONSTRUCTION ============================#
	# %% Activate 2 out 4 variables up to 4 variables
	d = len(LHS_f[0,:]); #<-------- Number of variables
	if d > 4: # maximum of four variables allowed
	    d = 4
	    
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
						height_ratios = np.ones(sp_shape[0],dtype=int),
						left=0.15, right=0.85, wspace=0.2)
	
	
	q = combinations(range(d),2); # choose 2 out d variables
	ss = comb(d,2,exact=True)
	
	if d != 1:
		if plot_2D:
			fig_2D.set_figheight(3*2); fig_2D.set_figwidth(4.67*2)
			plot_countour_code_2D(bounds,bounds_n,bounds_req,LHS_MCI_file,mu,Sigma,req_type,lob,upb,LHS_f,d,
							   	  nominal,threshold,nn,S,server,variable_lbls,plt,fig_2D,plot_index=plot_index,reliability=reliability)
									 
		plot_countour_code(q,bounds,bounds_n,bounds_req,LHS_MCI_file,mu,Sigma,req_type,lob,upb,LHS_f,d,
						   nominal,threshold,nn,S,server,variable_lbls,gs,plt,fig,plot_index=plot_index)
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