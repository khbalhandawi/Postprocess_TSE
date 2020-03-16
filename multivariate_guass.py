# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 16:04:09 2020

@author: Khalil
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy import integrate

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

    return np.exp(-fac / 2) / N

def integrand_multivariate_gaussian(*arg):
    """Return the multivariate Gaussian distribution on array pos.
    """
    # Mean vector and covariance matrix
#    mu = np.array([0., 1.])
#    Sigma = np.array([[ 1. , -0.5], [-0.5,  1.5]])
    
    x = arg[0:-2] 
    mu = arg[-2] 
    Sigma = arg[-1]
    
    # Pack X and Y into a single 3-dimensional array
    pos = np.empty((1,1) + (2,))
    
    print(x)
    i = 0
    for value in x:
        pos[:, :, i] = x[i]
        i += 1
    
    Z = multivariate_gaussian(pos, mu, Sigma)

    return Z

def TwoD_example():

    # Our 2-dimensional distribution will be over variables X and Y
    N = 60
    X = np.linspace(-3, 3, N)
    Y = np.linspace(-3, 4, N)
    X, Y = np.meshgrid(X, Y)
    
    # Mean vector and covariance matrix
    mu = np.array([0., 1.])
    Sigma = np.array([[ 1. , -0.0], [-0.,  1.5]])
    
    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    
    # The distribution on the variables X, Y packed into pos.
    Z = multivariate_gaussian(pos, mu, Sigma)
    
    Z1 = multivariate_gaussian(Sigma[0,0], np.array([mu[0]]), np.array([[Sigma[0,0]]]) )
    print(Z1)
    
    #Z_int = np.reshape(Z,3600)
    #integ = np.sum(Z_int)
    limits = [ [-3*Sigma[0,0],3 *Sigma[0,0]], [-3,4] ]
    
    integ = integrate.nquad(integrand_multivariate_gaussian, limits, args=(mu,Sigma))
    print(integ)
    
    # Create a surface plot and projected filled contour plot under it.
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=1, antialiased=True,
                    cmap=cm.viridis)
    
    cset = ax.contourf(X, Y, Z, zdir='z', offset=-0.15, cmap=cm.viridis)
    
    # Adjust the limits, ticks and view angle
    ax.set_zlim(-0.15,0.2)
    ax.set_zticks(np.linspace(0,0.2,5))
    ax.view_init(27, -21)
    
    plt.show()
    
def OneD_example():

    # Our 2-dimensional distribution will be over variables X and Y
    N = 60
    X = np.linspace(-3, 3, N)
    Y = np.linspace(-3, 4, N)
    X, Y = np.meshgrid(X, Y)
    
    # Mean vector and covariance matrix
    mu = np.array([0., 1.])
    Sigma = np.array([[ 1. , -0.0], [-0.,  1.5]])
    
    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    
    # The distribution on the variables X, Y packed into pos.
    Z = multivariate_gaussian(pos, mu, Sigma)
    
    Z1 = multivariate_gaussian(Sigma[0,0], np.array([mu[0]]), np.array([[Sigma[0,0]]]) )
    print(Z1)
    
    #Z_int = np.reshape(Z,3600)
    #integ = np.sum(Z_int)
    limits = [ [-3*Sigma[0,0],3 *Sigma[0,0]], [-3,4] ]
    
    integ = integrate.nquad(integrand_multivariate_gaussian, limits, args=(mu,Sigma))
    print(integ)
    
    # Create a surface plot and projected filled contour plot under it.
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=1, antialiased=True,
                    cmap=cm.viridis)
    
    cset = ax.contourf(X, Y, Z, zdir='z', offset=-0.15, cmap=cm.viridis)
    
    # Adjust the limits, ticks and view angle
    ax.set_zlim(-0.15,0.2)
    ax.set_zticks(np.linspace(0,0.2,5))
    ax.view_init(27, -21)
    
    plt.show()
  