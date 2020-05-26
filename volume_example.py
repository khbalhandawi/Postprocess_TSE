# -*- coding: utf-8 -*-
"""
Created on Mon May 25 19:12:23 2020

@author: Khalil
"""

import numpy as np

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



bounds_th = np.array( [[-100, 100],
                       [-100, 100],
                       [-100, 100],
                       [-100, 100]] )

bounds_req = np.array( [[ -100  , 50 ],
                        [-25 , 25 ],
                        [-25 , 25 ],
                        [-50 , 100 ]] ) # unused

lob = bounds_th[:,0]
upb = bounds_th[:,1]

lob_req = bounds_req[:,0]
upb_req = bounds_req[:,1]


UB_n = scaling(upb_req,lob,upb,1)
LB_n = scaling(lob_req,lob,upb,1)

V_HPrect = np.prod(UB_n - LB_n)