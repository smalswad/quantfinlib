# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 13:00:56 2023

@author: smalswad
"""
import numpy as np

def k_max(x, k, axis=None):
    '''
    Get k-th largest value of an array.

    Parameters
    ----------
    x : np.ndarray, shape(l,m)
        Data input.
    k : int
        Indicator for k-th position (where k=1 is the largeste value)

    Returns
    -------
    result : element
        k-th largest value of input array x (along specified axis).
    '''
    l = x.shape[0]
    idx = k-1 # Account for start of index at 0
    if k>l:
        raise ValueError("'k' is larger than the length of 'x'!")
    # 1-dim array input
    if x.ndim ==1:        
        if idx==0: 
            result = max(x)
        else:
            # Sort in descending order
            x_sort = -np.sort(-x)
            result = x_sort[idx]
    
    # n-dim array input
    else:
        x_sort = np.sort(-x, axis=axis)
        if axis == 1:
            result = -x_sort[:,idx]
        else:
            result = -x_sort[idx]        
  
    return result