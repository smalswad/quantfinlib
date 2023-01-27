# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 11:16:52 2023

@author: smalswad
"""

import numpy as np
import pandas as pd


def calc_columwise_correlations(x,y):
    '''
    Calculate column-wise corrolations of two data frames

    Parameters
    ----------
    x : pd.DataFrame, shape(T,N)
    y : pd.DataFrame, shape(T,M)

    Returns
    -------
    pd.DataFrame, shape(N,M)

    '''
    x = (x - x.mean(axis=0)) / x.std(axis=0)
    y = (y - y.mean(axis=0)) / y.std(axis=0)
    pearson_r = np.dot(x.T, y) / x.shape[0]
    
    return pd.DataFrame(pearson_r, index=x.columns, columns=y.columns)


def idx_intersect(indices):
    '''
    Helper function to get intersection of multiple pd.DataFrame indices

    Parameters
    ----------
    indices : list
        list of pd.DataFrame.index to comapare.

    Returns
    -------
    intersect : pd.DataFrame.index
        Index of intersection.

    '''
    intersect = indices[0].intersection(indices[1])
    if len(indices) > 2:        
      return(idx_intersect(indices[2:]+[intersect]))
    else:
        return intersect