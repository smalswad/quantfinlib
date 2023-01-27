# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 11:16:52 2023

@author: smalswad
"""

import pandas as pd


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