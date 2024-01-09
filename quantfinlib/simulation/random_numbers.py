# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 14:52:14 2024

@author: Alexander Swade
"""

import numpy as np

def generate_standardnormal_numbers(n, ap=False, mm=False):
    '''
    Function to generate n standard normal distributed random numbers.

    Parameters
    ----------
    n : int
        Number of rand.-numb. to create.
    ap : boolean, optional
        Use antithetic variates method. The default is False.
    mm : boolean, optional
        Use moment matching methods, i.e., normalize data. The default is False.

    Returns
    -------
    ran : TYPE
        DESCRIPTION.

    '''
    if ap:
            ran = np.random.standard_normal(n / 2)
            ran = np.concatenate((ran, -ran))
    else:
            ran = np.random.standard_normal(n)
    if mm:
        ran = ran - np.mean(ran)
        ran = ran / np.std(ran)
    
    return ran