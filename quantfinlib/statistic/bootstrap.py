# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 16:41:40 2023

@author: ROB6738
"""
from arch.bootstrap import MovingBlockBootstrap

def gen_bbs_samples(data, block_length=12, k=1000):
    '''
    Function to generate block bootstrap samples of the given time-series data.

    Parameters
    ----------
    data : ArrayLike, shape(T,n)
        Data to be bootstrapped.
    block_length : int, optional
        Length of the block bootstrap. The default is 12.
    k : int, optional
        Number of total bootstrap samples to be estimated. The default is 1000.

    Returns
    -------
    samples : list
        List with k bootstrapped samples.

    '''
    samples = list()
    bs = MovingBlockBootstrap(block_length, x=data, seed=12345)    
    for s in bs.bootstrap(k):
        samples.append(bs.x.reset_index(drop=True))
    
    return samples