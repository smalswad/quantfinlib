# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 16:41:40 2023

@author: ROB6738
"""
from arch.bootstrap import MovingBlockBootstrap

def gen_bbs_samples(data, block_length=12, k=1000, seed=12345):
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
    seed : int, optional
        Use seed number for pseudo-random variables (relevant for replicability).
        The default is 12345.

    Returns
    -------
    samples : list
        List with k bootstrapped samples.

    '''
    samples = list()
    idx = list()
    bs = MovingBlockBootstrap(block_length, x=data, seed=seed)    
    for s in bs.bootstrap(k):
        samples.append(bs.x.reset_index(drop=True))
        idx.append(bs.x.index)
    
    # Get positions of bootstrapped indices
    for i, samp in enumerate(idx):
        idx[i] = [data.index.get_loc(x) for x in samp]
    
    return (samples, idx)