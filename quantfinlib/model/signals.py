# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 09:21:04 2023

@author: ROB6738
"""
import numpy as np
import pandas as pd

def constr_timeseries_momentum(ret, k=1, j=12):
    '''
    Calculate timeseries factor momentum as in Gupta and Kelly (2019).

    Parameters
    ----------
    ret : pd.DataFrame, shape(T, n)
        Returns of n factor portfolios.
    k : int, optional
        Periods to skip for look back period. The default is 1.
    j : int, optional
        Periods to look back. The default is 12.

    Returns
    -------
    position : pd.DataFrame
        Boolean df indicating position (long/short) of each factor in time.
    tsfm : pd.Series
        Time series factor momentum strategy return.

    '''
    # Calculate volatility
    p = 36 if j<12 else 120      
    vola = ret.rolling(p).std() * np.sqrt(12)
    
    # Calculate scaling factor (z-score, capped at +-2)
    s = np.minimum(np.maximum(
        ret.shift(k).rolling(j-k).sum().divide(vola), -2), 2)
    
    # Calculate time series factor momentum returns
    f_tsm = ret * s.shift()
    
    # Identify long and short positions
    positions = pd.DataFrame(np.where(s.dropna(how='all') > 0, 1, -1),
                            index = s.dropna(how='all').index, 
                            columns = s.columns)
    tsfm_long = f_tsm.where(s > 0).sum(axis=1) / s.where(s > 0).sum(axis=1)
    tsfm_short = f_tsm.where(s <= 0).sum(axis=1) / s.where(s <= 0).sum(axis=1)
    tsfm = tsfm_long - tsfm_short
    
    # Calculate standardized weights
    wl = s.where(s>0).divide(s.where(s>0).sum(axis=1), axis=0)
    ws = -s.where(s<0).divide(s.where(s<0).sum(axis=1), axis=0)
    weights = wl.copy()
    weights[weights.isnull()] = ws
    
    return (positions,
            tsfm.dropna(),
            weights.shift().dropna(how='all', axis=0))