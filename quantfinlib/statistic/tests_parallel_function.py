# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 10:40:26 2023

@author: ROB6738
"""

def _run_boot(bs_idx, ret, pre, cand, min_obs):
    '''
    Helper function for factor_cand_test(). Get bootstrap sampled data and run
    _avg_statistics function on selected sample.

    Parameters
    ----------
    bs_idx : list
        Row indices to resample given data.
    ret : pd.DataFrame, shape(T,N)
        Excess returns of N stocks/factors. Missing data are given as nan.
    pre : pd.DataFrame, shape(T, kp)
        Return data of pre-selected factors, no missing data.
    cand : pd.DataFrame, shape(T,kc)
        Return data of candidate factors, no missing data.
    min_obs : int, optional
        Minimum number of observations per time-series to be included. The 
        default is 36.

    Returns
    -------
    sum_mm : np.array, shape(kc,)
        Mean-based statistics for candidate factors.
    sum_dd : np.array, shape(kc,)
        Median-based statistics for candidate factors.

    '''
    from quantfinlib.statistic.tests import _avg_statistics
    
    ret_bb = ret.iloc[bs_idx, :].reset_index(drop=True)
    pre_bb = pre.iloc[bs_idx, :].reset_index(drop=True)
    cand_bb = cand.iloc[bs_idx, :].reset_index(drop=True)
        
    # Calculate  statistics
    sum_mm, sum_dd = _avg_statistics(ret_bb, pre_bb, cand_bb, min_obs=min_obs)
    
    return (sum_mm, sum_dd)