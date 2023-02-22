# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 14:28:24 2023

@author: smalswad
"""

import numpy as np
import pandas as pd
from scipy import stats

from quantfinlib.statistic.regression import LinearRegression

def calc_raw_pval(t, tvals):
    '''
    Calculate raw pvalues based on vector of null test statistics.

    Parameters
    ----------
    t : float
        Scalar-valued test statistic (large values are indicative of H_1).
    tvals : np.array, shape(M,)
        Vector of null test statistics.

    Returns
    -------
    univariate p-value.

    '''
    return (len(tvals[tvals >= t]) + 1)/(len(tvals) + 1)

def grs_test(dep_var, indep_var):
    '''
    Calculate GRS (Gibbons, Ross, and Shanken) test statistic and associated
    values. See Fama, French (2018): "Choosing factors" JFE 
    
    Individual returns are represented as
    
    r_(i,t) = alpha_(i) + sum_j=1^k beta_(j,t) F_(j,t) + res_(i,t)
    

    Parameters
    ----------
    dep_var : pd.DataFrame or pd.Series, shape(tau,n)
        LHS/dependent variable(s).
    indep_var : pd.DataFrame or pd.Series, shape(tau,k)
        RHS/independant variable(s). Specifically, factors of given model. 

    Returns
    -------
    f_grs : float
        GRS test statisitc, f_grs ~ F(n, tau-n-k).
    p_grs : float
        GRS p value
    avg_abs_alpha : float
        Average absolute intercept/alpha
    sh2f : float
        Maximum squared Sharpe ratio for model's factors
    sh2f_adj : float
        Adjusted sh2f as in Barillas et al. (2019), i.e.
        sh2f_adj = sh2f * (tau-k-2)/tau - k/tau
    sr : float
        Sharpe ratio of EW RHS portfolio
    sh2a : float
        Maximum squared Sharpe ratio for the intercepts for a set of LHS portfolios.
        
    '''
    
    tau = len(dep_var) # Get number of observations (time)
    n = dep_var.shape[1] if dep_var.ndim != 1 else 1 # Get number of dependent variables
    k = indep_var.shape[1] if indep_var.ndim != 1 else 1 # Get number of independent variables
    
    # Run multiple OLS regressions
    model = LinearRegression(fit_intercept=True)
    res = model.fit(indep_var, dep_var)    
    
    # Collect variables
    alphas = res.intercept_ 
    residuals = pd.DataFrame(res.residuals) 
    sigma = residuals.T.dot(residuals).values/(tau-k-1) # Calculate sigma estimate
    mu_f = indep_var.mean().values # Mean factor returns
    omega = (indep_var - mu_f).T.dot(indep_var - mu_f).values/(tau-1) # Calculate omega estimate
   
    # Calculate sh2a (Use Moore-Penrose inverse in case of singularity)
    try: 
        sh2a = alphas.T.dot(np.linalg.inv(sigma)).dot(alphas)
    
    except np.linalg.LinAlgError as err:
        if 'Singular matrix' in str(err):
            sh2a = alphas.T.dot(np.linalg.pinv(sigma)).dot(alphas)
        else:
            raise 
    
    # Calculate sh2f
    sh2f = mu_f.T.dot(np.linalg.inv(omega)).dot(mu_f)
    
    # Calculate GRS statistic        
    f_grs = (tau/n) * (tau-n-k)/(tau-k-1) * sh2a / (1 + sh2f)
    
    # Calculate further statistics
    p_grs = 1 - stats.f.cdf(f_grs, dfn=n, dfd=tau-n-k) 
    avg_abs_alpha = np.mean(np.absolute(alphas))
    sr = mu_f.T.dot(np.linalg.inv(omega)).dot(np.ones_like(mu_f)/k)
    sh2f_adj = sh2f * (tau-k-2)/tau - k/tau
     
    return (f_grs, p_grs, avg_abs_alpha, sh2f, sh2f_adj, sr, sh2a)

def mht_step_m(tval, tval_boot, digits=3):
    '''
    Multiple hypothesis correction of p-values based on
    
    Romano, J. P., & Wolf, M. (2016). Efficient computation of adjusted 
    p-values for resampling-based stepdown multiple testing. 
    Statistics & Probability Letters, 113, 38-40.    

    Parameters
    ----------
    tval : np.array, shape(s,)
        Vector of test statistics.
    tval_boot : np.array, shape(m,s)
        Bootstrapped t-statistics.
    digits : int, optional
        Number of digits to be reported after coma. The default is 3.

    Returns
    -------
    results: np.array, shape(S,3)
        Test result array, containing the test statistics (c1), raw p-values (c2),
        as well as adjusted p-values (c3)
    '''
    # Get dimensions
    m,s = tval_boot.shape
    
    # Error handling
    if s != len(tval):
        raise ValueError("Length of tval and column number of tval_boot do"
                         " not match!")
   
    # Calculate raw p-values
    pval_raw = np.array([calc_raw_pval(tval[i], tval_boot[:,i]) \
                         for i in range(s)])
        
    # Calculate adjusted p-values
    pval_adj = np.zeros((s,))
    # Sort t-values in descending order
    r = np.argsort(-tval)    
    tval_sorted = tval[r]
    tval_boot_sorted = tval_boot[:,r]

    max_stat = tval_boot_sorted.max(axis=1)
    pval_adj[0] = calc_raw_pval(tval_sorted[0], max_stat)

    for i in range(1,s):
        max_stat = tval_boot_sorted[:,i:(s+1)].max(axis=1)
        temp = calc_raw_pval(tval_sorted[i], max_stat)
        pval_adj[i] = max(temp, pval_adj[i-1])
    
    # re-arrange adjusted p-values into original order of hypotheses
    inv_r = np.zeros((s,), dtype=int)
    inv_r[r] = np.arange(s, dtype=int)
    pval_adj = pval_adj[inv_r]

    # Combine tvalues with raw and adjusted p-values
    result = np.column_stack((tval, pval_raw, pval_adj))

    return result.round(digits)










