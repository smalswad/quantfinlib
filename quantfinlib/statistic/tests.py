# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 14:28:24 2023

@author: smalswad
"""
import itertools
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import binom

from quantfinlib.statistic.regression import LinearRegression
from quantfinlib.utils.arrays import k_max

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

def fdp_step_m(z, z_null, gamma, alpha=0.05, n_max=50, disp=False):
    '''
    Calculate the FDP procedure of Romano, Shaikh, and Wolf and control for the
    probability that FDP is below a certain threshold. The procedure keeps 
    increasing the k order of the k-StepM until it reaches the desired FDP 
    control.

    Parameters
    ----------
    np.array, shape(s,)
        Vector of test statistics.
    z_null : np.array, shape(m,s)
        Bootstrapped t-statistics.
    gamma: float
        Threshold of FDP.
    alpha : float, optional
        Nominal significance level. The default is 0.05.
    n_max : int, optional
        Value of Nmax for the Operative Method; see Remark 4.1. in the paper. 
        The default is 50.

    Returns
    -------
    fwe_res : dict
        Results of the k_fwe routine where k is chosen such that it reaches the 
        desired FDP control.
    k : int
        Final k chosen. 
    '''
    # Start with k=1 
    k = 1    
    while True:
        fwe_res = k_fwe(z, z_null, k, alpha=alpha, n_max=n_max)

        # Stop iterating 
        rej = len(fwe_res['hypo_rej'])
        if rej < (k/gamma-1):
            if disp:
                print(f'The final choices are k = {k} and rej = {rej}')
            break
        
        # Increase k to the lower bound of the constraint, instead of k+=1
        k = int(max(np.floor(len(fwe_res['hypo_rej'])*gamma), k+1))

    return (fwe_res, k)

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

def k_fwe(z, z_null, k, alpha=0.05, n_max=50, digits=5):
    '''
    Control the generalized familywise error rate, k-FWE, as described in 
    
    Romano, J. P., Shaikh, A. M., & Wolf, M. (2008). Formalized data snooping 
    based on generalized error rates. Econometric Theory, 24(2), 404-447.
    
    It is up to the user to supply:
    - basic or studentized statistics
    - statistics designed for the one-sided setup or for the two-sided setup
      (in the latter case, absolute values should be used everywhere)

    Parameters
    ----------
    z : np.array, shape(s,)
        Vector of test statistics.
    z_null : np.array, shape(m,s)
        Bootstrapped t-statistics.
    k : int
        Value of k for control of the k-FWE (where k=1 is the first FWE).
    alpha : float, optional
        Nominal significance level. The default is 0.05.
    n_max : int, optional
        Value of Nmax for the Operative Method; see Remark 4.1. in the paper. 
        The default is 50.
    digits : int, optional
        Number of digits to be reported after coma. The default is 5.

    Returns
    -------
    hypo_rej = indices of the hypotheses that were rejected
    step_rej = says in which step each element of hypo.rej was rejected
    crit_val = the critical values used in each step (j =1, 2, ...)
    '''
    # Get dimensions
    m,s = z_null.shape
    
    # Error handling
    if s != len(z):
        raise ValueError("Length of z and column number of z_null do"
                         " not match!")
    
    # Sort t-values in descending order
    r = np.argsort(-z)    
    z_ord = z[r]
    z_null_ord = z_null[:,r]
    
    # Construct output arrays
    hypo_rej = np.repeat(np.nan, s)
    step_rej = np.repeat(np.nan, s)
    crit_val = []
    
    # Looping indices
    j = 1
    rej = 0
    
    # Run main routine of mStep approach
    while rej < s:
        if j==1:
            max_stat = k_max(z_null_ord, k, axis=1)     
            d = np.quantile(max_stat, q=1-alpha)
          
        elif k < 3:
            max_stat = k_max(z_null_ord[:,(rej-k+1):s], k, axis=1)
            d = np.quantile(max_stat, q=1-alpha)
          
        else:
            count = 0
            while binom(rej-count, k-1) > n_max:
                count +=1
                
            index_mat = np.array(
                list(itertools.combinations(range(count, rej), k-1)))
            comb = index_mat.shape[0]
            d_vec  = np.repeat(0.0, comb)            
            for i in range(comb):
                index_rej = index_mat[i, ]
                if rej < s:
                    idx = np.concatenate((index_rej,
                                          np.array(list(range(rej, s)))))
                    max_stat = k_max(z_null_ord[:,idx], k, axis=1)
                d_vec[i] = np.quantile(max_stat, q=1-alpha)
                
            d = max(d_vec)
                                    
        # Save critical values
        crit_val.append(d)
            
        # Stop loop if t-stat is smaller than critical value
        if z_ord[rej] <= d:
          break
        
        # Identify significant t-stats (z_i > d)     
        while z_ord[rej] > d:
            hypo_rej[rej] = r[rej]
            step_rej[rej] = j
            rej +=1
            if rej >= s:
                break
        
        # Stop if less than k hypothesis are rejected
        if rej < k: 
          break
        
        # Increase loop iterator
        j +=1  
       
    return {'hypo_rej': hypo_rej[~np.isnan(hypo_rej)],
            'step_rej': step_rej[~np.isnan(step_rej)], 
            'crit_val': [round(x, digits) for x in crit_val]}

def mht_adj_pvalues(tval, tval_boot, digits=3):
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
   
    # Ensure absolute t-statistics
    tval = np.absolute(tval)
    tval_boot = np.absolute(tval_boot)
    
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










