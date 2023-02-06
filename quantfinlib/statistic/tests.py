# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 14:28:24 2023

@author: smalswad
"""

from quantfinlib.statistic.regression import LinearRegression
import numpy as np
from scipy import stats

import pandas as pd


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
    sigma = residuals.T.dot(residuals).values/(tau-k-1) # Calcualte sigma estimate
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
