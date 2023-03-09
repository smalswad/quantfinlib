# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 14:28:24 2023

@author: smalswad
"""
import itertools
import tqdm
import multiprocessing as mp
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import binom

from quantfinlib.statistic.regression import LinearRegression
from quantfinlib.statistic.tests_parallel_function import _run_boot
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
    return ((tvals >= t).sum() + 1)/(len(tvals) + 1)

def fdp_k_fwer(z, z_null, gamma, alpha=0.05, n_max=50, disp=False):
    '''
    Calculate the false discovery proportion (FDP) procedure of Romano, Shaikh,
    and Wolf and control for the probability that FDP is below a certain 
    threshold. The procedure keeps increasing the k order of the k-fwer until
    it reaches the desired FDP control.

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

def factor_cand_test(ret, pre, cand, bs_indices, min_obs=36, print_res=False,
                     parallel=False):
    '''
    Factor test to identify significant factors of candidate set. Based on
    
    Harvey, C. R., & Liu, Y. (2021). Lucky factors. 
    Journal of Financial Economics, 141(2), 413-435.
    
    including multiple hypothesis tests based on 
    
    Romano, J. P., & Wolf, M. (2016). Efficient computation of adjusted 
    p-values for resampling-based stepdown multiple testing. 
    Statistics & Probability Letters, 113, 38-40. 
    
    Parameters
    ----------
    ret : pd.DataFrame, shape(T, N)
        Excess returns of N stocks/factors. Missing data are given as nan.
    pre : pd.DataFrame, shape(T, kp)
        Return data of pre-selected factors, no missing data.
    cand : pd.DataFrame, shape(T, kc)
        Return data of candidate factors, no missing data.
    bs_indices: List of lists 
        List of k different bootstrap samples with length T each containing 
        positional row indices.  
    min_obs : int, optional
        Minimum number of observations per time-series to be included. The 
        default is 36.
    print_res : boolean, optional
        Indicate whether results shall be printed. the default is False.
    parallel : boolean, optional
        Indicate whether bootstrapped calculations shall be run parallel. 
        The default is False.

    Returns
    -------
    statistics : pd.DataFrame, shape(kc,2)
        test statistics of factor candidates being relevant contendors based on
        means (column 0) and medians (column 1).
    pvals : pd.DataFrame, shape(kc,2)
        P-values of factor candidates being relevant contendors based on mean 
        (c0) and median (c1) statistics, repsectively.
    mht_res : pd.DataFrame, shape(y,2x6)
        MHT results based on Romano & Wolf for negative t-stats only. Contains 
        t-stats, single p-values and multiple test p-values based on mean and 
        median based test statistics.
    mult_pval_min : tuple
        P-values of factor candidates with smallest pvalue after accounting for
        multiple testing based on mean (c0) and median (c1) statistics, 
        respectively. These pvalues are as defined in Harvey & Liu.

    '''
   
    T,N = ret.shape
    
    # Ensure that inputs are pd.DataFrames
    pre = pd.DataFrame(pre)
    cand = pd.DataFrame(cand)
    
    ### Original regression
    sum_mm, sum_dd = _avg_statistics(ret, pre, cand, min_obs=min_obs)
    
    ##### Bootstrap #######
    bslen = len(bs_indices)        
    
    # Calculate orthogonal factor candidates, i.e. here just adjust mean
    res = _regress(pre.values, cand.values)
    fac_cand = pd.DataFrame(cand.values - res.intercept_,
                            columns=cand.columns) 
    
    if parallel:
    ### RUN PARALLEL LOOP ###
        loop_res = list()
        pool = mp.Pool(mp.cpu_count()-1) 
        parallel_res_objs = [
            pool.apply_async(_run_boot, args=(bs_indices[i], ret, pre,
                                              fac_cand, min_obs)) 
            for i in range(bslen)]
        
        for job in tqdm.tqdm(parallel_res_objs):
            loop_res.append(job.get())
            
        pool.close()
        pool.join()
        
        # Organize outputs
        boot_mm = np.column_stack([loop[0] for loop in loop_res])
        boot_dd = np.column_stack([loop[1] for loop in loop_res])
    
    else:
    ### RUN SEQUENTIAL LOOP ###
        boot_mm = list() # mc x b_size matrix of mean-based test statistics under null; rows are factors; columns are bootstrap samples
        boot_dd = list() # mc x b_size matrix of mean-based test statistics under null; rows are factors; columns are bootstrap samples
        
        # Iterate over bootstrap samples
        for i, boot in enumerate(bs_indices):
            
            if i % 100 == 0:
                print(f'Calculating {i}-th bootstrap sample.')
            sum_mm_bb, sum_dd_bb = \
                _run_boot(boot, ret, pre, fac_cand, min_obs=min_obs)
            boot_mm.append(sum_mm_bb)
            boot_dd.append(sum_dd_bb)
        
        # Convert results to proper sized matrix
        boot_mm = np.array(boot_mm).T
        boot_dd = np.array(boot_dd).T
    
    # Calculate single-test p-values
    pval_mm = ((boot_mm < sum_mm[:, None]).sum(axis=1) + 1) / (bslen + 1)
    pval_dd = ((boot_dd < sum_dd[:, None]).sum(axis=1) + 1) / (bslen + 1)
    
    # Calculate multiple-test p-values
    pval_mm_mult = ((boot_mm.min(axis=0) < sum_mm.min()).sum()+1) / (bslen + 1)
    pval_dd_mult = ((boot_dd.min(axis=0) < sum_dd.min()).sum()+1) / (bslen + 1)
    
    if print_res:
        print(f'Output:\n'
              f'Mean-based tests:\n'
              f'Single-test 10th percentile = {np.quantile(boot_mm, 0.1, axis=0).round(3)}\n'
              f'Single-test 5-th percentile = {np.quantile(boot_mm, 0.05, axis=0).round(3)}\n'
              f'Single-test p-values = {pval_mm.round(3)}\n'
              f'Multiple-test p-value = {round(pval_mm_mult, 3)}\n')
        print(f'Output:\n'
              f'Median-based tests:\n'
              f'Single-test 10th percentile = {np.quantile(boot_dd, 0.1, axis=0).round(3)}\n'
              f'Single-test 5-th percentile = {np.quantile(boot_dd, 0.05, axis=0).round(3)}\n'
              f'Single-test p-values = {pval_dd.round(3)}\n'
              f'Multiple-test p-value = {round(pval_dd_mult, 3)}\n')
   
    # Collect test statistics
    statistics = pd.DataFrame(np.column_stack((sum_mm, sum_dd)),
                              columns=['mean-based', 'median-based'],
                              index=cand.columns)
    
    # Collect single-test p-values
    pvals = pd.DataFrame(np.column_stack((pval_mm, pval_dd)),
                         columns=['mean-based', 'median-based'],
                         index=cand.columns)
    
    # Calculate multiple-test p-values based on Romano and Wolf (2016) 
    # Use negative t-stats in order to prioritize factors reducing mean alpha
    # the most with two-sided test (i.e. set absolute=False)
    
    # Mean-based
    mm_res = pd.DataFrame(mht_adj_pvalues(-sum_mm, -boot_mm.T,
                                          absolute=False, digits=16),
                          index=cand.columns,
                          columns=['t-stat', 'pval', 'mht_pval'])
    mm_res['t-stat'] = -mm_res['t-stat']
    
    # Median based
    dd_res = pd.DataFrame(mht_adj_pvalues(-sum_dd, -boot_dd.T,
                                          absolute=False, digits=16),
                          index=cand.columns,
                          columns=['t-stat', 'pval', 'mht_pval'])
    dd_res['t-stat'] = -dd_res['t-stat']         
    
    mht_res = pd.concat([mm_res, dd_res], axis=1, names=['method', 'metric'],
                        keys=['mean-based', 'median-based']).sort_index()
    
    # Original multiple-test p-values 
    mult_pval_min = (pval_mm_mult, pval_dd_mult)
    
    return (statistics, pvals, mht_res, mult_pval_min)    
    
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

def mht_adj_pvalues(tval, tval_boot, digits=8, absolute=True):
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
   
    if absolute:
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

def tstat(data, t_hat, t0):
    '''
    Calculate t-statistic based on estimate t_hat of value t0

    Parameters
    ----------
    data : np.array
        Original data.
    t_hat : flaot
        Estimate of true value.
    t0 : float
        True value to be tested.

    Returns
    -------
    t : float
        T-statistic.

    '''
   
    t = (t_hat - t0) / (np.std(data)/np.sqrt(len(data)))
    return t

def _avg_statistics(ret, pre, cand, min_obs=36):
    '''
    Helper function for factor_cand_test(). Calculate average (mean- and median-based)
    statistics for factor significance test.

    Parameters
    ----------
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
    T,N = ret.shape
   
    # Initialize storing lists
    sum_mm = list()
    sum_dd = list()
    
    # Iterate over candidate factors
    for kk in cand.columns:
        cand_ret = cand.loc[:,kk]
        
        # Run regressions factor by factor if any values are missing (slow)
        if ret.isna().sum().sum() != 0:
            icept0 = list() # store cross-section of scaled intercepts under X0
            icept1 = list() # store cross-section of scaled intercepts under X0 + one candidate factor
            
            for nn in range(N):
                ret_n = ret.iloc[:,nn]
                t_ind = ~ret_n.isna() # if not missing, t_ind==1
                
                if t_ind.sum() < min_obs:
                    icept0.append(np.nan)
                    icept1.append(np.nan)
                
                else:
                    # Gather regression vectors/matrices as numpy arrays
                    ret_nm = ret_n.loc[t_ind].values # conditional on non-missing
                    
                    # Desgin matrix under null
                    reg0 = np.column_stack((np.ones(t_ind.sum()),
                                           pre.loc[t_ind].values))
                    # Design matrix under alternative
                    reg1 = np.column_stack((reg0, cand_ret.loc[t_ind].values))
                    
                    # Betas under null
                    inv_reg0 = np.linalg.inv(reg0.T.dot(reg0))
                    beta0 = inv_reg0.dot(reg0.T).dot(ret_nm)
                    # Betas under alternative
                    beta1 = np.linalg.inv(
                        reg1.T.dot(reg1)).dot(reg1.T).dot(ret_nm)
                    
                    ### Calculate test statistics ###
                    res0 = ret_nm - reg0.dot(beta0) # residual vector under null
                    
                    nobs, nvar = reg0.shape
                    sig2e = (res0.T.dot(res0))/(nobs-nvar)
                    
                    # standard errors for coefficients
                    olsse = np.sqrt(sig2e*np.diag(inv_reg0)) 
                    
                    icept0.append(np.abs(beta0[0]/olsse[0]))
                    icept1.append(np.abs(beta1[0]/olsse[0]))
            
            icept0 = np.array(icept0)
            icept1 = np.array(icept1)
        
        # If no data is missing, run multiple OLS (fast)
        else: 
            
            # Regression under null
            res0 = _regress(pre.values, ret.values)
            icept0 = np.abs(res0.intercept_ / 
                            np.asarray(res0.se[:,0]).squeeze())
            
            # Regression under alternative
            res1 = _regress(pd.concat([pre, cand_ret], axis=1).values,
                            ret.values)
            icept1 = np.abs(res1.intercept_ / 
                            np.asarray(res0.se[:,0]).squeeze())
            
        # Calculate average (mean- or median-based) statistics     
        sum_mm.append(
            (np.nanmean(icept1)-np.nanmean(icept0))/np.nanmean(icept0))
        sum_dd.append(
            (np.nanmedian(icept1)-np.nanmedian(icept0))/np.nanmedian(icept0))
    
    # Transform average statistics to np.array        
    sum_mm = np.array(sum_mm)    
    sum_dd = np.array(sum_dd)

    return (sum_mm, sum_dd)

def _regress(xx, yy):
    '''
    Helper function for factor_cand_test(). Run (multiple) OLS of yy (lhs) on 
    xx (rhs). 

    Parameters
    ----------
    xx : np.array, shape(T,p)
        RHS variables (i.e. return series).
    yy : np.array, shape(T,c)
        LHS variables (i.e. return series).

    Returns
    -------
    res 

    '''
    # Ensure to have ndarrays
    xx = xx.reshape(len(xx),1) if xx.ndim == 1 else xx
    yy = yy.reshape(len(yy),1) if yy.ndim == 1 else yy
    
    # Estimate regressions
    model = LinearRegression(fit_intercept=True)
    res = model.fit(xx, yy)
    
    return res

