# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 13:07:41 2023

@author: smalswad
"""

import numpy as np

def calc_hhi(weights):
    '''
    Calculate Herfindahl Hirsch index.

    Parameters
    ----------
    grp : pd.DataFrame, shape(T,n)
        Portfolio weights to calculate HHI from.

    Returns
    -------
    pd.Series, shape(T,)
        HHI for each point in time.

    '''
    return (weights**2).sum(axis=1)

def calc_perc(data):
    '''
    Calculate relative values (percentage) based on row-sums.

    Parameters
    ----------
    data : pd.DataFrame, shape(T,n)
        Nominal data.

    Returns
    -------
    pd.DataFame, shape(T,n)
        Relative values for each point in time.

    '''
    
    return data.divide(data.sum(axis=1), axis=0)
def calc_return(ret, N):
    '''
    Calculate annualised return.

    Parameters
    ----------
    ret : pd.Series, shape (T,)
        Return series.
    N : int
        Number of observations per year, i.e. 250 for daily or 52 for monthly
        data.

    Returns
    -------
    float
        Average annualised return.

    '''    
    return ret.mean() * N

def calmar_ratio(ret,N):
    '''
    Calculate annualised calmar ratio.

    Parameters
    ----------
    ret : pd.Series, shape (T,)
        Return series.
    N : int
        Number of observations per year, i.e. 250 for daily or 52 for monthly
        data.

    Returns
    -------
    float
        Calmar ratio.

    '''
    return calc_return(ret, N)/abs(max_drawdown(ret))

def expected_shortfall(returns, confidence_level=.05):
	"""
	It calculates the Expected Shortfall (ES) of some time series. It represents 
	the average loss according to the Value at Risk.
	
	Parameters
	----------
	returns : pandas.DataFrame
		Returns of each time serie. It could be daily, weekly, monthly, ...
		
	confidence_level : int
		Confidence level. 5% by default.
			
	Returns
	-------
	es : pandas.Series
		Expected Shortfall for each time series.
	
	"""
	
	# Calculating VaR
	var = value_at_risk(returns, confidence_level)
	
	# ES is the average of the worst losses (under var)
	return returns[returns.lt(var)].mean()

def max_drawdown(ret):
    '''
    Calculate maximum drawdown

    Parameters
    ----------
    ret : pd.Series, shape(T,)
        Return data.

    Returns
    -------
    pd.Series
        max dd for each column (asset).

    '''
    comp_ret = (ret+1).cumprod()
    peak = comp_ret.expanding(min_periods=1).max()
    dd = (comp_ret/peak)-1
    
    return dd.min()
   
def sharpe_ratio(ret, N, rf=None):
    '''
    Calculate annualised shapre ratio

    Parameters
    ----------
    ret : pd.Series, shape (T,)
        Return data.
    N : int
        Number of observations per year, i.e. 250 for daily or 52 for monthly
    rf : series, optional
        Risk-free rate. The default is None.

    Returns
    -------
    pd.Series
        Sharpe ratio.

    '''    
    if rf is None:
        rf = 0
    else:
        rf = rf.mean()
        
    return (ret.mean() - rf) / ret.std() * np.sqrt(N)

def sortino_ratio(series, N):
    '''
    Calculate annualised sortino ratio

    Parameters
    ----------
    series : pd.Series, shape(T,)
        Return series.
    N : int
        Number of observations per year, i.e. 250 for daily or 52 for monthly 
        data.
    Returns
    -------
    float
        Sortino ratio.

    '''
    mean = calc_return(series, N) 
    std_neg = series[series<0].std()*np.sqrt(N)
    
    return mean/std_neg

def value_at_risk(returns, confidence_level=.05):
	"""
	It calculates the Value at Risk (VaR) of some time series. It represents 
	the maximum loss with the given confidence level.
	
	Parameters
	----------
	returns : pandas.Series
		Returns of each time serie. It could be daily, weekly, monthly, ...
		
	confidence_level : int
		Confidence level. 5% by default.
			
	Returns
	-------
	var : pandas.Series
		Value at Risk for each time series.
	
	"""
	
	# Calculating VaR
	return returns.quantile(confidence_level, interpolation='higher')

def volatility(ret, N):
    '''
    Calculate annualised volatility

    Parameters
    ----------
    ret : pd.Series, shape(T,)
        Return series.
    N : int
        Number of observations per year, i.e. 250 for daily or 52 for monthly 
        data.

    Returns
    -------
    pd.Series
        Annualised volatility.

    '''
    
    return ret.std() * np.sqrt(N)