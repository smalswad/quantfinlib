# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 12:46:23 2023

@author: smalswad
"""

import pandas as pd
import numpy as np

from quantfinlib.optimization.riskparity import calc_rpo

#Create mapping dict for data frequencies
FREQUENCIES = {"monthly": 12, "weekly": 52, "daily": 252}

class Portfolio(object):
    
    def __init__(self, asset_ret, asset_weights=None, asset_weights_method='ew',
                 ret_freq="monthly"):
        '''
        Generate portfolio class object with attributes at single point in time,
        i.e. time-series are only used for (co-)variance estimates

        Parameters
        ----------
        asset_ret : pd.DataFrame, shape(T,N)
            Asset returns over time.
        asset_weights : np.array, shape(N,1), optional
            Asset weights at specific point in time. The default is None.
        asset_weights_method : str, optional
            Method on how to construct asset_weights if not explicitly defined.
            The default is 'ew'.

        Raises
        ------
        ValueError
            Rises value error if inputs don't fulfill data type criteria.

        Returns
        -------
        None.

        '''        
        # Initialize asset returns
        if isinstance(asset_ret, pd.DataFrame):
            self.asset_ret = asset_ret
        else:
            data_type = type(asset_ret)
            raise ValueError("Value passed to 'asset_ret' did not match "
                             "expected data type. Expected pd.DataFrame but "
                             f"got {data_type} instead.")
        
        # Get number of assets N
        self.N = self.asset_ret.shape[1]
        
        # Initialize asset weights
        if isinstance(asset_weights, pd.DataFrame):
            self.asset_weights = asset_weights.values
        elif isinstance(asset_weights, np.ndarray):
            self.asset_weights = asset_weights
        elif asset_weights is None:
            self.calc_asset_weights(method=asset_weights_method)
        else:
            data_type = type(asset_weights)
            raise ValueError("Value passed to 'asset_weights' did not match "
                             "expected data type. Expected pd.DataFrame or "
                             f"np.ndarry but got {data_type} instead")
        
        # Pass further variables to class
        self.ret_freq = ret_freq                
        
    def calc_asset_cov(self):
        '''
        Calculate asset covariance matrix
        '''
        self.asset_sigma = np.cov(self.asset_ret.values, rowvar=False) \
            * FREQUENCIES[self.ret_freq]
            
        return self.asset_sigma    
    
    def calc_asset_risk_contribution(self, how='variance'):
        '''
        Helper function to calculate asset risk contribution based on specified
        risk measure 'how'

        Parameters
        ----------
        how : 'str', optional
            Define how to calculate risk using either variance or std. 
            The default is 'variance'.

        Returns
        -------
        None.

        '''
        # asset_rc := risk contribution by assets, rc in R^Nx1
        if how=='variance':
            self.asset_rc = self.asset_weights \
                * (self.asset_sigma @ self.asset_weights) \
                / self.sig2
        
        elif how=='std':
            self.asset_rc = self.asset_weights.T @ self.asset_sigma \
                / np.sqrt(self.sig2)
        
        else: 
            raise ValueError("Value passed to 'how' did not match expected "
                             "options. Expected 'variance' or 'std'.")            
            
    def calc_port_variance(self):
        self.sig2 = \
            self.asset_weights @ self.asset_sigma @ self.asset_weights.T        
        
    def calc_asset_weights(self, method='ew'):
        if method=='ew':
            self.asset_weights = np.full((self.N, 1), 1/self.N)  
            
        elif method=='inv_vol':
            invvola = 1 / self.asset_ret.std(axis=0)
            self.asset_weights = (invvola/ invvola.sum()).values
            
        else:
            raise ValueError("Value passed to 'method' did not match "
                             "expected input. Expected 'ew' or 'inv_vol' "
                             f"but got {method} instead.")    
    
    def evaluate(self, risk='variance'):
        self.calc_asset_cov()
        self.calc_port_variance()
        self.calc_asset_risk_contribution(how=risk)
            

class FactorPortfolio(Portfolio):
    
    def __init__(self, factor_model, *args, factor_weights=None, 
                 factor_mrc=None, **kwargs):
        '''
        Create a portfolio which is relient on a factor model. 

        Parameters
        ----------
        factor_model : macromod.factormodel.FaktorModel object
            Factor model containing all factor related information.
        factor_weights : np.array, shape(K,), optional
            Factor weights to use. The default is None.
        factor_mrc : list, optional
            Target marginal factor risk contributions, i.e. risk budget of total 
            portfolio risk (e.g. equal risk). The default is None.
        *args : TYPE
            see documentation of Portfolio class.
        **kwargs : TYPE
            see documentation of Portfolio class.

        Returns
        -------
        None.

        '''
        self.model = factor_model 
        self.factor_weights = factor_weights
        self.factor_target_rc = factor_mrc
        super().__init__(*args, **kwargs)     
               
    def calc_asset_weights(self, method='ew'):
        if method=='inv_factor_vola' or method=='fixed_factor_rc':
            if self.factor_weights is None:
                self.calc_factor_weights(method)
                
            # Transform factor weights to asset weights based on loadings
            self.asset_weights = \
                self.model.mimicking_portf_weights @ self.factor_weights 
            
            # Scale asset weights
            self.asset_weights = self.asset_weights/np.sum(self.asset_weights)
        
        else:
            super().calc_asset_weights(method=method)            
    
    def calc_factor_risk_contribution(self):
        '''
        Helper function to calculate factor risk contribution based on specified
        risk definiton 'how'.
        factor_rc := risk contribution by factor, rc in R^Kx1

        '''
        if self.model.standard_model:
            factor_weights = self.asset_weights @ self.model.B
            risk_contr = \
                self.model.B_inv @ self.asset_sigma @ self.asset_weights
            self.factor_rc = factor_weights * risk_contr / np.sqrt(self.sig2)
            
        else:
            #Orth factor volas
            lambda_ = np.sqrt(np.diag(self.model.orth_factor_sigma))
            
            #Inverse torsion matrix
            inv_tor = np.linalg.inv(self.model.torsion)
            
            #Orth factor weights
            implied_vola_per_factor = \
                self.asset_weights @ self.model.B @ inv_tor @ np.diag(lambda_)
        
            #MRC of factors
            self.factor_rc = implied_vola_per_factor**2 / np.sqrt(self.sig2)            
            
    def calc_factor_weights(self, method='inv_factor_vola'):
        if method=='assets_to_factors_via_loadings':
            self.factor_weights = \
                np.linalg.pinv(self.model.mimicking_portf_weights) \
                    @ self.asset_weights
            
        elif method=='inv_factor_vola':
            # Calculate factor weights based on inv factor vola
            if self.model.standard_model: 
                lambda_ = np.diag(self.model.factor_sigma)
            else:
                lambda_ = np.diag(self.model.orth_factor_sigma)
            
            self.factor_weights = \
                (1/np.sqrt(lambda_)) / np.sum(1/np.sqrt(lambda_))  
            
        elif method=='fixed_factor_rc':
            # Calculate factor weights based on fixed factor risk contriutions
            if self.model.standard_model: 
                sigma_ = self.model.factor_sigma
            else:
                sigma_ = self.model.orth_factor_sigma
            
            # Define initial weights just as EW
            w0 = np.array([1/sigma_.shape[0]] * sigma_.shape[0])                
            self.factor_weights = calc_rpo(w0, sigma_, self.factor_target_rc)
    
    def calc_number_of_bets(self):
        '''
        Calculate number of uncorrelated bets (based on factors)
        '''
        #Scale factor rc to ignore idiosyncratic risk as additional source (bet)
        scaled_factor_rc = self.factor_rc / self.factor_rc.sum()
        
        #Avoid risk contributions close to zero
        scaled_factor_rc = np.maximum(10**-10, scaled_factor_rc)
        
        self.number_of_bets = \
            np.exp((-scaled_factor_rc).dot(np.log(scaled_factor_rc)))    
    
    def evaluate(self, risk='variance'):
        self.calc_asset_cov()
        self._factor_risk_wrapper()
        self.calc_asset_risk_contribution(how=risk)        
        
    def _factor_risk_wrapper(self):
        '''
        Wrapping function of sequential execution of class methods
        '''
        self.calc_port_variance()
        
        if self.factor_weights is None:
            self.calc_factor_weights('assets_to_factors_via_loadings')
        
        self.calc_factor_risk_contribution()
        self.calc_number_of_bets()
         
               
class LargeFactorPortfolio(FactorPortfolio):    
    ''' 
    Create subclass of FactorPortfolio which has too many assets to estimate
    the asset covariance matrix directly. Instead, approximate via factor
    estimation and transformation from factor model via loadings matrix.
    '''
     
    def estimate_asset_cov(self):
        '''
        Calculate portfolio risk accordingly to BARRA model
        https://www.alacra.com/alacra/help/barra_handbook_GEM.pdf
    
        Delta := diagonal matrix of idiosyncratic risk in R^NxN
        '''        
        est_ret = self.factor_ret.dot(self.model.B.T)
        u = self.asset_ret - est_ret
        Delta = np.diag(u.std(axis=0).flatten()**2)
        
        self.asset_sigma = \
            np.matmul(np.matmul(self.model.B, self.model.factor_sigma),
                      self.B.T) + Delta
            
    def evaluate(self):
        self.estimate_asset_cov()
        self._factor_risk_wrapper()  


def calc_portf_ret(ret, w, rf=None, excess=True):
    '''
    Calculate portfolio returns over time

    Parameters
    ----------
    ret : pd.DataFrame, shape(T,N+x)
        Returns for individual assets.
    w : pd.DataFrame, shape(T,N)
        Portfolio asset weights.
    rf : pd.Series, shape(T,1), optional
        Risk free rate over time. The default is None.
    excess : boolean, optional
        Are returns already given as excess returns? If rf is not None, this
        parameter specifies whether rf should be subtracted (False) or
        added (True)

    Returns
    -------
    pd.Series, shape(T,) : portfolio returns

    '''
    ret = ret.loc[w.index]
    
    portf_ret = pd.Series(np.diag(w @ ret.T), index=w.index)
    
    if rf is not None:
        if excess:
            portf_ret = portf_ret + rf.loc[portf_ret.index]
        else:
            portf_ret = portf_ret - rf.loc[portf_ret.index]
    
    return portf_ret

def risk_comp(data, how, data2=None):
    '''
    Calculate risk composition (returns) based on specified method

    Parameters
    ----------
    data : pd.DataFrame
        Return data.
    how : str
        Method to choose for risk composition. Choose from ('ew', 'erc', 'inv_vol')
    data2 : pd.DataFrame, optional
        Second return data. The default is None.

    Raises
    ------
    ValueError
        Rises error if undefined method for risk composition is passed.

    Returns
    -------
    out : pd.Series
        New return series based on specified risk composition method.

    '''
    out=None
    if isinstance(data, type(None)):
        data2 = data
        
    if how=='ew':
        #EQUAL WEIGHTS
        out = data.mean(axis=1)
        
    elif how=='erc':
        #EQUAL RISK CONTRIBUTION
        cov_mat = data.cov().values
        w0 = np.array([1/cov_mat.shape[0]] * cov_mat.shape[0])    
        mrc = [1/cov_mat.shape[0]] * cov_mat.shape[0]
        
        #Calculate erc
        erc_weights = calc_rpo(w0, cov_mat, mrc)
        
        #Calculate return series based on erc
        out = pd.Series((data2.values * np.asmatrix(erc_weights).T).A1,
                        index=data.index)
        
    elif how=='inv_vol':
        #INVERSE VOLATILITY
        invvola = 1 / data.std(axis=0)
        invvol_weights = invvola/ invvola.sum()
        out = data2.dot(invvol_weights)
    
    else:
        raise ValueError("Please choose appropriate method to construct risk"
                         "composition parameter 'how'.")
                         
    return out

def vw_ret(group, avg_name, weight_name):
    '''
    Helper function to calculate value-weighted portfolio returns

    Parameters
    ----------
    group : pd.DataFrame
        Goupby frame or whole data frame in long format.
    avg_name : string
        Column name of return values.
    weight_name : string
        Column name of determining weight column.

    Returns
    -------
    float
        VW return.

    '''
    d = group[avg_name]
    w = group[weight_name]
    try:
        return (d * w).sum() / w.sum()
    except ZeroDivisionError:
        return np.nan
    
    