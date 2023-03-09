# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 10:54:53 2022

@author: ROB6738
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm

from sklearn.linear_model import LinearRegression
from numpy import linalg 
from scipy.linalg import sqrtm

from quantfinlib.portfolio.portfolio import risk_comp

#Create mapping dict for data frequencies
FREQUENCIES = {"monthly": 12, "weekly": 52, "daily": 252}

class CAPM(object):
    '''
    Construct CAPM model
    PARAMETERS:
        rf          [float] - risk free rate
        ret_m       [df]    - market returns
        ret_stock   [df]    - stock returns
    '''
    def __init__(self, rf, ret_m, ret_stock):
        if rf is not None:
            self.rf = rf
        else:
            self.rf = 0
        self.ret_m = ret_m.astype(np.float64)
        self.ret_stock = ret_stock.astype(np.float64)
        self.alpha = np.NAN
        self.alpha_se = np.NAN
        self.alpha_tvalue = np.NAN
        self.beta = np.NAN
        self.beta_se = np.NAN
        self.beta_tvalue = np.NAN
        self.conf_int = np.NAN
        self.rsqr = np.NAN
        self.start = max(self.ret_m.index[0], self.ret_stock.index[0])
        self.end = min(self.ret_m.index[-1], self.ret_stock.index[-1])
        
    def evaluate(self, alpha_inc = True, conf_alpha=0.05, hyp='x1=1'):
        '''
        Estimate model parameters: alpha, beta, tstats (H0: alpha=0, beta=1), conf_int, rsqr
        Returns beta, if called directly
        '''
        x = np.array(self.ret_m.sort_index()) - self.rf
        y = np.array(self.ret_stock.sort_index()) - self.rf
        if alpha_inc == True:
            x = sm.add_constant(x)
        model = sm.OLS(y,x, missing='drop')
        results = model.fit()
        
        #save estimation results and tstats
        self.beta = results.params[-1]
        self.beta_se = results.bse[-1]
        self.beta_tvalue = results.t_test(hyp).tvalue[-1][0]
        if alpha_inc==True:
            self.alpha = results.params[0]
            self.alpha_se = results.bse[0]
            self.alpha_tvalue = results.tvalues[0]
        
        self.conf_int = results.conf_int(alpha=conf_alpha)
        self.rsqr = results.rsquared
        
        return self.beta

class FactorModel(object):
    '''
    Attributes:
        factor_names : list, len = K
            Names of factors in specified factor model
        invest_assets : list, len = N
            Names of assets to invest in
        ret_freq : str
            Frequency of return data    
        asset_ret : pd.DataFrame, shape(T,N+x)
            Time-series data of all returns (including but not limited to
            investable assets)
        B : np.ndarray, shape(N,K)
            Loadings matrix of investable asset for factors, i.e. R = B@F
            R in R^(N,1), F in R^(K,1)
        B_inv : np.ndarray, shape(K,N)
            Inverse of B
        mimicking_portf_weights : np.ndarray, shape(N,K)
            Matrix with factor to assets weights     
        torsion : np.ndarray, shape(K,K)
            Torsion matrix. For the standard FactorModel this it identical to 
            np.eye(K)
        model_name : str
            Name of the model
            
        
    Methods:
        evaluate
        name_model
        _create_returns
        _estimate_factor_covariance
        _estimate_loadings        
        __define_mimicking_portfolios
    
    '''    
    
    def __init__(self, factor_ret=None, factor_names=None, asset_ret=None, 
                 mapping_dict=None, asset_to_factor='ew', invest_assets=None,
                 ret_freq="monthly"):
        '''
        Create FactorModel class object with attributes at single point in time,
        i.e. time-series are only used for (co-)variance estimates

        Parameters
        ----------
        factor_ret : pd.DataFrame, optional
            Assign factor returns directly. If unspecified, some value has to be
            passed to asset_ret. The default is None.
        factor_names : list of str, optional
            Names for the factors. If unspecified, names are inferred from
            factor_ret column names. The default is None.
        asset_ret : pd.DataFrame, optional
            Asset returns used for factor return construction. Only relevant
            if factor_ret is not specified. The default is None.
        mapping_dict : dict, optional
            Dictionary containing constituents of each factor. Only relevant
            if factor_ret is not specified. The default is None.
        asset_to_factor : str, optional
            Method to map assets to factors. The default is 'ew'.
        invest_assets: list of str, optional
            Names of investable assets, i.e. those relevant for estimating 
            loadings. If unspecified, all assets are used. The default is None.
        ret_freq : str, optinal
            Frequency of returnd ata. The default is 'monthly'.

        Raises
        ------
        ValueError
            Raises error if factor_ret is not specified correctly.

        Returns
        -------
        None.

        '''        
        # Initialize factor returns
        if isinstance(factor_ret, type(None)):
            self._create_returns(asset_ret, mapping_dict, method=asset_to_factor)
        elif isinstance(factor_ret, pd.DataFrame):
            self.factor_ret = factor_ret
        else: 
            data_type = type(factor_ret)
            raise ValueError("Value passed to 'factor_ret' did not match "
                             "expected data type. Expected pd.DataFrame but "
                             f"got {data_type} instead.")
         
        # Pass variables to class
        self.factor_names = factor_names
        self.invest_assets = invest_assets
        self.ret_freq = ret_freq
        
        
        #Calculate factor covariance matrix
        self.factor_sigma = self._estimate_cov_matrix(ret=self.factor_ret)
        
        #Calculate torsion
        self.torsion = np.eye(self.factor_ret.shape[1])
        
        # Initialize further attributes as None
        self.mimicking_portf_weights = None
        self.model_name = None
        self.standard_model = True
        
    def evaluate(self, package='sklearn'):
        '''
        Evaluate factor model and run calculations sequential
        '''
        self._estimate_loadings(factor_ret=self.factor_ret, package=package)
        self.__define_mimicking_portfolios()
        
    def name_model(self, name):
        self.model_name = name   
        
    def _create_returns(self, asset_ret, mapping_dict, method='ew'):
        '''
        Function to create factor returns (based on specific time-series)

        Parameters
        ----------
        asset_ret : pd.DataFrame of shape (T,N)
            Time series of asset returns.
        mapping_dict : dict
            Mapping dictionary with key:value pairs as factor_name:[asset_names]
        method : string, optional
            Define combination method for asset returns to factor returns.
            The default is 'ew'.

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
        
        # Create factor returns based on mapping dictionary 
        factor_ret = list()
        for factor in mapping_dict.keys():
            assets_to_use = mapping_dict.get(factor)
            factor_ret.append(risk_comp(self.asset_ret[assets_to_use],
                                        how=method))

        self.factor_ret = pd.concat(factor_ret, axis=1)
        self.factor_ret.columns = mapping_dict.keys()     
      
    def _estimate_cov_matrix(self, ret):
        '''
        Estimate covariance matrix for givenr return
        '''
        return ret.cov().values * FREQUENCIES[self.ret_freq]
    
    def _estimate_loadings(self, factor_ret, package='sklearn'):
        '''
        Estimate loadings matrix for assets based on given factor returns

        Parameters
        ----------
        factor_ret : pd.DataFrame, shape(T,N)
            Factor returns to regress assets on.
        package: str, optional
            Select statistical package to estimate loadings with. "sklearn"
            only calculates the coefficient estimates whereas "statsmodels" 
            also estimates tstats and R² values. The defaul is 'sklearn'.

        Returns
        -------
        None.

        '''
        if package == 'sklearn':
            multi_lin_reg = LinearRegression(fit_intercept=False). \
                fit(factor_ret.values,
                    self.asset_ret.loc[:, self.invest_assets].values)
            
            # B := Loadings matrix for R = B F for R in R^Nx1, B in R^NxK, F in R^Kx1
            self.B = multi_lin_reg.coef_
        
        elif package == 'statsmodels':
            param_list, r2_list, tval_list = [], [], []
            for asset in self.invest_assets:
                multi_lin_reg = sm.OLS(self.asset_ret.loc[:, asset].values,
                                       factor_ret.values)
                results = multi_lin_reg.fit()
                param_list.append(results.params)
                r2_list.append(results.rsquared)
                tval_list.append(results.tvalues)
            
            self.B = np.stack(param_list, axis=0)  
            self.tvalues = np.stack(tval_list, axis=0)
            self.R2 = np.stack(r2_list)
        
        # Calculate inverse of loadings matrix B
        self.B_inv = np.linalg.pinv(self.B)
                
    def __define_mimicking_portfolios(self):
        '''
        Define factor mimicking portfolio (FMP) weights
        '''
        self.mimicking_portf_weights = self.B_inv.copy().T
        


class OrthFactorModel(FactorModel):
    
    def __init__(self, model='minimum-torsion', **kwargs):
        '''
        Create orthogonal FactorModel class object with attributes at single 
        point in time, i.e. time-series are only used for (co-)variance 
        estimates.

        Parameters
        ----------
        model : str, optional
            Method to choose to estimate the torsion matrix.
            The default is 'minimum-torsion'.
        **kwargs : TYPE
            see documentation of FactorModel class.

        Returns
        -------
        None.

        '''
        super().__init__(**kwargs)
        self._orthogonalize(model=model)
           
    
    def __define_mimicking_portfolios(self):
        '''
        Calculate factor mimicking portfolio (FMP) weights
        '''
        self.mimicking_portf_weights = self.B_inv.T @ self.torsion.T
        
        
    def evaluate(self, package='sklearn'):
        '''
        Evaluate factor model and run calculations sequential
        '''
        self._estimate_loadings(factor_ret=self.factor_ret, package=package)
        self.__define_mimicking_portfolios()
        self.orth_factor_sigma = \
            self._estimate_cov_matrix(ret=self.orth_factor_ret)
            
            
    def _orthogonalize(self, model='minimum-torsion'):
        '''
        Orthogonalize original factor returns

        Parameters
        ----------
        model : str, optional
            Choose model to calculate torsion matrix. 
            The default is 'minimum-torsion'.

        Returns
        -------
        pd.DataFrame
            Orthogonalized factor returns.
        np.matrix
            Torsion matrix (such that F_orth = torsion * F_original for each
                            point in time and factor returns shape(K,)).

        '''
        
        #Calc torsion matrix
        self.torsion = torsion(cov_mat=self.factor_sigma, model=model)
        
        #Orthogonalize factor retuns
        self.orth_factor_ret = self.factor_ret.dot(self.torsion.T)
        self.orth_factor_ret.columns = self.factor_names
        
        self.standard_model = False
        
        
    
    
def torsion(cov_mat, model, method='exact', max_niter=10000):
    '''
    Function to calculate the torsion matrix as introduced in Meucci (2013)
    Code taken from: https://github.com/HarperGuo/Risk-Parity-and-Beyond/blob/master/torsion.py
    
    Parameters
    ----------
    cov_mat : np.matrix
        Covariance matrix of returns to orthogonalize.
    model : 'pca' or 'minimum-torsion'
        Define model estimation approach.
    method : 'exact' or 'approximate'
        Define method estimation for 'minimum-torsion' approach. The default is 'exact'.
    max_niter : int, optional
        Max number of iterations for 'exact' method. The default is 10000.

    Returns
    -------
    t : np.matrix
        Torsion matrix necessary to orthogonalize given returns.

    '''
    n = cov_mat.shape[0]    
    
    if model == 'pca':
        eigval, eigvec = linalg.eig(cov_mat)
        idx = np.argsort(-eigval) 
        t = eigvec[:,idx]
        
    elif model == 'minimum-torsion':
        # C: correlation matrix
        sigma = np.sqrt(np.diag(cov_mat))
        C = np.asmatrix(np.diag(1.0/sigma)) * np.asmatrix(cov_mat) * np.asmatrix(np.diag(1.0/sigma))
        # Riccati root of correlation matrix
        c = sqrtm(C)
        if method == 'approximate':
            t = (np.asmatrix(sigma) / np.asmatrix(c)) * np.asmatrix(np.diag(1.0/sigma))
        elif method == 'exact':
            # initialize
            d = np.ones((n))
            f = np.zeros((max_niter))
            # iterating
            for i in range(max_niter):
                U = np.asmatrix(np.diag(d)) * c * c * np.asmatrix(np.diag(d))
                u = sqrtm(U)
                q = linalg.inv(u) * np.asmatrix(np.diag(d)) * c
                d = np.diag(q * c)
                pi = np.asmatrix(np.diag(d)) * q
                f[i] = linalg.norm(c - pi, 'fro')
                # if converge
                if i > 0 and abs(f[i]-f[i-1])/f[i] <= 1e-4:
                    f = f[0:i]
                    break
                elif i == max_niter and abs(f[max_niter]-f[max_niter-1])/f[max_niter] >= 1e-4:
                    print('number of max iterations reached: n_iter = ' + str(max_niter))
            x = pi * linalg.inv(np.asmatrix(c))
            t = np.asmatrix(np.diag(sigma)) * x * np.asmatrix(np.diag(1.0/sigma))
    return t.A
     
   
    
    
