# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 14:25:37 2023

@author: smalswad
"""
from scipy import stats
from sklearn import linear_model

import numpy as np
import pandas as pd
import pathlib
import statsmodels.api as sm
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class LinearRegression(linear_model.LinearRegression):

    def __init__(self,*args,**kwargs):
        # *args is the list of arguments that might go into the LinearRegression object
        # that we don't know about and don't want to have to deal with. Similarly, **kwargs
        # is a dictionary of key words and values that might also need to go into the orginal
        # LinearRegression object. We put *args and **kwargs so that we don't have to look
        # these up and write them down explicitly here. Nice and easy.

        if not "fit_intercept" in kwargs:
            kwargs['fit_intercept'] = False

        super().__init__(*args,**kwargs)

    # Adding in t-statistics for the coefficients.
    def fit(self,x,y):
        # This takes in numpy arrays (not matrices). Also assumes you are leaving out the column
        # of constants.

        # Fit model (estimate coefficients)
        self = super().fit(x,y)
        n, k = x.shape
        y_hat = np.matrix(self.predict(x))

        # Change X and Y into numpy matricies. x also has a column of ones added to it.
        x = np.hstack((np.ones((n,1)),np.matrix(x)))
        y = np.matrix(y)

        # Degrees of freedom.
        df = float(n-k-1)
        
        # Residuals
        self.residuals = y - y_hat
        residual_sum_of_squares = self.residuals.T @ self.residuals
        
        # Estimate standard errors per regression
        se_list = []
        for i in range(residual_sum_of_squares.shape[0]):
            # Get variance estimates
            sigma_squared_hat = residual_sum_of_squares[i,i] / df
            # Estimate covariance matrix 
            cov_mat = np.linalg.inv(x.T @ x) * sigma_squared_hat
            # Collect regression specific standard errors
            se_list.append(np.sqrt(cov_mat.diagonal()))
        
        # Transform standard errors to matrix
        self.se = np.concatenate(se_list)
        
        # Concat alphas and betas to one coefficient matrix
        coef = np.hstack((
            np.reshape(self.intercept_, (len(self.intercept_),1)),
            self.coef_))
        
        # T statistic for all coefficients
        self.tstats_ = np.divide(coef, self.se)

        # P-value for each beta. This is a two sided t-test, since the betas can be 
        # positive or negative.
        self.pvalues = 1 - stats.t.cdf(abs(self.tstats_),df)
        
        return self

class MultiFactorModel:
    def __init__(self, reg_data, dep_var, ind_var, alpha_inc=True):
        '''
        Create a multi factor class with severval statistics (but only capable
        of one dependant variable).
        
        Parameters
        ----------
        reg_data : dataframe
            All the return data used to estimate the factor model.
        dep_var : string
            column name of dependant variable.
        ind_var : list
            List of column names of independant variables.
        alpha_inc : boolean
            Indicate whether factor model includes an intercept, i.e. alpha

        Returns
        -------
        None.

        '''
        self.reg_data = reg_data
        self.dep_var = dep_var
        self.ind_var = ind_var
        self.alpha_inc = alpha_inc         
        self.evaluate()
        
        
    def evaluate(self, conf_alpha=0.05):

        reg_formula = self.dep_var + ' ~ ' + ' + '.join(self.ind_var)
        
        if not self.alpha_inc:
            reg_formula = reg_formula + '-1'
        
        self.model = sm.formula.ols(formula=reg_formula, data=self.reg_data).fit()
        self.coef = self.model.params.rename('coef')
        self.se = self.model.bse.rename('se')
        self.tvalues = self.model.tvalues.rename('tval')
        self.rsqr = self.model.rsquared
        self.conf_int = self.model.conf_int(alpha=conf_alpha).rename({0:'lb',1:'ub'}, axis=1)
        
        
        
    def save_summary(self, path, filename, output_format = 'latex'):

        #Create output summary table
        summaryTable = self.model.summary()
        
        # Determine output format
        if output_format == 'latex':
            summaryTable = summaryTable.as_latex()
        else:
            summaryTable = summaryTable.as_text()
        
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        # Save results as text file
        with open(path + filename, 'w') as f:
            f.write(summaryTable)

def multiple_ols(dep_var, indep_var, incl_alpha=True):
    '''
    Run multiple (multivariate) OLS regressions

    Parameters
    ----------
    dep_var : pd.DataFrame, shape(T,n)
        Dependent variables.
    indep_var : pd.DataFrame, shape(T,k)
        Independent variables.
    incl_alpha : bool, optional
        Indicate whether a constant should be included. The default is True.

    Returns
    -------
    dict
        Result dict with variables:
            params  -   coefficients
            res     -   residuals
            tstats  -   t-statistics.
            se      -   standard errors
    '''
    
    params = list()
    residuals = list()
    tstats = list()
    se = list()
    
    # Add intercepts
    if incl_alpha:
        indep_var = sm.add_constant(indep_var)
    
    # Run regression for each dependent variable
    for i in range(dep_var.shape[1]):             
        model = sm.OLS(dep_var.iloc[:,i], indep_var).fit()
        params.append(model.params)
        residuals.append(model.resid)
        tstats.append(model.tvalues)
        se.append(model.bse)
        
    return {'params':   pd.concat(params, axis=1),
            'res':      pd.concat(residuals, axis=1),
            'tstats':   pd.concat(tstats, axis=1),
            'se':       pd.concat(se, axis=1)}
  