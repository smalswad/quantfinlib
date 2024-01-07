# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 14:18:00 2024

@author: Alexander Swade
"""

import numpy as np
import pandas as pd

from math import log, sqrt, exp
from scipy import stats

class EuropeanOption(object):
    ''' Class for European options in BSM Model.
    
    Attributes
    ==========
    S0 : float
        initial stock/index level
    K : float
        strike price
    t : datetime/Timestamp object
        pricing date
    M : datetime/Timestamp object
        maturity date
    r : float
        constant risk-free short rate
    sigma : float
        volatility factor in diffusion term
        
    Methods
    =======
    value : float
        return present value of call option
    vega : float
        return vega of call option
    imp_vol : float
        return implied volatility given option quote
    '''
    
    def __init__(self, S0, K, t, M, r, sigma, otype='call'):
        self.S0 = float(S0)
        self.K = K
        self.t = t
        self.M = M
        self.r = r
        self.sigma = sigma
        self.otype = otype
        
        
    def _update_ttm(self):
        ''' Updates time-to-maturity self.T. '''
        if self.t > self.M:
            raise ValueError("Pricing date later than maturity.")
        self.T = (self.M - self.t).days / 365.
        
    def _d1(self):
        ''' Helper function to calculate d1 of BSM model. '''
        nom = np.log(self.S0/self.K)+(self.r+ 0.5*self.sigma**2)*self.T
        denom = self.sigma*sqrt(self.T)
        
        self.d1 = nom/denom
        
    def _d2(self):
        ''' Helper function to calculate d2 of BSM model. '''
        self.d2 = self.d1 - self.sigma*sqrt(self.T)

    def _calc_value(self):
        ''' Helper function to calculate option value'''
        
        value = (self.S0 * stats.norm.cdf(self.d1, 0.0, 1.0) \
            - self.K * exp(-self.r * self.T) * stats.norm.cdf(self.d2, 0.0, 1.0))
        return value
        
    def __call__(self):
        '''
        Override call funtion to create function object, i.e., call other
        functions as specified.
        
        Returns
        -------
        value : float
            Fair option value accoirding to BSM model.

        '''
        
        # Calculate all helper values
        self._update_ttm()
        self._d1()
        self._d2()
        
        return self._calc_value()
        
# =============================================================================
# Test only
# =============================================================================
if __name__ == '__main__':
    euro_call = EuropeanOption(S0=110, K=100, t=pd.Timestamp('30-09-2014'),
                               M=pd.Timestamp('30-09-2018'), r=0.01, sigma=0.2)
    
    print(f"The value of the {euro_call.otype}-option is: {euro_call():.2f}")
    
    