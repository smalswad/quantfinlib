# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 17:25:27 2024

@author: Alexander Swade
"""

import pandas as pd
import numpy as np

from typeguard import typechecked
from math import sqrt

from european_options import EuropeanOptionCRR

@typechecked
class AmericanOptionCRR(EuropeanOptionCRR):
    ''' Class for American options based on CRR valuation.
    
    Attributes
    ==========
    N : int
        Number of time intervals
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
        
    '''
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def _calc_value(self):
        '''
        Calculate payoff matrix h, continuing values C, intrinsic values C, and
        execution indicator ex

        Returns
        -------
        float
            Option value at t0.

        '''
        # Get stochastic prices of underlying
        S = self._simulate_underlying_process()
        
        h = self._inner_value(S)  # payoff matrix (i.e. option vs. underlying)
        V = self._inner_value(S)  # value matrix
        C = np.zeros((self.N + 1, self.N + 1), dtype=float)  # continuation values
        ex = np.zeros((self.N + 1, self.N + 1), dtype=float)  # exercise matrix
    
        # Compute the values of continuation, valuation and exercise matrices 
        # via backward induction
        z = 0
        for i in range(self.N - 1, -1, -1):
            
            C[0:self.N-z, i] = (self.q*V[0:self.N-z, i+1] \
                                +(1-self.q)*V[1:self.N-z+1, i+1])*self.df
            V[0:self.N-z, i] = np.where(h[0:self.N-z, i] > C[0:self.N-z,i],
                                        h[0:self.N-z, i],
                                        C[0:self.N-z, i])
            ex[0:self.N-z, i] = np.where(h[0:self.N-z, i] > C[0:self.N-z,i],
                                         1,
                                         0)
            z +=1
        
        self.C, self.V, self.ex = C, V, ex

        return V[0, 0]
    
class AmericanOptionLSMPrimal(EuropeanOptionCRR):
    ''' Class for American options based on Least-Square Monte Carlo Primal
    algorithm valuation.
    
    Attributes
    ==========
    I : int
        Number of simulated paths
    N : int
        Number of time intervals
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
    nbasis_function : int
        Degree of fitting polynomial
        
    '''
    def __init__(self, I, nbasis_function=6, **kwargs):
        super().__init__(**kwargs)
        self.I = I
        self.nbf = nbasis_function

    def _simulate_underlying_process(self):
        ''' Simulate I Paths with N time steps and store in S '''
        S = self.S0*np.exp(np.cumsum(
            (self.r-0.5*self.sigma**2)*self.dt + self.sigma*sqrt(self.dt) \
                *np.random.standard_normal((self.N+1, self.I)), axis=0))
            
        S[0] = self.S0
        
        return S
    
    def _calc_value(self):
        # Get set of stochastic prices of underlying
        S = self._simulate_underlying_process()
        
        # Initialize payoff matrix h and present value vector V
        h = self._inner_value(S)  # payoff matrix (i.e. option vs. underlying)
        V = h[-1]
        
        # Option valuation by backward induction
        for t in range(self.N - 1, 0, -1):
            rg = np.polyfit(S[t], V * self.df, self.nbf)
            C = np.polyval(rg, S[t])  # continuation values
            V = np.where(h[t] > C, h[t], V * self.df) # option's value
            
        return self.df*np.sum(V)/self.I # LSM estimator
    
class AmericanOptionLSMDual():
    pass
    

# =============================================================================
# Test only
# =============================================================================
if __name__ == '__main__':
    # VERSION 1: CRR model
    american_crr = AmericanOptionCRR(N=500, S0=90, K=100, t=0., M=1., r=0.06,
                                     sigma=0.2, otype='put')
    
    value_crr = american_crr()
    print(f"The value of the {american_crr.otype}-option in the CRR model is: {value_crr:.2f}")
    
    # VERSOIN 2: LSM Primal
    american_lsm_primal = \
        AmericanOptionLSMPrimal(I=10000, N=500, S0=90., K=100., t=0., M=1.,
                                r=0.06, sigma=0.2, otype='put')
    value_lsm_primal = american_lsm_primal()
    print(f"The value of the {american_lsm_primal.otype}-option in the LSM Primal model is: {value_lsm_primal:.2f}")
    
    
    
    
    
    
    