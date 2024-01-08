# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 17:25:27 2024

@author: Alexander Swade
"""

import pandas as pd
import numpy as np

from typeguard import typechecked

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
        # Get stochastik prices of underlying
        S = self._calc_binomial_parameter()
        
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
    
    
def set_parameters():
    pass


# =============================================================================
# Test only
# =============================================================================
if __name__ == '__main__':
    american_put = AmericanOptionCRR(N=500, S0=90, K=100, t=0., M=1., r=0.06,
                                     sigma=0.2, otype='put')
    
    value_crr = american_put()
    print(f"The value of the {american_put.otype}-option is: {value_crr:.2f}")