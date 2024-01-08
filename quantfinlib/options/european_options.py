# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 14:18:00 2024

@author: Alexander Swade
"""

import math
import numpy as np
import pandas as pd

from math import sqrt, exp
from scipy import stats
from scipy.integrate import quad
from scipy.optimize import fmin
from typeguard import typechecked


def dN(x):
    ''' PDF of standard normal random variable x.'''
    return exp(-0.5 * x ** 2) / sqrt(2 * math.pi)

def N(d):
    ''' CDF of standard normal random variable x. '''
    return quad(lambda x: dN(x), -20, d, limit=50)[0]

@typechecked
class OptionBase(object):
    ''' Base class for European options. Relies on children for valuation.
    
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
        
    '''
    def __init__(
            self,
            S0: float,
            K: float,
            t,
            M,
            r: float,
            sigma: float,
            otype='call'
    ):
        self.S0 = float(S0)
        self.K = K
        self.t = t
        self.M = M
        self.r = r
        self.sigma = sigma
        self.otype = otype
        
    def _update_ttm(self):
        ''' Updates time-to-maturity self.T. '''
        if isinstance(self.t, pd.Timestamp) and isinstance(self.M, pd.Timestamp):
            if self.t > self.M:
                raise ValueError("Pricing date later than maturity.")
            self.T = (self.M - self.t).days / 365.
        
        elif isinstance(self.t, float) and isinstance(self.M, float):
            self.T = self.M - self.t
        
        else:
            raise ValueError("Values passed to 't' and 'M' need to be of "
                             "same data type (float or pd.Timestamp")

@typechecked
class EuropeanOptionBSM(OptionBase):
    ''' Class for European options in Black Scholes Merton Model.
            
    Methods
    =======
    value : float
        return present value of call option
    get_greeks : tuple of 5 floats
        return option greeks
    implied_vol : float
        return implied volatility given option quote
    '''
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
                      
    def _d1(self):
        ''' Helper function to calculate d1 of BSM model. '''
        nom = np.log(self.S0/self.K)+(self.r + 0.5*self.sigma**2)*self.T
        denom = self.sigma*sqrt(self.T)
        
        self.d1 = nom/denom
        
    def _d2(self):
        ''' Helper function to calculate d2 of BSM model. '''
        self.d2 = self.d1 - self.sigma*sqrt(self.T)

    def _calc_value(self):
        ''' Helper function to calculate option value'''
        
        call_value = (self.S0*stats.norm.cdf(self.d1, 0.0, 1.0) \
                 - self.K*exp(-self.r*self.T)*stats.norm.cdf(self.d2, 0.0, 1.0))
            
        if self.otype == 'call':
            return call_value
        else:              
            # Calculate put value via put-call parity
            return call_value - self.S0 + self.K*exp(-self.r*self.T)
    
    def _greeks_delta(self):
        ''' Calculate option delta '''
        if self.otype == 'call':
            self.delta = N(self.d1)
        else:
            self.delta = N(self.d1) - 1
    
    def _greeks_gamma(self):
        ''' Calculate option gamma '''
        self.gamma = dN(self.d1)/self.S0*self.sigma*sqrt(self.T)
    
    def _greeks_vega(self):
        ''' Calculate option vega '''
        self.vega = self.S0*dN(self.d1)*sqrt(self.T)
        
    def _greeks_theta(self):
        ''' Calculate option theta '''
        fraction = -(self.S0*dN(self.d1)*self.sigma)/(2*sqrt(self.T))
    
        if self.otype == 'call':
            self.theta = fraction - self.r*self.K*exp(-self.r*self.T)*N(self.d2)            
        else:
            self.theta = fraction + self.r*self.K*exp(-self.r*self.T)*N(-self.d2)
    
    def _greeks_rho(self):
        ''' Calculate option rho'''
        if self.otype == 'call':
            self.rho = self.K*self.T*exp(-self.r*self.T)*N(self.d2)        
        else:
            self.rho = -self.K*self.T*exp(-self.r*self.T)*N(-self.d2) 
    
    def implied_vol(self, price: float, sigma_est=0.2, verbose=False):
        '''
        Calculate implied volatility for given market price price P0 of the 
        option.

        Parameters
        ----------
        price : float
            Market price of option.
        sigma_est : float, optional
            Initial volatility guess. The default is 0.2.
        verbose : boolean
            Print convergence message of optimizer.
        
        Returns
        -------
        float
            Implied volatility based on given market price of option.

        '''
        
        # Define nested function used for root finding
        def calc_diff(sigma):
            # Create option and evaluate price based on estimated volatility
            option = EuropeanOptionBSM(
                self.S0, self.K, self.t, self.M, self.r, sigma[0], self.otype)
            option()

            return (option.value - price)**2
        
        iv = fmin(calc_diff, [sigma_est], disp=verbose)[0]
        
        return iv
    
    def get_greeks(self):
        '''
        Calculate option greeks.

        Returns
        -------
        float
            Option delta (dV dS).
        float
            Option gamma (d2V d2S).
        float
            Option vega (dV dSigma).
        float
            Option theta (dV dt).
        float
            Option rho (dV dr).

        '''
        self._greeks_delta()
        self._greeks_gamma()
        self._greeks_vega()
        self._greeks_theta()
        self._greeks_rho()        
        
        return self.delta, self.gamma, self.vega, self.theta, self.rho
    
    def __call__(self):
        '''
        Override call funtion to create function object, i.e., call other
        functions as specified.
        
        Returns
        -------
        value : float
            Fair option value according to BSM model.

        '''
        
        # Calculate all helper values
        self._update_ttm()
        self._d1()
        self._d2()
        self.value = self._calc_value()
        
        return self.value

@typechecked    
class EuropeanOptionCRR(OptionBase):
    ''' Class for European options based on Cox-Ross-Rubinstein.
    
    Attributes
    ==========
    N : int
        Number of time intervals
        
    '''
    
    def __init__(self, N:int, **kwargs):
        super().__init__(**kwargs)
        self.N = N
        self.dt = (self.M - self.t)/self.N # length of each time interval
        self.df = exp(-self.r * self.dt) # discount per interval
        
    def _inner_value(self, S):
        ''' Calculate inner values of option given the price of the underlying S
        
        Parameters
        ----------
        S : np.ndarray
            Price of the underlying.

        Returns
        -------
        V : np.ndarray
            Inner values of option.
            
        '''
        if self.otype == 'call':
            V = np.maximum(S - self.K, 0) 
        elif self.otype == 'put':
            V = np.maximum(self.K - S, 0)
        else: 
            print('Please provide valid option type: call or put')
            
        return V
    
    def _calc_value(self):
        # Calculate up, down, and probability q according to the CRR model
        u = exp(self.sigma*sqrt(self.dt))
        d = 1/u
        q = (exp(self.r*self.dt) - d)/(u - d)
        
        # Initialize arrays to calculate binomial model
        mu = np.arange(self.N + 1)
        mu = np.resize(mu, (self.N + 1, self.N + 1))
        md = np.transpose(mu)
        mu = u**(mu - md)
        md = d**md
        S = self.S0*mu*md
        
        # Calculate initial option values
        V = self._inner_value(S)
        
        # Fill inner values iteratively backwards
        z = 0
        for t in range(self.N-1, -1, -1):
            V[0:self.N - z, t] = \
                (q*V[0:self.N-z, t+1] + (1 - q)*V[1:self.N-z+1, t+1])*self.df
            z += 1
        
        self.inner_values = V
        
        return V[0,0]
    
    def __call__(self):
        '''
        Override call funtion to create function object, i.e., call other
        functions as specified.
        
        Returns
        -------
        value : float
            Fair option value according to CRR model.

        '''
        value = self._calc_value()
        
        return value
        
    

        
# =============================================================================
# Test only
# =============================================================================
if __name__ == '__main__':
    ''' TEST 1: BSM model valuation '''
    euro_opt = EuropeanOptionBSM(
        S0=60, K=65, t=pd.Timestamp('30-09-2014'), M=pd.Timestamp('30-12-2014'),
        r=0.08, sigma=0.3, otype='put')
    value_bsm = euro_opt()
    delta, gamma, vega, theta, rho = euro_opt.get_greeks()
    p0 = 8
    impl_vol = euro_opt.implied_vol(p0, sigma_est=0.25)
    
    print(f"The value of the {euro_opt.otype}-option is: {value_bsm:.2f}")
    print(f"The option's delta is: {delta:.2f}")
    print(f"The option's gamma is: {gamma:.2f}")
    print(f"The option's vega is: {vega:.2f}")
    print(f"The option's theta is: {theta:.2f}")
    print(f"The option's rho is: {rho:.2f}")
    print(f"The option's implied volatility at P0 = {p0} is {impl_vol:.4f}")
    
    
    ''' TEST 2: CRR model valuation '''
    crr_opt = EuropeanOptionCRR(N=5, S0=60, K=65, t=0, M=0.25, r=0.08, sigma=0.3,
                                otype='put')
    value_crr = crr_opt()
    print(f"The value of the {crr_opt.otype}-option is: {value_crr:.2f}")
    
    
    
    
    