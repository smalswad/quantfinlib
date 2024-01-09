# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 17:25:27 2024

@author: Alexander Swade
"""

import sys
import pandas as pd
import numpy as np

from typeguard import typechecked
from math import sqrt

# Insert path to library if run as main program
if __name__ == '__main__':
    sys.path.insert(1, 'C:/Users/Alexander Swade/OneDrive - Lancaster University/'
                    'PhD research/quantfinlib')
    
from quantfinlib.simulation.stock_simulation import StockSim
from quantfinlib.options.european_options import EuropeanOptionCRR
from quantfinlib.simulation.random_numbers import generate_standardnormal_numbers

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
                                         1, 0)
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
    itm_only : boolean
        Use ITM options only to estimate polynomial coefficients. 
        
    '''
    def __init__(self, I, nbasis_function=6, itm_only=False, **kwargs):
        super().__init__(**kwargs)
        self.I = I
        self.nbf = nbasis_function
        self.itm = itm_only
    
    # def _calc_value(self):
    #     # Get set of stochastic prices of underlying
    #     S = self._simulate_underlying_process()
        
    #     # Initialize payoff matrix h and present value vector V
    #     h = self._inner_value(S)  # payoff matrix (i.e. option vs. underlying)
    #     V = h[-1]
        
    #     # Option valuation by backward induction
    #     for t in range(self.N - 1, 0, -1):
    #         rg = np.polyfit(S[t], V * self.df, self.nbf)
    #         C = np.polyval(rg, S[t])  # continuation values
    #         V = np.where(h[t] > C, h[t], V * self.df) # option's value
            
    #     return self.df*np.sum(V)/self.I # LSM estimator
    
    def _backward_iteration(self):
        '''
        Run backward iteration (Primal algorithm) to calculate option value at
        time t0.

        Returns
        -------
        S : np.ndarray, shape(self.N+1, self.I)
            Simulated underlying paths.
        V : np.ndarray, shape(self.N+1, self.I)
            Value matrix of option.
        rg : np.ndarray, shape(self.N+1, self.nbf+1)
            Polynomial coefficients for each point in time.

        '''
        # Get set of stochastic prices of underlying
        S = self._simulate_underlying_process()
        
        # Initialize payoff matrix h and value matrix V
        h = self._inner_value(S) # payoff matrix (i.e. option vs. underlying)
        V = self._inner_value(S) # value matrix (preliminarily)
        
        # Regression parameter matrix
        rg = np.zeros((self.N + 1, self.nbf + 1), dtype=float) 
        
        # Option valuation by backward induction
        for t in range(self.N - 1, 0, -1):
            
            # Use only ITM options to evaluate coefficients of polynomial
            if self.itm:
                S_itm, V_itm = self._in_the_money_only(t, h, S, V)
                rg[t] = np.polyfit(
                    S_itm, V_itm*self.df, self.nbf) if len(V_itm) != 0 else 0.0
            
            else:
                rg[t] = np.polyfit(S[t], V[t+1] * self.df, self.nbf)
            
            C = np.polyval(rg[t], S[t])  # continuation values
            V[t] = np.where(h[t]>C, h[t], V[t+1]*self.df) # option's value
            
        return S, V, rg
            
    def _calc_value(self):
        '''
        Calculate option value at t0.

        Returns
        -------
        float
            Option value at t0.

        '''
        # Estimate option value via backward iteration
        _, V, _ = self._backward_iteration()
        
        return self.df*np.sum(V[1])/self.I # LSM estimator
    
    def _in_the_money_only(self, idx, h, S, V):
        '''
        Return only k simulated values where the option is in the money at time
        idx.

        Parameters
        ----------
        idx : int
            Time (row) index.
        h : np.ndarray, shape(self.N+1, self.I)
            Payoff matrix.
        S : np.ndarray, shape(self.N+1, self.I)
            Simulated underlying paths.
        V : np.ndarray, shape(self.N+1, self.I)
            Value matrix of option (all moneyness).

        Returns
        -------
        S_itm : np.array, shape(k, )
            Simulated values corresponding to ITM option at time idx.
        V_itm : np.array, shape(k, )
            Values of option corresponding to ITM options and simulated values 
            at time idx.

        '''
        itm = np.greater(h, 0)
        S_itm = np.compress(itm[idx]==1, S)
        V_itm = np.compress(itm[idx]==1, V[idx+1]) 
        
        return S_itm, V_itm
    
    # def _simulate_underlying_process(self):
    #     ''' Simulate I Paths with N time steps and store in S '''
    #     S = self.S0*np.exp(np.cumsum(
    #         (self.r-0.5*self.sigma**2)*self.dt + self.sigma*sqrt(self.dt) \
    #             *np.random.standard_normal((self.N+1, self.I)), axis=0))
            
    #     S[0] = self.S0
        
    #     return S
    
    def _simulate_underlying_process(self):
        '''
        Simulate I Paths with N time steps following a standard GBM.

        Returns
        -------
        simulated_path : np.ndarray(), shape(self.N+1, self.I)
            Simulated paths following a standard GBM. First row contains 
            starting value S0 for each path.

        '''
        simulated_path = StockSim(S0=self.S0, dt=self.dt, T=int(self.T), mu=0,
                                  sigma=self.sigma, paths=self.I)
        
        return simulated_path()
    
class AmericanOptionLSMDual(AmericanOptionLSMPrimal):
    
    def __init__(self, I, **kwargs):
        super().__init__(**kwargs)

    def _calc_value(self):
        
        ''' 
        Simulate underlying paths and polynomial coefficient via backward
        iteration (Primal algorithm) 
        '''
        S, V, rg = self._backward_iteration()
        primal_est = self.df*np.sum(V[1])/self.I # LSM estimator
        
        '''
        Calculate dual part via foreward interation (Dual algorithm)
        '''
        
        # Initialize payoff matrix h, martingale matrix Q, and upper bound U
        h = self._inner_value(S)  # payoff matrix (i.e. option vs. underlying)
        Q = np.zeros((self.N + 1, self.I), dtype=float)  # martingale matrix
        U = np.zeros((self.N + 1, self.I), dtype=float)  # upper bound matrix
        
        for t in range(1, self.N + 1):
            
            V = np.maximum(h[t, :], np.polyval(rg[t], S[t, :])) # estimated values V(t)
            
            
            for i in range(self.I):
                Vt = max(h[t, i], np.polyval(rg[t], S[t, i])) # estimated values V(t,i)
                St = nested_monte_carlo(S[t - 1, i], J)  # nested MCS
                Ct = np.polyval(rg[t], St)  # cv from nested MCS
                ht = inner_value(St,K)  # iv from nested MCS
                # average of V(t,i,j)
                VtJ = np.sum(np.where(ht > Ct, ht, Ct)) / len(St)
                Q[t, i] = Q[t - 1, i] / df + (Vt - VtJ)  # "optimal" martingale
                # high estimator values
                U[t, i] = max(U[t - 1, i] / df, h[t, i] - Q[t, i])
                if t == M:
                    U[t, i] = np.maximum(U[t - 1, i] / df,
                                         np.mean(ht) - Q[t, i])
        dual_est = np.sum(U[M]) / I2 * df ** M  # DUAL estimator
    
        return primal_est, dual_est
    
    def _nested_monte_carlo(self, St: float, k: int, l: int):        
        '''
        Calculate k simulated 1-step GBM steps beginning at St 
        for l MC simulations/paths.

        Parameters
        ----------
        St : float
            Start value for S.
        k : int
            Number of random numbers to create per path
        l : in
            NUmber of paths

        Returns
        -------
        S_nmc : np.ndarray, shape(k,l)
            Data of k simulated 1-step GBM steps starting at St each 
            for l MC simulations/paths.

        '''
        random_numbers = np.vstack(
            [generate_standardnormal_numbers(n=k, ap=False, mm=False)] 
            for _ in range(l)).T
        
        S_nmc = St * np.exp((self.r - self.sigma ** 2 / 2) * self.dt +
                            self.sigma * random_numbers * sqrt(self.dt))
        
        return S_nmc
    
# =============================================================================
# Test only
# =============================================================================
if __name__ == '__main__':
    # # VERSION 1: CRR model
    # american_crr = AmericanOptionCRR(N=500, S0=90, K=100, t=0., M=1., r=0.06,
    #                                  sigma=0.2, otype='put')
    
    # value_crr = american_crr()
    # print(f"The value of the {american_crr.otype}-option in the CRR model is: {value_crr:.2f}")
    
    # # VERSION 2: LSM Primal
    # american_lsm_primal = \
    #     AmericanOptionLSMPrimal(I=10000, N=500, S0=90., K=100., t=0., M=1.,
    #                             r=0.06, sigma=0.2, otype='put')
    # value_lsm_primal = american_lsm_primal()
    # print(f"The value of the {american_lsm_primal.otype}-option in the LSM Primal model is: {value_lsm_primal:.2f}")
    
    
    # VERSION 3: LSM Dual
    american_lsm_dual = \
        AmericanOptionLSMDual(I=1000, N=500, S0=90., K=100., t=0., M=1.,
                                r=0.06, sigma=0.2, otype='put')
        
    # value_lsm_dual = american_lsm_dual()
    
    
    