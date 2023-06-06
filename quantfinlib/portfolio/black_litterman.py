# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 15:56:32 2023

@author: ROB6738
"""

import numpy as np

from scipy.linalg import issymmetric

from quantfinlib.linalg.posdef import nearestPD
from quantfinlib.linalg.posdef import isPD

def bl_masterformula(pi, Sigma, P, q, Omega, tau=0.015):
    '''
    Implementation of the Black-Litterman master formula to get the adjusted
    expected return as well as adjusted Covariance matrix.

    Parameters
    ----------
    pi : np.array, shape(n,)
        Implied excess equilibrium return vector.
    Sigma : np.array, shape(n,n)
        Covariance matrix for all assets.
    P : np.array, shape(k,n)
        Population matrix for k views on n assets.
    q : np.array, shape(k,)
        Views on specified combination of assets (relative or absolute).
    Omega : np.array, shape(k,k)
        Omega is a diagonal covariance matrix with 0’s in all of the off-diagonal
        elements describing the uncertainty about the individual views.
    tau : float, optional
        Scaling parameter. The default is 0.015.

    Returns
    -------
    bl_sigma : np.array, shape(n,n)
        Covariance matrix adjusted for specified views.
    bl_mu : np.array, shape(n,)
        Black-Litterman adjusted expected returns.

    '''  
    # Calculate Black-Litterman adjusted weights and covariance matrix
    Z = np.linalg.inv(tau * P @ Sigma @ P.T + Omega)     
    bl_mu = pi + tau * Sigma @ P.T @ Z @ (q.T - P @ pi)    
    bl_sigma = (1+tau)*Sigma - (tau**2*Sigma @ P.T @ Z @ P @ Sigma)
    
    # Adjust BL Sigma matrix if not positive definite
    if  not isPD(bl_sigma) or not issymmetric(bl_sigma):
        bl_sigma = nearestPD(bl_sigma)
    
    return (bl_sigma, bl_mu)
