# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 12:30:21 2022

@author: ROB6738
"""

import numpy as np
from scipy.optimize import minimize

TOLERANCE = 1e-10

def calc_rpo(w0, cov_mat, mrc):
    '''
    Calculate risk parity optimization (or alternative marginal risk contr)

    Parameters
    ----------
    w0 : np.array, shape (n,)
        Initial guess. Array of real elements of size (n,),
        where 'n' is the number of independent variables.
    cov_mat : np.ndarray, shape (n,n)
        Covariance matrix/ dispersion matrix.
    mrc : list
        Target marginal risk contributions, i.e. risk budget of total portfolio
        risk (e.g. equal risk)

    Returns
    -------
    w_rb : np.array, shape (n,)
        Risk budget weights.

    '''
    
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},   #full investment
            {'type': 'ineq', 'fun': lambda x: x})               #long only
    res = minimize(fun=_risk_budget_objective_error, x0=w0,
                   args=[cov_mat, mrc], method='SLSQP', constraints=cons,
                    tol=TOLERANCE, options={'disp': False})
    
    return res.x 



def _allocation_risk(weights, covariances):
    # Calc portf vola
    portfolio_risk = np.sqrt((weights * covariances * weights.T))[0, 0]

    return portfolio_risk


def _assets_risk_contribution_to_allocation_risk(weights, covariances):

    portfolio_risk = _allocation_risk(weights, covariances)

    # Calculate risk contributions
    assets_risk_contribution = np.multiply(weights.T, covariances * weights.T) \
        / portfolio_risk

    return assets_risk_contribution


def _risk_budget_objective_error(weights, args):

    covariances = args[0]
    assets_risk_budget = args[1]
    weights = np.matrix(weights)

    portfolio_risk = _allocation_risk(weights, covariances)

    # Calc risk contributions
    assets_risk_contribution = \
        _assets_risk_contribution_to_allocation_risk(weights, covariances)

    # Calc risk target for each asset (based on risk budget)
    assets_risk_target = \
        np.asmatrix(np.multiply(portfolio_risk, assets_risk_budget))

    # Calc difference between target and current risk contribution
    error = \
        sum(np.square(assets_risk_contribution - assets_risk_target.T))[0, 0]

    return error


