# -*- coding: utf-8 -*-
"""
Created on Wed May  4 10:34:57 2022

@author: ROB6738
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize

TOLERANCE = 1e-10

class BaseOptimizer(object):
    '''
    Basic optimizer class used as parent class for following optimizer. 
    Optimization problems are of the structure
    
        min x   f(x)
        s.t.    Ax <= b
        
    f is defined in children classes. 
    '''
    def __init__(self, w0, asset_names=None, eq_A=None, eq_b=None, ineq_A=None,
                 ineq_b=None, long_only=False, full_invest=False):
        '''
        Parameters
        ----------
        w0 : np.array, shape(N,)
            Initial starting value for optimization, i.e. starting weights.
        asset_names : iterable, optional
            List of asset names in same order as w0. The default is None.
        eq_A : np.array, shape(k,N), optional
            Coefficients matrix for k equality constraints. The default is None.
        eq_b : np.array, shape(k,) optional
            Equality constraints values. The default is None.
        ineq_A : np.array, shape(l,N), optional
            Coefficients matrix for l inequality constraints.
            The default is None.
        ineq_b : np.array, shape(l,) optional
            Inequality constraints values. The default is None.
        long_only : boolean, optional
            Indicator to include long-only constraint. The default is False.
        full_invest : boolean, optional
            Indicator to include full-invest constraint. The default is False.

        Raises
        ------
        ValueError
            Raises error if input doesn't match expected data types/ options.

        Returns
        -------
        None.

        '''
        
        # Initialize starting weights for optimization
        if isinstance(w0, np.ndarray):
            self.w0 = w0
        else:
            raise ValueError("Value passed to 'w0' did not match expected "
                             "data type. Expected np.array but got "
                             f"{type(w0)} instead.")
        
        #Define asset names
        if asset_names is None:
            self.asset_names = list(range(w0.shape[0]))
        else:
            self.asset_names = asset_names
        
        #Optimized output weights
        self.w_opt = None
        
        #Transform constraints to list format for optimizer
        self._create_constraints_as_list(eq_A=eq_A, eq_b=eq_b, ineq_A=ineq_A,
                                         ineq_b=ineq_b, long_only=long_only,
                                         full_invest=full_invest)
        
    def _create_constraints_as_list(self, eq_A=None, eq_b=None, ineq_A=None,
                                    ineq_b=None, long_only=False,
                                    full_invest=False):
        '''
        Helper function to construct constraints list for minimize function.
        Constraints are given as Ax=b and Ax=>b, resp.

        Parameters
        ----------
        eq_A : np.ndarray, optional
            Coefficient matrix for equality constraints. The default is None.
        eq_b : np.ndarray, optional
            RHS vector for equality constraints. The default is None.
        ineq_A : np.ndarray, optional
            Coefficient matrix for inequality constraints. The default is None.
        ineq_b : np.ndarray, optional
            RHS vector for inequality constraints. The default is None.
        long_only : bool, optional
            Indication whether long-only constraints shall be included.
            The default is False.
        full_invest : bool, optional
            Indication whether full-investment constraint shall be included.
            The default is False.

        Returns
        -------
        None.

        '''
        self._cons = list()        
        if long_only:
            self._cons.append(self._long_lony_constraint())        
        if full_invest:
            self._cons.append(self._full_investment_constraint())        
        if eq_A is not None:
            self._cons.append(self._get_equality_constraints(eq_A, eq_b))
        if ineq_A is not None:
            self._cons.append(self._get_inequality_constraints(ineq_A, ineq_b))
            
    def _convert_weights_to_series(self): 
        '''
        Helper function to convert optimized weights as named series instead of
        raw numpy data array. 
    
        Returns
        -------
        pd.Series
            Optimized weigthts as series including asset names.
        '''
        return pd.Series(self.w_opt, index=self.asset_names)
    
    @staticmethod
    def _full_investment_constraint():
        return {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}
    
    @staticmethod
    def _get_equality_constraints(A, b):
        eq_cons = {'type': 'eq',
                   'fun':lambda x: b - np.dot(A,x),
                   'jac':lambda x: -A}
        
        return eq_cons 
    
    @staticmethod 
    def _get_inequality_constraints(A, b):
        ineq_cons = {'type': 'ineq',
                     'fun':lambda x: b - np.dot(A,x),
                     'jac':lambda x: -A}
        
        return ineq_cons
    
    @staticmethod 
    def _long_lony_constraint():
        return {'type': 'ineq', 'fun': lambda x: x}
    
    @staticmethod
    def _round_result(res, decimals=12):        
        return np.around(res, decimals)
    


class QuadraticOptimizer(BaseOptimizer):
    '''
    Quadratic optimizer class. Standard optimization problem yields
        min x   0.5*(x.T @ D @ x) + d.T @ x + c
        s.t.    Ax <= b
    '''
    def __init__(self, func=None, jac=None, func_args=None, *args, **kwargs):
        '''
        Parameters
        ----------
        func : str or callable, optional
            Optimization target function. Choose from predefined options such
            as {'mean_variance', 'mean_variance_with_tc'} or pass individual
            callable function. If None is given, use standard function for 
            quadratic optimization problem. The default is None.
        func_args : list, optional
            Function arguments. The default is None.
        *args : TYPE
            see documentation of BaseOptimizer class.
        **kwargs : TYPE
            see documentation of BaseOptimizer class.

        Raises
        ------
        ValueError
            Raises error if input doesn't match expected data types/ options.

        Returns
        -------
        None.

        '''
        super().__init__(*args, **kwargs)
        
        # Define optimization function 
        if func is not None:
            if isinstance(func, str):
                # Use class functions as selcted
                if func == "mean_variance":
                    self._func = self.__mean_variance_function 
                    self._jac = self.__mean_variance_jac
                elif func == "mean_variance_with_tc":
                    self._func = self.__mean_variance_with_tc_function
                    self._jac = self.__loss_jac
                else:
                    raise ValueError('Argument passed to "func" does not '
                                     'match with any implemented function.')
            elif callable(func):
                # Use passed function 
                self._func = func 
                self._jac = jac
        else: 
            #Use standard loss function for optimizer
            self._func = self.__loss_function
            self._func = self.__loss_jac
        
        #Save function arguments as class variable
        self._func_args = func_args
        self.terminated_successful = None
        
        
    @staticmethod
    def __loss_function(x, args):
        D = args[0]
        d = args[1]
        c = args[2]
        sign = args[3]
        return sign * (0.5* x.T@D@x + d.T@x + c)
    
    @staticmethod 
    def __loss_jac(x, args):
        D = args[0]
        d = args[1]
        sign = args[3]
        return sign * (x.T@D + d)
    
    @staticmethod
    def __mean_variance_function(x, args):
        '''
        Target function for mean-variance optimization problem:
            min x   0.5* x.T@D@x
            s.t.    Ax <= b
        '''
        D = args[0]
        return 0.5* x.T@D@x 
    
    @staticmethod
    def __mean_variance_jac(x, args):
        D = args[0]
        return x.T@D 
    
    @staticmethod
    def __mean_variance_with_tc_function(x, args):
        '''
        Target function for mean-variance optimization problem:
            min x   0.5* x.T@D@x + d.T@x
            s.t.    Ax <= b
        '''
        D = args[0]
        d = args[1]
        return 0.5* x.T@D@x + d.T@x 
    
    def solve(self, round=True):
        res = minimize(fun=self._func, x0=self.w0, args=self._func_args,
                       method='SLSQP', constraints=self._cons, jac=self._jac, 
                       tol=TOLERANCE, options={'disp': False})
        
        self.w_opt = res.x
        if round:
            self.w_opt = self._round_result(self.w_opt)
            
        self.terminated_successful = res.success
    
    
    
class MaxSharpe(BaseOptimizer):
    '''
    Maximum Sharpe Ratio Optimization class. The optimization problem yields
        max x   (x.T @ mu - rf) / sqrt(x.T @ COV @ x) 
        s.t.    Ax <= b
    '''
    def __init__(self, returns, rf=0, *args, **kwargs):
        '''
        Parameters
        ----------
        returns : pd.DataFrame or np.adarray, shape(T,N)
            (Historic) asset returns.
        rf : float, optional
            Risk-free rate. The default is 0.
        *args : TYPE
            see documentation of BaseOptimizer class..
        **kwargs : TYPE
            see documentation of BaseOptimizer class..

        Raises
        ------
        ValueError
            Raises error if input doesn't match expected data types/ options.

        Returns
        -------
        None.

        '''
        
        super().__init__(*args, **kwargs)
        
        if isinstance(returns, pd.DataFrame):
            self.mu = returns.mean().values
            self.cov = returns.cov().values
            
        elif isinstance(returns, np.ndarray):
            self.mu = np.mean(returns)
            self.cov = np.cov(returns)
        
        else:
            raise ValueError('Argument passed to "returns" does not '
                             'match with expected data type.')
      
        # Use passed function 
        self._func = self.__min_sharpe_function 
        self._rf = rf
        self._func_args = (self.mu, self.cov, rf)
        self.terminated_successful = None        
        
    @staticmethod 
    def __min_sharpe_function(x, mu, cov, rf):
        return -(x.T @ mu - rf) / np.sqrt(x.T @ cov @ x)        
        
    def solve(self, round=True):
        res = minimize(fun=self._func, x0=self.w0, args=self._func_args,
                       method='SLSQP', constraints=self._cons,
                       tol=TOLERANCE, options={'disp': False})
        
        self.w_opt = res.x
        self.sharpe = - self.__min_sharpe_function(
            self.w_opt, self.mu, self.cov, self._rf)
        
        if round:
            self.w_opt = self._round_result(self.w_opt)
            
        self.terminated_successful = res.success 
        
        
        
        
        
        
        
        
        
        
        