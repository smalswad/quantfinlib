# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 10:16:02 2023

@author: Alexander Swade
"""
import os
import math
import pathlib
import numpy as np
import pandas as pd

from typeguard import typechecked

@typechecked
class StockSim():
    '''
    StockSim is an object generating stochastic stock price movement paths 
    using a standard geometric Brownian motion (GBM).
    
    Parameters
    ----------
    S0:         int
                Stock price at starting point
    dt:         int
                Size of time steps (per year)
    T:          int
                Length of time horizon (in years)
    mu:         float
                Mean return of stock prices
    sigma:      float
                Standard deviation of stock price returns
    paths:      int
                Number of paths to simulate
    output_dir: str, optional
                Directory for the output to be saved. The default is None.
    filename:   str, optional
                Filename of the output csv file. The default is None.
            
    '''
    
    def __init__(
        self,
        S0: float,
        dt: float,
        T: int,
        mu: float,
        sigma: float,
        paths: int,
        output_dir=None, 
        filename=None
    ):
        self.S0 = S0
        self.dt = dt
        self.T = T
        self.mu = mu
        self.sigma = sigma
        self.paths = paths
        self.N = int(self.T/self.dt)
        self.output_dir = output_dir
        self.filename = filename
    
    def set_od(self, directory: str):
        '''
        Set output directory.

        Parameters
        ----------
        directory : str
            New output directory.

        '''
        self.output_dir = directory
        
    def _generate_paths(self):
        '''
        Generate stock prices for different paths.

        Returns
        -------
        np.array, shape(N, paths)
        Simulated stock prices for different paths.

        '''
        # Vectorized implementation of asset path generation
        asset_paths = np.exp(
            (self.mu - self.sigma**2 / 2) * self.dt +
            self.sigma * np.random.normal(0, math.sqrt(self.dt),
                                          size=(self.N, self.paths))
        )
        
        # Add array of 1's
        asset_paths = np.vstack([np.ones(self.paths), asset_paths])
        
        return self.S0 * asset_paths.cumprod(axis=0)
    
    def _save_as_csv(self, data):
        '''
        Write data as csv file. Use defined output directory or create folder 
        in current working directory.

        Parameters
        ----------
        data : np.array
            Data to save as csv file.

        Returns
        -------
        None.

        '''
        if self.output_dir is None:
            cwd = os.getcwd()
            od = os.path.join(cwd, 'outputs')
            pathlib.Path(od).mkdir(parents=True, exist_ok=True)
            
        else:
            od = self.output_dir
        
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(od, self.filename+'.csv')) 
    
    def __call__(self, pct=False):
        '''
        Override call funtion to create function object, i.e., call other
        functions as specified.

        Returns
        -------
        data : np.array, shape(N, paths)
            Generated simulated stock prices.

        '''
        # Generate data
        data = self._generate_paths()
        
        # Calculate returns if specified
        if pct:
            data = data[1:]/data[:-1] - 1
        
        # Either write results as csv or return directly
        if self.filename is not None:
            self._save_as_csv(data)
        else:
            return data

# =============================================================================
# Test only
# =============================================================================
if __name__ == '__main__':
    sim = StockSim(S0=200, dt=1/252, T=10, mu=0.01, sigma=0.2, paths=5)
    data = sim(pct=True)