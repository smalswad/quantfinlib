# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 12:42:00 2024

@author: Alexander Swade
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mpl_toolkits.mplot3d.axes3d as p3

sys.path.insert(1, 'C:/Users/Alexander Swade/OneDrive - Lancaster University/'
                'PhD research/quantfinlib')
from quantfinlib.options.european_options import EuropeanOption

def plot_3d_mesh(x_data, y_data, z_data, filepath=None, title=None, 
                 xyz_labels=None, filename='3d_mesh.png'):
    
    # Data prep
    x, y = np.meshgrid(x_data, y_data)
    x_lab, y_lab, z_lab = xyz_labels
    # Plotting section
    fig, ax = plt.subplots(1,1, subplot_kw={"projection": "3d"}, figsize=(12,8))
    ax.plot_wireframe(x, y, z_data)
    
    #Configurate title and yaxis
    ax.set_title(title)
    ax.set_xlabel(x_lab)
    ax.set_ylabel(y_lab)
    ax.set_zlabel(z_lab)
    
    # Save figure or just plot it
    if filepath is not None:
        fig.savefig(os.path.join(filepath, filename), bbox_inches='tight')  
    else:
        plt.show()
# =============================================================================
# Test only
# =============================================================================
if __name__ == '__main__':
    
    '''
    Generate mesh data (here vega of European call for different strikes and 
    maturities) 
    '''
    dates = pd.date_range(start='30-09-2014', end='30-09-2015', periods=13)
    strikes = np.linspace(80,120,25)
    V = np.zeros((len(dates)-1, len(strikes)), dtype=float)
    for j in range(len(strikes)):
        for i in range(len(dates)-1):
            call_opt = EuropeanOption(
                S0=60, K=strikes[j], t=dates[0],M=dates[i+1], r=0.08,
                sigma=0.3, otype='call')
            call_opt()
            _, _, vega, _, _ = call_opt.get_greeks()
            V[i,j] = vega
    
    # Plot data
    xyz_lab = ('strike $K$', 'maturity $M$', 'vega (K, M)')
    plot_3d_mesh(strikes, range(1,13), V, xyz_labels=xyz_lab)