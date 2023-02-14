# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 15:23:15 2023

@author: smalswad
"""

import os
import matplotlib.pyplot as plt

from matplotlib import dates

def plot_scatter(data, filepath, title=None, ylabel=None, 
                 filename='scatter.png', year_interval=1):
    '''
    Create scatter plot of given data.

    Parameters
    ----------
    data : pd.Series
        Data to plot.
    filepath : str, optional
        Directory to save final plot. The default is None.    
    title : string, optional
        Plot title. The default is None.
    ylabel : string, optional
        Y axis label. The default is None.
    filename : str, optional
        Filename of final plot. The default is 'scatter.png'.
    year_interval : int, optional
        Ticker interval for x-axis labels. The default is 1. 

    Returns
    -------
    None.

    '''
    
    #Plotting section
    fig, ax = plt.subplots(1,1, figsize=(12,8))
    
    ax.scatter(data.index, data)
    ax.set_axisbelow(True)
    ax.grid(True, which='major', linewidth=1)
    
    #Configurate xaxis
    ax.set_xlim([data.index.min(), data.index.max()])
    years = dates.YearLocator(year_interval)
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(dates.DateFormatter('%Y'))
    ax.get_xaxis().set_tick_params(which='major', pad=10)
    plt.setp(ax.get_xmajorticklabels(), rotation=0, weight='bold', ha='center')
    
    #Configurate title and yaxis
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    
    # Save figure or just plot it
    if filepath is not None:
        fig.savefig(os.path.join(filepath, filename), bbox_inches='tight')  
    else:
        plt.show()