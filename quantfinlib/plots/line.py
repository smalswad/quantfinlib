# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 15:35:37 2023

@author: ROB6738
"""
import os
import matplotlib.pyplot as plt

from matplotlib import dates

def plot_line(data, filepath=None, title=None, ylabel=None, scale='linear',
              filename='line.png', year_interval=1, fill_area=None,
              colors=None):
    '''
    Create line plot for given data.

    Parameters
    ----------
    data : pd.DataFrame
        Data to plot.
    filepath : str, optional
        Directory to save final plot. The default is None.    
    title : string, optional
        Plot title. The default is None.
    ylabel : string, optional
        Y axis label. The default is None.
    scale : string, optional
        Indicate scale of y-axis. The default is 'linear'.
    filename : str, optional
        Filename of final plot. The default is 'line.png'.
    year_interval : int, optional
        Ticker interval for x-axis labels. The default is 1. 
    fill_area : pd.Series of boolean, optional
        Series indicating highlighted background (to be shaded grey).
        The default is None.
    colors : list, optional
        Colors used for plotting. The default is None. 

    Returns
    -------
    None.

    '''
    #Plotting section
    fig, ax = plt.subplots(1,1, figsize=(12,8))
    
    ymin, ymax = data.min().min()*1.05, data.max().max()*1.05
    
    if fill_area is not None:
        fill_area = fill_area.loc[data.index]
        ax.fill_between(fill_area.index, y1=ymin, y2=ymax, where=fill_area,
                        alpha=0.4, facecolor='grey')
    
    ax.plot(data.index, data, label=data.columns)
    ax.set_axisbelow(True)
    ax.grid(True, which='major', linewidth=1)
    
    # Set colors if specified
    if colors is not None:
        for i,j in enumerate(ax.lines):
            j.set_color(colors[i])
        
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
    ax.set_yscale(scale)
    ax.set_ylim([ymin, ymax])
    
    #Configurate legends
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, loc='upper center')
    
    # Save figure or just plot it
    if filepath is not None:
        fig.savefig(os.path.join(filepath, filename), bbox_inches='tight')  
    else:
        plt.show()