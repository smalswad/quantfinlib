# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 15:18:10 2023

@author: ROB6738
"""

import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as spc

def plot_correlation(data, title=None, fmt='.2f', cbar=True, filepath=None, 
                     filename='correlation_heatmap.png', labels='auto', 
                     percentage=False):
    '''Function to plot correlation heatmap. 
    Parameters
    ----------
    data : pd.DataFrame
        Column-wise data to use for correlation plot.
    title : str, optional
        Title of the plot. The default is None.
    fmt : str, optional
        Float format. The default is '.2f'.
    cbar : boolean, optional
        indicate whether color bar should be plotted. The default is True.
    filepath : str, optional
        Filepath to save the plot. The default is None.
    labels : str, optional
        Labels for the columns/rows of the correlation matrix. The default is 'auto'.
    percentage : boolean, optional
        Scale correlations by 100. If true, fmt will be set to .0f. The default
        is False.

    Returns
   -------
    None.
    '''
    if percentage:
        fmt = '.0f'
        scale = 100
    else:
        scale = 1
    
    #Calc correlations
    df_corr = data.corr()
    scaled_corr = df_corr*scale
    
    #mask for colored areas
    mask = np.zeros_like(df_corr)
    mask[np.tril_indices_from(mask, k=0)] = True

    #get lower triangular values only
    annot = scaled_corr.where(np.tril(np.ones(df_corr.shape)).astype(np.bool))
    annot_mask = np.zeros_like(df_corr)
    annot_mask[np.triu_indices_from(annot_mask, k=1)] = True

    # Plot correlation 
    fig, axs = plt.subplots(1,1, figsize=(12, 8))    
    axs = sns.heatmap(df_corr, mask=mask, cbar=cbar, vmin=-1, vmax=1,
                      cmap="bwr_r", linewidths=.5, linecolor='black', square=True,
                      xticklabels=labels, yticklabels=labels)
    axs = sns.heatmap(df_corr, mask=annot_mask, annot=annot, fmt=fmt, cbar=False, 
                      vmin=-1, vmax=1, cmap=['white'], square=True, linewidths=.5,
                      linecolor='black', xticklabels=labels, yticklabels=labels)
    
    # Make all spines visible
    for _, spine in axs.spines.items():
        spine.set_visible(True)
    
    # Adjust axis and title
    axs.tick_params(axis=u'both', which=u'both',length=0)
    axs.xaxis.tick_top()
    axs.set_xticklabels(axs.get_xticklabels(), rotation=90)    
    axs.set_title(title)
    
    # Save figure or just plot it
    if filepath is not None:
        fig.savefig(os.path.join(filepath, filename), bbox_inches='tight')  
    else:
        fig.show()
        
def plot_dendrogram(data, filepath, filename='correlation_dendrogram.png'):
    '''
    Plot dendrogram for correlation distances of data based on wards method
    and save to filepath.

    Parameters
    ----------
    data : pd.DataFrame, shape(T,N)
        Data to use for correlation calculations.
    filepath : str
        Directory to save the plot.

    Returns
    -------
    None.

    '''
    #Calculate distance matrix and create linkage matrix for dendrogram
    dist_mat = np.sqrt(0.5*(1-data.corr().values))
    linkage = spc.linkage(spc.distance.squareform(dist_mat), 'ward')
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    #Adjust color_threshold to redefine number of colored clusters
    spc.dendrogram(linkage, color_threshold=.9, labels=data.columns, ax=ax)
    
    #Format axis
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='center',
                        fontsize=7)
    ax.set_ylabel("Distance / (Dis)similarity")
    ax.spines[:].set_visible(False)
    
    # Save figure or just plot it
    if filepath is not None:
        fig.savefig(os.path.join(filepath, filename), bbox_inches='tight')  
    else:
        fig.show()
        
        
        
        
        
        