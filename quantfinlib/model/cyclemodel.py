# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 20:04:56 2022

@author: ROB6738
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import dates
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from pandas.tseries.offsets import MonthEnd


def calc_zscore(data, how='mean'):
    '''
    Function to normalize given series. Also caps standardized series to three
    standard deviations on either side.

    Parameters
    ----------
    data : pd.DataFrame
        Data to be standardized.
    how : string, optional
        Method to chose for averaging. The default is 'mean'.

    Returns
    -------
    score : pd.DataFrame
        Z-scores for time series.

    '''
    if how == 'mean':
        avg = data.mean()
    else:
        avg = data.median()
        
    score = (data.iloc[-1] - avg)/data.std()
    
    # Cap values to 3 standard deviations on either side
    score = np.minimum(score, 3)
    score = np.maximum (score, -3)
    
    return score
   
def def_regime(zscores):
    '''
    Helper function to map zscore to actual regimes. Regimes are defined as:
        expansion:  positive and increasing Z-score,
        peak:       positive but decreasing Z-score,
        recession:  negative and decreasing Z-score, 
        recovery:   negative but increasing Z-score. 

    Parameters
    ----------
    zscores : pd.Series
        Standardized data series.

    Returns
    -------
    regime_final : pd.Series
        Regime classifications based on 2 dimensional split as described above.

    '''
    # Define regimes based on scores and yearly changes
    yoy_changes = zscores.diff(12).dropna()
    pos_trend = yoy_changes > 0
    pos_score = zscores.loc[pos_trend.index] > 0
    
    regime = pd.Series(index=pos_trend.index, dtype=str)
    regime.loc[pos_trend & pos_score] = 'expansion'
    regime.loc[pos_trend & ~pos_score] = 'recovery'
    regime.loc[~pos_trend & ~pos_score] = 'recession'
    regime.loc[~pos_trend & pos_score] = 'peak'
    
    # Only change regime after 
    # a) two consecutive periods of uniform changes or
    mtly_change = regime.ne(regime.shift(2)) & regime.eq(regime.shift())
    
    # b) a monthly change of more than one std above its average
    mtly_peaks = zscores.diff() > \
        (zscores.expanding().mean() + zscores.expanding().std())
    
    regime_final = regime.where(mtly_change | mtly_peaks).fillna(method='ffill')
    
    return regime_final

def plot_regime_model(regimes, zscores, filepath, filename='regime_model.png',
                      year_interval=1, nber=None, title=None):
    '''
    Function to plot regime model based on z-scores

    Parameters
    ----------
    regimes : pd.Series
        Regime classifications.
    zscores : pd.Series
        Z scores over time..
    filepath : string
        Directory to save the graphic.
    filename : string, optional
        Name of the file. The default is 'regime_model.png'.
    year_interval : int, optional
        Tick interval for xaxis. The default is 1.
    nber : pd.Series of boolean, optional
        Series indicating recession periods (to be shaded grey). The default is None.
    title : string, optional
        Title name of the graphic. The default is None.


    '''
    # Prepare data and concat regimes with scores
    df = pd.concat([regimes, zscores], axis=1).dropna()
    df.columns = ['regime', 'score']
    rn = np.sort(df['regime'].unique())
    
    # Convert dates to numbers and construct segments
    inxval = dates.date2num(df.index)
    points = np.array([inxval, df['score'].values]).T.reshape(-1,1,2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # Construct color map
    cm = dict(zip(rn, list("gyrb")))
    colors = list(map(cm.get, df['regime']))
    
    # Plotting section
    fig, ax = plt.subplots(1,1, figsize=(12,8))    
    ymin, ymax = df['score'].min()*1.05, df['score'].max()*1.05
    
    if nber is not None:
        nber = nber.loc[df.index]
        ax.fill_between(nber.index, y1=ymin, y2=ymax, where=nber, alpha=0.4, 
                        facecolor='grey')
        
    lc = LineCollection(segments, colors=colors)
    ax.add_collection(lc)    
    
    ax.axhline(y=0, color='black')
    ax.set_axisbelow(True)
    ax.grid(True, which='major', linewidth=1)       
    
    #Configurate xaxis
    ax.set_xlim([inxval.min(), inxval.max()])
    #ax.set_xlim([df.index.min(), df.index.max()])
    years = dates.YearLocator(year_interval)
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(dates.DateFormatter('%Y'))
    ax.get_xaxis().set_tick_params(which='major', pad=10)
    plt.setp(ax.get_xmajorticklabels(), rotation=0, weight='bold', ha='center')
    
    # Configurate title and yaxis
    ax.set_title(title)
    ax.set_ylabel('z-score')
    ax.set_ylim([ymin, ymax])
    
    
    
    # Configurate legend
    def make_proxy(regime, cm, **kwargs):
        color = cm.get(regime)
        return Line2D([0, 1], [0, 1], color=color, **kwargs)

    proxies = [make_proxy(item, cm, linewidth=5) for item in rn]
    ax.legend(proxies, rn, loc='lower center', ncol=len(rn))
  
    # Save figure
    fig.savefig(os.path.join(filepath, filename), bbox_inches='tight')
            




