# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 10:11:44 2023

@author: Alexander Swade
"""
import pandas as pd
import numpy as np
from pandas.tseries.offsets import MonthEnd


# =============================================================================
# Functions
# =============================================================================
def constr_2x3_sort(crsp_data, signal_data, hml=True):
    '''
    Function to calculate 2x3 FF1993-style factor returns based on given signal

    Parameters
    ----------
    crsp_data : pd.DataFrame, shape(T,3)
        CRSP data containing stock-specific returns and market cap.
        DF containts columns ['retadj', 'me', 'exchcd'] 
        with multiindex: FrozenList(['permno', 'date']).
    signal_data : pd.Series, shape(T,)
        Signal specific return with multiindex: FrozenList(['permno', 'date']).
    hml : boolean
        Calculate high minus low (True) or low minus high (False). The default
        is True.

    Returns
    -------
    pd.Series, shape(T,)
        Factor return series (based on selected signal).
        
    '''
    sig_name = signal_data.name
    df = crsp_data.join(signal_data)
    
    # Get NYSE breakpoints
    nyse_df = df.loc[df['exchcd']==1]
    nyse_sz = nyse_df.groupby(['date'])['me'].quantile()
    nyse_signal = nyse_df.groupby(['date'])[sig_name].quantile([0.3,0.7])
    nyse_signal = nyse_signal.reset_index().pivot(
        index='date',columns='level_1', values=sig_name)    
    breaks = nyse_signal.join(nyse_sz)
    breaks.columns = ['signal30', 'signal70', 'size50']
        
    # Classify buckets    
    temp = df.join(breaks, on='date')
    temp['q_me'] = np.where(temp['me'] > temp['size50'], 'B', 'S')
    temp['q_sig'] = np.where(temp[sig_name] < temp['signal30'], 'L', '')
    temp['q_sig'].mask((temp[sig_name] >= temp['signal30']) &
                       (temp[sig_name] <= temp['signal70']),'M',inplace=True)
    temp['q_sig'].mask(temp[sig_name] > temp['signal70'], 'H', inplace=True)
    
    # Get lagged market equity 
    temp['me_lag'] = temp.groupby('permno')['me'].shift(1)
    
    # Calculate VW portfolio returns for each bucket
    def vw_ret(group, avg_name, weight_name):
        d = group[avg_name]
        w = group[weight_name]
        try:
            return (d * w).sum() / w.sum()
        except ZeroDivisionError:
            return np.nan
        
    port = temp.groupby(['date','q_me','q_sig']).apply(vw_ret,'retadj','me_lag')
    port = port.dropna()
    
    # Construct 2x3 factor returns based on EW of extremes
    idx = pd.IndexSlice
    portfolio_returns = port.groupby('date').apply(
        lambda x: 0.5*(x[idx[:,'S','H']] + x[idx[:,'B','H']]) \
                - 0.5*(x[idx[:,'S','L']] + x[idx[:,'B','L']]))
    
    # Invert to Low minus High if specified
    if not hml:
        portfolio_returns = portfolio_returns *(-1)
        
    return portfolio_returns.droplevel(1).rename(sig_name)



# Not Run
if __name__ == '__main__':

    dd5 = ('C:\\Users\\...\\2023_FactorZoo\\Data\\')
    
    h5file = dd5 + 'crsp_monthly_df' + '.h5'
    h5 = pd.HDFStore(path=h5file, mode='a')
    
    #Load returns and check for duplicates
    crsp = h5['crsp']
    crsp = crsp.drop_duplicates().reset_index(drop=True)
    crsp = crsp.sort_values(by=['permno','date'])
    
    crsp2 = crsp.set_index(['permno','date'])
    # ret.index = ret.index + MonthEnd(0)
    
    h5.close()
    
    # Calculate factor returns based on monthly rebalancing
    str_dummy = crsp2['retadj'].groupby('permno').shift().rename('str')
    str_factor = constr_2x3_sort(crsp2, str_dummy, hml=False)

    
                                
    
    
    
    
    

    








