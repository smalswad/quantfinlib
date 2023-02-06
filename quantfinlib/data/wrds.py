# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 11:14:35 2023

@author: smalswad
"""

import pandas as pd
import wrds
from pandas.tseries.offsets import MonthEnd

db = wrds.Connection()

#Open h5 store
h5file = 'crsp_monthly.h5'
h5 = pd.HDFStore(path=h5file, mode='a')

batch_range = range(1920,2030,10)


# =============================================================================
# CRSP Download (run on WRDS cloud)
# =============================================================================

# download delisting return
dlret = db.raw_sql("""
                     select permno, dlret, dlstdt 
                     from crsp.msedelist
                     
                     """)

# change variable format to int and date, respectively
dlret.permno = dlret.permno.astype(int)
dlret['date'] = pd.to_datetime(dlret['dlstdt']) + MonthEnd(0)

# download crsp data in batches
for start in batch_range:    
    
    #Download chunk
    crsp_d = db.raw_sql("""
                          select a.permno, a.permco, a.date, a.ret, a.retx 
                          a.shrout, a.prc, b.exchcd, b.shrcd
                          from crsp.msf as a
                          left join crsp.msenames as b
                          on a.permno=b.permno
                          and b.namedt<=a.date
                          and a.date<=b.nameendt
                          where b.exchcd between 1 and 3                     
                          and a.date between '01/01/{start}' and '12/31/{end}'
                          """.format(start=start, end=start+10))

    # change variable format to int and date, respectively
    crsp_d[['permco','permno','exchcd','shrcd']] = \
        crsp_d[['permco','permno','exchcd','shrcd']].astype(int)
    crsp_d['date'] = pd.to_datetime(crsp_d['date']) + MonthEnd(0)
    
    
    # add delisting return
    crsp = pd.merge(crsp_d, dlret, how='left',on=['permno','date'])
    crsp['dlret'] = crsp['dlret'].fillna(0)
    crsp['ret'] = crsp['ret'].fillna(0)
    crsp['retadj'] = (1+crsp['ret'])*(1+crsp['dlret'])-1
    
    # calculate market equity
    crsp['me'] = crsp['prc'].abs()*crsp['shrout'] 
    crsp = crsp.drop(['dlret','dlstdt','prc','shrout'], axis=1)
    crsp = crsp.sort_values(by=['date','permco','me'])
    
    
    ### Aggregate Market Cap ###
    # sum of me across different permno belonging to same permco a given date
    crsp_summe = crsp.groupby(['date','permco'])['me'].sum().reset_index()
    # largest mktcap within a permco/date
    crsp_maxme = crsp.groupby(['date','permco'])['me'].max().reset_index()
    # join by jdate/maxme to find the permno
    crsp1 = pd.merge(crsp, crsp_maxme, how='inner', on=['date','permco','me'])
    # drop me column and replace with the sum me
    crsp1 = crsp1.drop(['me'], axis=1)
    # join with sum of me to get the correct market cap info
    crsp2 = pd.merge(crsp1, crsp_summe, how='inner', on=['date','permco'])
    # sort by permno and date and also drop duplicates
    crsp2 = crsp2.sort_values(by=['permno','date']).drop_duplicates()

    del crsp_summe, crsp_maxme, crsp1, crsp
    
    crsp3 = crsp2[['permno','date','retadj','me','exchcd','shrcd']]    
    h5.put(key='crsp_{start}-{end}'.format(start=start, end=start+10), value=crsp3)

#close connections
h5.close()
db.close()


# =============================================================================
# Combining single batches to dataframes (run locally on PC)
# =============================================================================


# dd5 = ('C:\\Users\\Alexander Swade\\OneDrive - Lancaster University\\'
#       'PhD research\\2023_FactorZoo\\Data\\')

# h5file = dd5 + 'crsp_monthly' + '.h5'
# h5 = pd.HDFStore(path=h5file, mode='a')
# crsp_list = list()

# for start in batch_range:
#     crsp_list.append(h5['crsp_{start}-{end}'.format(start=start, end=start+10)])

# crsp = pd.concat(crsp_list).drop_duplicates()

# h5.close()


# h5file = dd5 + 'crsp_monthly_df' + '.h5'
# h5 = pd.HDFStore(path=h5file, mode='a')

# h5.put(key='crsp', value=crsp)

# h5.close()
