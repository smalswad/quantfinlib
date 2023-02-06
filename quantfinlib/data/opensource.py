# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 19:04:15 2022

@author: ROB6738
"""

import os
import pandas as pd
import urllib.request
import zipfile

class OpenSourceData(object):
    def __init__(self, directory):
        self.url = None
        self.data = None
        self.directory = directory
    
    def set_url(self, url):
        self.url = url
        
    def wrapper(self, download=False):
        '''
        Helper function to wrap up the loading & cleaning process as defined
        in every child class.

        Parameters
        ----------
        download : bool, optional
            Indicating whether data should be downloaded or just loaded
            from local directory. The default is False.

        Returns
        -------
        None.

        '''
        if download:
            self.download_data()
            
        self.load_data()
        self.clean_data()
    
    def download_data(self):
        raise NotImplementedError()


class FamaFrenchData(OpenSourceData):
    def __init__(self, *args, url=None, auto=False, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Set url to access data
        if url is None:            
            b = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
            factors = "F-F_Research_Data_Factors_CSV.zip"
            self.set_url(b+factors)
        else:
            self.set_url(url)
            
        # Run standardized loading and updating
        if auto:
            self.wrapper()

    def clean_data(self, N=None):
        '''
        Function to clean the data based on standard Fama French format

        Parameters
        ----------
        N : int, optional
            Index of last monthly observation. The default is None.

        Returns
        -------
        None.

        '''
        if N is None:
            # Get number of second part of data
            N = self.data.loc[self.data.iloc[
                :,0]== ' Annual Factors: January-December '].index.values[0]
       
        # Trim data
        self.data = self.data.iloc[:N, :]
        
        # Adjust index 
        self.data.set_index('Unnamed: 0', inplace=True, drop=True)
        self.data.index = pd.to_datetime(self.data.index, format= '%Y%m')
        self.data.index = self.data.index + pd.offsets.MonthEnd(0)
        self.data.index.rename('date', inplace=True)
        
        # Scale data
        self.data = self.data.astype('float') /100  
        
        # Rename RMRF
        self.data = self.data.rename(columns={'Mkt-RF':'RMRF'})
    
    def download_data(self):
        '''
        Function to download Fame French data as specified by class URL. 
        Unzippes data in class directory.

        Returns
        -------
        None.

        '''
        temp_zip = self.directory+'/ff_data.zip'
        urllib.request.urlretrieve(self.url, temp_zip)
        zip_file = zipfile.ZipFile(temp_zip, 'r')
        zip_file.extractall(path=self.directory)
        zip_file.close()
        os.remove(temp_zip)
                 
    def load_data(self):
        '''
        Function to load data from given directory.

        Returns
        -------
        None.

        '''
        file = self.directory + '/F-F_Research_Data_Factors.csv'
        self.data = pd.read_csv(file, skiprows=3)       
        
    def save_data(self):
        pass
    
    
class OpenAssetPricingData(OpenSourceData):
    def __init__(self, file_name, *args, auto=False, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.file = file_name
        
        # Run standardized loading and updating
        if auto:
            self.wrapper()
        
    def load_data(self):
        # Load OAP data
        df = pd.read_csv(os.path.join(self.directory, self.file))
        self.data = df.loc[df['port']=='LS', :]
    
    def clean_data(self):
        # Convert df to wide format
        self.data = self.data.pivot(
            index='date', columns='signalname', values='ret') / 100
        self.data.index = pd.to_datetime(self.data.index, format='%Y-%m-%d') \
            + pd.offsets.MonthEnd(0)
    
        
class KellyData(OpenSourceData):
    def __init__(self, file_name, *args, auto=False, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.file = file_name
        
        # Run standardized loading and updating
        if auto:
            self.wrapper()
            
    def load_data(self):
        self.data = pd.read_csv(os.path.join(self.directory, self.file))
        
    def clean_data(self):
        # Convert df to wide format
        self.data = self.data.pivot(
            index='date', columns='name', values='ret')
        self.data.index = pd.to_datetime(self.data.index, format='%Y-%m-%d') \
            + pd.offsets.MonthEnd(0)
    
    
