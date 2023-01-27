# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 13:05:51 2023

@author: smalswad
"""

import os

from datetime import datetime
from pathlib import Path

def create_subfolder(directory_from_cwd='outputs', timestamp=None, name=None):
    '''
    Create subfolder based on given directory 

    Parameters
    ----------
    directory_from_cwd : str, optional
        String of nested directory to include in folder hierarchy. Logic of 
        hierarchy: 'cwd//directory_from_cwd//timestamp'. The default is
        'outputs'.
        
    timestamp : str, optional
        Timestamp used as registry for output. In None given, use current time.
        The default is None
    name : str, optional
        Additional name to add to folder name. The default is None.

    Returns
    -------
    output_path : str
        Created path name.

    '''
    if name is None:
        name = '' 
   
    if timestamp is None:
       timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
   
    output_path = os.path.join(
        os.getcwd(),
        directory_from_cwd,
        timestamp,
        name)
    
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    return output_path