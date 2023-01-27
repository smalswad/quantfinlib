# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 13:22:24 2023

@author: smalswad
"""

import collections

def apply_to_nested_dict(ob, func, args=None):
    '''
    Apply given function to each element of nested dictionary 

    Parameters
    ----------
    ob : dict
        (Nested) dictionary.
    func : callable
        Function to apply to each element of the dict.
    args : iterable
        Function arguments. The default is None.

    Returns
    -------
    None.

    '''
    for k, v in ob.items():
        if isinstance(v, collections.abc.Mapping):
            apply_to_nested_dict(v, func, args)
        else:
            ob[k] = func(v, args)   