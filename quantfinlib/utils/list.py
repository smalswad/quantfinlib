# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 12:49:28 2023

@author: smalswad
"""



def list_intsect(l1, l2, duplicates=False):
    '''
    Return list intersection with same order as list 1

    Parameters
    ----------
    l1 : iterable
        List 1.
    l2 : iterable
        List 2.
    duplicates : boolean, optinal
        Should duplicates be allowed? The default is False.

    Returns
    -------
    list
        Intersection of list1 and list2.

    '''
    lint = [x for x in l1 if x in l2]
    if not duplicates:
        lint = list(dict.fromkeys(lint))
        
    return lint
