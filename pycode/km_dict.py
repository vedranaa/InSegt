#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 17:42:18 2020

@author: abda
"""

import ctypes
import numpy.ctypeslib as ctl 
import numpy as np
libfile = 'km_dict_lib.so'
lib = ctypes.cdll.LoadLibrary(libfile)


def build_km_tree(image, patch_size, branching_factor, number_training_patches, number_layers, normalization=False):
    
    # python function for building km_tree
    py_km_tree = lib.build_km_tree
    # say which inputs the function expects
    py_km_tree.argtypes = [ctl.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                             ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                             ctypes.c_bool,
                             ctl.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]
    # say which output the function gives
    py_km_tree.restype = None
    
    rows, cols = image.shape[0:2]
    channels = 1
    if ( image.ndim > 2 ):
        channels = image.shape[2]
    
    total_patches = (rows-patch_size+1)*(cols-patch_size+1)
    if (number_training_patches > total_patches ):
        number_training_patches = total_patches
    print(f'number of training patches {number_training_patches}')
    # number of elements in tree
    n = int((branching_factor**(number_layers+1)-branching_factor)/(branching_factor-1))
    while( n > number_training_patches ):
        number_layers -= 1
        n = int((branching_factor**(number_layers+1)-branching_factor)/(branching_factor-1))
        print(f'number of layers {number_layers} number of elements {n}')
        
    # make input
    tree = np.empty((n, patch_size*patch_size*channels), dtype=np.float) # will be overwritten
    image = np.asarray(image, order='C')
    
    py_km_tree(image, rows, cols, channels, patch_size, number_layers, branching_factor, number_training_patches, normalization, tree)
    
    return tree



def search_km_tree(image, tree, branching_factor, normalization=False):
    # say where to look for the function
    py_search_km_tree = lib.search_km_tree
    # say which inputs the function expects
    py_search_km_tree.argtypes = [ctl.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                             ctypes.c_int, ctypes.c_int, ctypes.c_int,
                             ctl.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                             ctypes.c_int, ctypes.c_int, ctypes.c_int,
                             ctypes.c_bool,
                             ctl.ndpointer(ctypes.c_int, flags="C_CONTIGUOUS")]
    # say which output the function gives
    py_search_km_tree.restype = None
    
    rows, cols = image.shape[0:2]
    channels = 1
    if ( image.ndim > 2 ):
        channels = image.shape[2]
    
    number_nodes = tree.shape[0]
    patch_size = int(np.sqrt(tree.shape[1]/channels))
    image = np.asarray(image, order='C')
    
    # make input
    A = np.empty((rows,cols), dtype=np.int32) # will be overwritten

    py_search_km_tree(image, rows, cols, channels, tree, patch_size, number_nodes, branching_factor, normalization, A)
    
    return A, number_nodes


def improb_to_dictprob(A, P, number_nodes, patch_size):

    # say where to look for the function
    py_prob_to_dict = lib.prob_im_to_dict
    # say which inputs the function expects
    py_prob_to_dict.argtypes = [ctl.ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                             ctypes.c_int, ctypes.c_int,
                             ctl.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                             ctypes.c_int, ctypes.c_int, ctypes.c_int,
                             ctl.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]
    # say which output the function gives
    py_prob_to_dict.restype = None 

    Pt = np.array(P.transpose((2,0,1)), order='C')
    
    rows, cols = A.shape
    number_layers = P.shape[2]
    
    # make input
    D = np.empty((number_nodes, number_layers*patch_size**2), dtype=np.float) # will be overwritten

    py_prob_to_dict(A, rows, cols, Pt, number_layers, patch_size, number_nodes, D)
    
    return D


def dictprob_to_improb(A, D, patch_size):
    '''
    

    Parameters
    ----------
    A : numpy array
        Assignment image.
    D : numpy array
        dictionary probabilities. Each column contains patch_size x patch_size 
        x layers probabilities.
    patch_size : int
        side length of patch.

    Returns
    -------
    TYPE numpy array
        Probability image of size rows x cols x layers.

    '''

    # say where to look for the function
    py_dict_to_prob_im_opt = lib.dict_to_prob_im_opt
    # say which inputs the function expects
    py_dict_to_prob_im_opt.argtypes = [ctl.ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                              ctypes.c_int, ctypes.c_int,
                              ctl.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                              ctypes.c_int, ctypes.c_int,
                              ctl.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]
    # say which output the function gives
    py_dict_to_prob_im_opt.restype = None 
    
    rows, cols = A.shape
    number_layers = int(D.shape[1]/(patch_size**2))
    

    # make input
    P = np.empty((number_layers, rows, cols)) # will be overwritten
    
    py_dict_to_prob_im_opt(A, rows, cols, D, patch_size, number_layers, P)
    return P.transpose((1,2,0))




















