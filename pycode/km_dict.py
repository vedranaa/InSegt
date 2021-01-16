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
    '''
    Builds a k-means search tree from image patches. Image patches of size 
    patch_size are extacted, and the tree is build by hierarchical k-means where 
    k is the branching_factor. To reduce processing time, the tree is build from 
    number_training_patches. If this exceeds the the total number of patches in 
    the image, then the trainin is done on all possible patches from the image.
    The resulting kmeans tree is a 2D array.

    Parameters
    ----------
    image : numpy array
        2D image with variable number of channels.
    patch_size : integer
        Side length of patch.
    branching_factor : integer
        Branching of kmtree.
    number_training_patches : integer
        Number of patches used for training the kmtree. If the number exceeds 
        the total number of patches in the image, then the trainin is done on 
        all possible patches from the image.
    number_layers : integer
        Number of layeres in the kmtree.
    normalization : Boolean, optional
        Normalize patches to unit length. The default is False.

    Returns
    -------
    tree : numpy array
        kmtree which is a 2D array. Each row is the average patch intensities 
        for the node. The number of column elements are patch_size x patch_size x
        channels in the image.

    '''
    # Check patch size input
    if ( patch_size < 0 or patch_size%2 != 1 ):
        print('Patch size must be positive and odd!')
        return -1
    
    
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
        channels, rows, cols = image.shape
    
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
    '''
    Search kmtree for all patches in the image to create an assignment image. 
    The assignment image is an image of indicies of closest kmtree node for 
    each pixel.
    
    Parameters
    ----------
    image : numpy array
        2D image.
    tree : numpy array
        kmtree which is a 2D array. Each row is the average patch intensities 
        for the node.
       DESCRIPTION.
    branching_factor : integer
        Branching of kmtree.
    normalization : Boolean, optional
        Normalize patches to unit length. The default is False.

    Returns
    -------
    A : numpy array
        Assignment image of the same rows and cols as the input image.
    number_nodes : integer
        Number of nodes in kmtree.

    '''
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
        channels, rows, cols = image.shape
    
    number_nodes = tree.shape[0]
    patch_size = int(np.sqrt(tree.shape[1]/channels))
    image = np.asarray(image, order='C')
    
    # make input
    A = np.empty((rows,cols), dtype=np.int32) # will be overwritten

    py_search_km_tree(image, rows, cols, channels, tree, patch_size, number_nodes, branching_factor, normalization, A)
    
    return A, number_nodes


def improb_to_dictprob(A, P, number_nodes, patch_size):
    '''
    Taking image probabilities and assigning them to dictionary probabilities 
    according to an assignment image A.

    Parameters
    ----------
    A : numpy array
        Assignment image with the same rows and cols as the input image.
    P : numpy array
        Probability image with the same rows and cols as the image and channels
        the same as number of labels in the iamge.
    number_nodes : integer
        Number of nodes in kmtree.
    patch_size : integer
        Side length of patch. Patch size is used for determining number of labels.

    Returns
    -------
    D : numpy array
        Dictionary probabilities. Each row corresponds to the kmtree node and 
        the cols encodes the pixelwise probability. There are patch_size x 
        patch_size x number of labels.

    '''
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
    Dictionary probabilities to iamge probabilities.

    Parameters
    ----------
    A : numpy array
        Assignment image.
    D : numpy array
        Dictionary probabilities. Each column contains patch_size x patch_size 
        x layers probabilities.
    patch_size : integer
        Side length of patch.

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




















