#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 00:06:22 2021

@author: abda
"""

import numpy as np
import numpy.random
import scipy.ndimage
import ctypes
import numpy.ctypeslib as ctl 
libfile = 'image_feat_lib.so'
lib = ctypes.cdll.LoadLibrary(libfile)


def get_feat_vec(image, patch_size, n_train, n_keep = 10):

    # Extract patches and put a random subset of n_train into an array
    r,c = image.shape[:2]
    
    l = 1
    if (image.ndim == 3):
        l = image.shape[2]
    
    patch_size_h = int(np.floor(patch_size/2))
    
    n_tot = (r-patch_size+1)*(c-patch_size+1)
    n_train = np.min([n_tot, n_train])
    
    idx = np.random.permutation(n_tot)[:n_train]
    
    c_id = (np.floor(idx/(r-patch_size+1)) + patch_size_h).astype(np.int)
    r_id = ((idx - (c_id - patch_size_h)*(r-patch_size+1)) + patch_size_h).astype(np.int)
    
    P = np.zeros((l*patch_size**2,n_train))
    
    for i in range(0,n_train):
        tmp = image[r_id[i]-patch_size_h:r_id[i]+patch_size_h+1,c_id[i]-patch_size_h:c_id[i]+patch_size_h+1]
        P[:,i] = tmp.ravel()
        
    cv = np.cov(P)
    # cv[cv<1e-5] = 0
    
    # Compute eigen values and eigen vectors of the covariance matrix of the images
    val, vec = np.linalg.eig(cv)
    # print(vec.dtype)
    vec = np.real(vec)#.astype(np.float)
    # print(vec.dtype)
    
    if (n_keep > 1):
        n_keep = np.minimum(n_keep,patch_size**2)
    else:
        v = np.sqrt(val)
        vp = v/np.sum(v)
        i = 0
        vs = 0
        while ( i < vp.shape[0] and vs < n_keep ):
            vs += vp[i]
            i += 1
        n_keep = i
    # eigen vectors for computing PCA on new image patches
    vec = vec[:,0:n_keep]
    
    # Compute the mean patches when PCA should be applied to a new image
    mean_patch = np.mean(P,axis = 1)
    return vec, mean_patch

def get_im_dev(image, order_keep = (True, True, True)):
    # Gaussian filters
    # g, dg, ddg = get_gauss_dev(sigma)
    g = np.array([[1,1,1]])
    dg = np.array([[1,0,-1]])
    ddg = np.array([[1,-2,1]])
    
    # image and derivatives
    I = []
    if ( order_keep[0] ):
        I.append(np.asarray(image, order='C'))
    if ( order_keep[1] ):
        I.append(np.asarray(scipy.ndimage.convolve(image, dg), order='C')) # Ix
        I.append(np.asarray(scipy.ndimage.convolve(image, dg.T), order='C')) # Iy
    if ( order_keep[2] ):
        I.append(np.asarray(scipy.ndimage.convolve(image, ddg), order='C')) # Ixx
        I.append(np.asarray(scipy.ndimage.convolve(scipy.ndimage.convolve(image, dg), dg.T), order='C')) # Ixy
        I.append(np.asarray(scipy.ndimage.convolve(image, ddg.T), order='C')) # Iyy
    # if ( order_keep[0] ):
    #     I.append(np.asarray(image, order='C'))
    # if ( order_keep[1] ):
    #     I.append(np.asarray(scipy.ndimage.convolve(scipy.ndimage.convolve(image, dg), g.T), order='C')) # Ix
    #     I.append(np.asarray(scipy.ndimage.convolve(scipy.ndimage.convolve(image, g), dg.T), order='C')) # Iy
    # if ( order_keep[2] ):
    #     I.append(np.asarray(scipy.ndimage.convolve(scipy.ndimage.convolve(image, ddg), g.T), order='C')) # Ixx
    #     I.append(np.asarray(scipy.ndimage.convolve(scipy.ndimage.convolve(image, dg), dg.T), order='C')) # Ixy
    #     I.append(np.asarray(scipy.ndimage.convolve(scipy.ndimage.convolve(image, g), ddg.T), order='C')) # Iyy
    # Normalize to have same standard deviation
    std_im = np.std(image)
    for i in range(0,len(I)):
        I[i] = I[i]*std_im/np.std(I[i])
    return I
    


def get_gauss_dev(sigma, size=4):
    x = np.c_[np.arange(-np.ceil(size*size),np.ceil(size*size)+1)].T
    g = np.exp(-x**2/(2*size*size))
    g /= np.sum(g)
    dg = -x/(size*size)*g
    ddg = -1/(size*size)*g -x/(size*size)*dg
    return g, dg, ddg




def get_pca_feat_im(I, vec, mean_patch, order_keep = (True, True, True)):
    # python function for building km_tree
    py_vec_to_feat_im = lib.vec_to_feat_im
    
    # extern "C" void vec_to_feat_im(const double *I, int rows, int cols, int channels, const double *vec, int n_keep, int patch_size, const double *mean_patch, double *feat_im) 
    
    # say which inputs the function expects
    py_vec_to_feat_im.argtypes = [ctl.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), # Image
                             ctypes.c_int, ctypes.c_int, ctypes.c_int, # rows, cols, channels
                             ctl.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), # eigen vectors
                             ctypes.c_int, ctypes.c_int,  # n_keep, patch_size
                             ctl.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), # mean_patch
                             ctl.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")] # feat_image
    # say which output the function gives
    py_vec_to_feat_im.restype = None
    
    patch_size = int(np.sqrt(mean_patch[0].shape[0]))

    rows, cols = I[0].shape[:2]
    channels = 1
    if ( I[0].ndim == 3 ):
        channels = I[0].shape[2]
    
    
    # Compute feature vectors
    feat_im = np.array([])
    for i in range(0,len(I)):
        # Get right order for vector and mean patch
        vec_in = np.asarray(vec[i].transpose(), order = 'C')
        mean_patch_in = np.asarray(mean_patch[i], order='C')
        n_keep = vec_in.shape[0]
        # print(f'rows {rows} cols {cols} channels {channels} n_keep {n_keep} patch_size {patch_size}')
        # print(f'I {I[i].dtype} vec {vec_in.dtype} mean_patch {mean_patch_in.dtype}')
        feat_im_cpp = np.empty((rows, cols, n_keep), dtype=np.float) # will be overwritten
        py_vec_to_feat_im(I[i], rows, cols, channels, vec_in, n_keep, patch_size, mean_patch_in, feat_im_cpp)
        if (feat_im.size == 0):
            feat_im = feat_im_cpp
        else:
            feat_im = np.append(feat_im, feat_im_cpp, axis = 2)
    return feat_im


def get_pca_feat(im, patch_size = None, n_train = None, n_keep = None, order_keep = (True, True, True), vec = None, mean_patch = None):
    I = get_im_dev(im, order_keep = order_keep)
    if ( vec == None ):
        vec = []
        mean_patch = []
        for im in I:
            vec_tmp, mean_patch_tmp = get_feat_vec(im, patch_size, n_train, n_keep)
            vec.append(vec_tmp)
            mean_patch.append(mean_patch_tmp)
    feat_im = get_pca_feat_im(I, vec, mean_patch, order_keep = order_keep)
    return feat_im, vec, mean_patch



def get_pca_feat_slow(I, vec, mean_patch, order_keep = (True, True, True), sigma = 1):
    # python function for building km_tree
    py_vec_to_feat_im = lib.vec_to_feat_im_slow
    
    # extern "C" void vec_to_feat_im(const double *I, int rows, int cols, int channels, const double *vec, int n_keep, int patch_size, const double *mean_patch, double *feat_im) 
    
    # say which inputs the function expects
    py_vec_to_feat_im.argtypes = [ctl.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), # Image
                             ctypes.c_int, ctypes.c_int, ctypes.c_int, # rows, cols, channels
                             ctl.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), # eigen vectors
                             ctypes.c_int, ctypes.c_int,  # n_keep, patch_size
                             ctl.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), # mean_patch
                             ctl.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")] # feat_image
    # say which output the function gives
    py_vec_to_feat_im.restype = None
    
    patch_size = int(np.sqrt(mean_patch[0].shape[0]))

    rows, cols = I[0].shape[:2]
    channels = 1
    if ( I[0].ndim == 3 ):
        channels = I[0].shape[2]
    
    
    # Compute feature vectors
    feat_im = np.array([])
    for i in range(0,len(I)):
        # Get right order for vector and mean patch
        vec_in = np.asarray(vec[i].transpose(), order = 'C')
        mean_patch_in = np.asarray(mean_patch[i], order='C')
        n_keep = vec_in.shape[0]
        # print(f'rows {rows} cols {cols} channels {channels} n_keep {n_keep} patch_size {patch_size}')
        # print(f'I {I[i].dtype} vec {vec_in.dtype} mean_patch {mean_patch_in.dtype}')
        feat_im_cpp = np.empty((n_keep, rows, cols), dtype=np.float) # will be overwritten
        py_vec_to_feat_im(I[i], rows, cols, channels, vec_in, n_keep, patch_size, mean_patch_in, feat_im_cpp)
        if (feat_im.size == 0):
            feat_im = feat_im_cpp.transpose(1,2,0)
        else:
            feat_im = np.append(feat_im, feat_im_cpp.transpose(1,2,0), axis = 2)
    return feat_im