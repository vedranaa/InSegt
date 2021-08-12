#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 21:46:41 2021

@author: abda
"""

import numpy as np
import scipy.ndimage
import cv2

def get_gauss_feat_im(im, sigma=1, normalize=True, norm_fac = None):
    """Gauss derivative feaures for every image pixel.
    Arguments:
        image: a 2D image, shape (r,c).
        sigma: standard deviation for Gaussian derivatives.
        normalize: flag indicating normalization of features.
    Returns:
        imfeat: a 3D array of size (r,c,15) with a 15-dimentional feature
            vector for every image pixel.
    Author: vand@dtu.dk, 2020
    """
      
    s = np.ceil(sigma*4)
    x = np.arange(-s,s+1).reshape((-1,1));

    g = np.exp(-x**2/(2*sigma**2));
    g /= np.sum(g);
    dg = -x/(sigma**2)*g;
    ddg = -1/(sigma**2)*g - x/(sigma**2)*dg;
    dddg = -2/(sigma**2)*dg - x/(sigma**2)*ddg;
    ddddg = -2/(sigma**2)*ddg - 1/(sigma**2)*ddg - x/(sigma**2)*dddg;
    
    r,c = im.shape
    imfeat = np.zeros((r,c,15))
    imfeat[:,:,0] = cv2.filter2D(cv2.filter2D(im,-1,g),-1,g.T)
    imfeat[:,:,1] = cv2.filter2D(cv2.filter2D(im,-1,dg),-1,g.T)
    imfeat[:,:,2] = cv2.filter2D(cv2.filter2D(im,-1,g),-1,dg.T)
    imfeat[:,:,3] = cv2.filter2D(cv2.filter2D(im,-1,ddg),-1,g.T)
    imfeat[:,:,4] = cv2.filter2D(cv2.filter2D(im,-1,dg),-1,dg.T)
    imfeat[:,:,5] = cv2.filter2D(cv2.filter2D(im,-1,g),-1,ddg.T)
    imfeat[:,:,6] = cv2.filter2D(cv2.filter2D(im,-1,dddg),-1,g.T)
    imfeat[:,:,7] = cv2.filter2D(cv2.filter2D(im,-1,ddg),-1,dg.T)
    imfeat[:,:,8] = cv2.filter2D(cv2.filter2D(im,-1,dg),-1,ddg.T)
    imfeat[:,:,9] = cv2.filter2D(cv2.filter2D(im,-1,g),-1,dddg.T)
    imfeat[:,:,10] = cv2.filter2D(cv2.filter2D(im,-1,ddddg),-1,g.T)
    imfeat[:,:,11] = cv2.filter2D(cv2.filter2D(im,-1,dddg),-1,dg.T)
    imfeat[:,:,12] = cv2.filter2D(cv2.filter2D(im,-1,ddg),-1,ddg.T)
    imfeat[:,:,13] = cv2.filter2D(cv2.filter2D(im,-1,dg),-1,dddg.T)
    imfeat[:,:,14] = cv2.filter2D(cv2.filter2D(im,-1,g),-1,ddddg.T)

    # r,c = im.shape
    # imfeat = np.zeros((r,c,15))
    # imfeat[:,:,0] = scipy.ndimage.gaussian_filter(im,sigma,order=0)
    # imfeat[:,:,1] = scipy.ndimage.gaussian_filter(im,sigma,order=[0,1])
    # imfeat[:,:,2] = scipy.ndimage.gaussian_filter(im,sigma,order=[1,0])
    # imfeat[:,:,3] = scipy.ndimage.gaussian_filter(im,sigma,order=[0,2])
    # imfeat[:,:,4] = scipy.ndimage.gaussian_filter(im,sigma,order=[1,1])
    # imfeat[:,:,5] = scipy.ndimage.gaussian_filter(im,sigma,order=[2,0])
    # imfeat[:,:,6] = scipy.ndimage.gaussian_filter(im,sigma,order=[0,3])
    # imfeat[:,:,7] = scipy.ndimage.gaussian_filter(im,sigma,order=[1,2])
    # imfeat[:,:,8] = scipy.ndimage.gaussian_filter(im,sigma,order=[2,1])
    # imfeat[:,:,9] = scipy.ndimage.gaussian_filter(im,sigma,order=[3,0])
    # imfeat[:,:,10] = scipy.ndimage.gaussian_filter(im,sigma,order=[0,4])
    # imfeat[:,:,11] = scipy.ndimage.gaussian_filter(im,sigma,order=[1,3])
    # imfeat[:,:,12] = scipy.ndimage.gaussian_filter(im,sigma,order=[2,2])
    # imfeat[:,:,13] = scipy.ndimage.gaussian_filter(im,sigma,order=[3,1])
    # imfeat[:,:,14] = scipy.ndimage.gaussian_filter(im,sigma,order=[4,0])

    if normalize:
        if norm_fac is None:
            norm_fac = np.zeros((15,2))
            norm_fac[:,0] = np.mean(imfeat,axis=(0,1))
            norm_fac[:,1] = np.std(imfeat,axis=(0,1))
        imfeat -= norm_fac[:,0]
        imfeat /= norm_fac[:,1]
    
    return imfeat, norm_fac

def get_gauss_feat_multi(im, sigma = [1,2,4], normalize = True, norm_fac = None):
    '''Multi-scale Gauss derivative feaures for every image pixel.
    Arguments:
        image: a 2D image, shape (r,c).
        sigma: list of standard deviations for Gaussian derivatives.
        normalize: flag indicating normalization of features.
    Returns:
        imfeat: a 3D array of size (r,c,15*n_scale) with a 15*n_scale-dimentional feature
            n_scale is length of sigma
            vector for every image pixel.
    Author: abda@dtu.dk, 2021

    '''
    imfeats = []
    if norm_fac is None:
        norm_fac = [None]*len(sigma)
    for i in range(0,len(sigma)):
        feat, norm_fac[i] = get_gauss_feat_im(im, sigma[i], normalize, norm_fac[i])
        imfeats.append(feat)
    imfeats = np.asarray(imfeats).transpose(1,2,3,0)
    
    r,c,nf,ns = imfeats.shape
    # imfeats = np.asarray(imfeats).transpose(1,0,2)
    return imfeats.reshape((r,c,nf*ns)), norm_fac
