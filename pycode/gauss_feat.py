#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 21:46:41 2021

@author: abda, niejep
"""

import cv2
import numpy as np


def get_gauss_feat_im(im, sigma=1, normalize=True, norm_fac=None, dtype='float32'):
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

    # Ensure image is float32.
    # This data type is often much faster than float64.
    if im.dtype != dtype:
        im = im.astype(dtype)

    # Create kernel array.
    s = np.ceil(sigma * 4)
    x = np.arange(-s, s + 1).reshape((-1, 1))

    # Create Gaussian kernels.
    g = np.exp(-x**2 / (2 * sigma**2))
    g /= np.sum(g)
    g = g.astype(im.dtype)  # Make same type as image.
    dg = -x / (sigma**2) * g
    ddg = -1 / (sigma**2) * g - x / (sigma**2) * dg
    dddg = -2 / (sigma**2) * dg - x / (sigma**2) * ddg
    ddddg = -2 / (sigma**2) * ddg - 1 / (sigma**2) * ddg - x / (sigma**
                                                                2) * dddg

    # Create image feature arrays and temporary array. Features are stored
    # on the first access for fast direct write of values in filter2D.
    imfeat = np.zeros((15, ) + im.shape, dtype=im.dtype)
    imfeat_tmp = np.zeros_like(im)

    # Extract features. Order is a bit odd, as original order has been
    # kept even though calculation order has been updated. We use the tmp
    # array to store results and avoid redundant calcalations. This
    # reduces calls to filter2D from 30 to 20. Results are written
    # directly to the destination array.
    cv2.filter2D(im, -1, g, dst=imfeat_tmp)
    cv2.filter2D(imfeat_tmp, -1, g.T, dst=imfeat[0])
    cv2.filter2D(imfeat_tmp, -1, dg.T, dst=imfeat[2])
    cv2.filter2D(imfeat_tmp, -1, ddg.T, dst=imfeat[5])
    cv2.filter2D(imfeat_tmp, -1, dddg.T, dst=imfeat[9])
    cv2.filter2D(imfeat_tmp, -1, ddddg.T, dst=imfeat[14])

    cv2.filter2D(im, -1, dg, dst=imfeat_tmp)
    cv2.filter2D(imfeat_tmp, -1, g.T, dst=imfeat[1])
    cv2.filter2D(imfeat_tmp, -1, dg.T, dst=imfeat[4])
    cv2.filter2D(imfeat_tmp, -1, ddg.T, dst=imfeat[8])
    cv2.filter2D(imfeat_tmp, -1, dddg.T, dst=imfeat[13])

    cv2.filter2D(im, -1, ddg, dst=imfeat_tmp)
    cv2.filter2D(imfeat_tmp, -1, g.T, dst=imfeat[3])
    cv2.filter2D(imfeat_tmp, -1, dg.T, dst=imfeat[7])
    cv2.filter2D(imfeat_tmp, -1, ddg.T, dst=imfeat[12])

    cv2.filter2D(im, -1, dddg, dst=imfeat_tmp)
    cv2.filter2D(imfeat_tmp, -1, g.T, dst=imfeat[6])
    cv2.filter2D(imfeat_tmp, -1, dg.T, dst=imfeat[11])

    cv2.filter2D(im, -1, ddddg, dst=imfeat_tmp)
    cv2.filter2D(imfeat_tmp, -1, g.T, dst=imfeat[10])

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
            norm_fac = np.zeros((2, 15), np.float32)
            norm_fac_mean = np.mean(imfeat, axis=(1, 2), dtype=norm_fac.dtype, out=norm_fac[0])
            norm_fac_std = np.std(imfeat, axis=(1, 2), dtype=norm_fac.dtype, out=norm_fac[1])
        else:
            norm_fac = np.moveaxis(norm_fac, 0, -1)
        imfeat -= norm_fac_mean[:, np.newaxis, np.newaxis]
        imfeat /= norm_fac_std[:, np.newaxis, np.newaxis]

    # Move axis last again.
    imfeat = np.moveaxis(imfeat, 0, -1)
    norm_fac = np.moveaxis(norm_fac, 0, -1)

    return imfeat, norm_fac


def get_gauss_feat_multi(im, sigma=[1, 2, 4], normalize=True, norm_fac=None):
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
        norm_fac = [None] * len(sigma)
    for i in range(0, len(sigma)):
        feat, norm_fac[i] = get_gauss_feat_im(im, sigma[i], normalize,
                                              norm_fac[i])
        imfeats.append(feat)
    imfeats = np.asarray(imfeats).transpose(1, 2, 3, 0)

    r, c, nf, ns = imfeats.shape
    # imfeats = np.asarray(imfeats).transpose(1,0,2)
    return imfeats.reshape((r, c, nf * ns)), norm_fac
