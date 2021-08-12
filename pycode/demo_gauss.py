#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo showing how km_dict and insegtannotator may be used together for
interactive segmentation. 

@author: vand and abda
"""

import sys
import insegtprobannotator
import skimage.io
import skimage.data
import km_dict
import gauss_feat
import numpy as np

#%% EXAMPLE 1: glass fibres

## loading image
print('Loading image')
filename = '../data/glass.png'
image = skimage.io.imread(filename)
# image = (skimage.transform.rescale(image, 2)*255).astype(np.uint8)
#%% EXAMPLE 2: nerve fibres

## loading image
print('Loading image')
filename = '../data/nerve_im_scale.png'
image = skimage.io.imread(filename)

#%% EXAMPLE 3: randen image

## loading image
print('Loading image')
filename = '../data/bee_eye.png'
image = (skimage.color.rgb2gray(skimage.io.imread(filename)[:,:,0:3])*255).astype(np.uint8)

#%%
import skimage.transform
image = skimage.io.imread('/Users/abda/Documents/Projects/data/20210305_histology/M41-20_1_Wholeslide_Default_Extended.tif')
image = skimage.color.rgb2gray(image)
image = (skimage.transform.rescale(image, 0.2)*255).astype(np.uint8)
# image = (skimage.transform.rescale(skimage.io.imread('/Users/abda/Documents/Teaching/02506/2021/Illustrations/data/Bone/im_08_450.png'), 0.5)*255).astype(np.uint8)
print(image.shape)

#%%
import skimage.transform

# image = (skimage.transform.rescale(skimage.io.imread('/Users/abda/Documents/Teaching/Vejledning/Ida/data/PD6X2_img17_NormalizedContrast_crop.png'), 0.5)*255).astype(np.uint8)
# image = (skimage.transform.rescale(skimage.io.imread('/Users/abda/Documents/Teaching/Vejledning/Ida/data/PD6X2_img10_NormalizedContrast.png'), 0.5)*255).astype(np.uint8)
image = (skimage.transform.rescale(skimage.io.imread('/Users/abda/Documents/Teaching/Vejledning/Ida/data/PD6X2_img17_NormalizedContrast.png'), 0.5)*255).astype(np.uint8)
# image = image[:,:,0]
print(image.shape)


#%% EXAMPLE: Tungsten
import matplotlib.pyplot as plt
## loading image
print('Loading image')
# filename = '../../data/fibre_tungsten.png'
# filename = '../../data/cells/cebolla-05.jpg'
# filename = '../../data/randen15B.png'
# image = (skimage.transform.rescale(skimage.io.imread(filename),1)*255).astype(np.uint8)

filename = '../../data/onionst40.jpg'
image = (skimage.transform.rescale(skimage.color.rgb2gray(skimage.io.imread(filename))*255,1)).astype(np.uint8)

#%% COMMON PART

int_patch_size = 9
branching_factor = 5
number_layers = 5
number_training_patches = 30000
normalization = False


image_float = image.astype(np.float)/255

# Compute feature image
sigma = [1,2,4]
feat_im, norm_fac = gauss_feat.get_gauss_feat_multi(image_float, sigma)

# feat_im, vec, mean_patch, norm_fac = feat_seg.get_uni_pca_feat(image_float, patch_size_feat, n_train, n_keep, order_keep, sigma = -1)
feat_im = np.asarray(feat_im.transpose(2,0,1), order='C')

# Build tree
T = km_dict.build_km_tree(feat_im, 1, branching_factor, number_training_patches, number_layers, normalization)

# Search km-tree and get assignment
A, number_nodes = km_dict.search_km_tree(feat_im, T, branching_factor, normalization)
# number of repetitions for updating the segmentation
number_repetitions = 2

def processing_function(labels):
    r,c = labels.shape
    l = np.max(labels)+1
    if(l>1):
        label_image = np.zeros((r,c,l))
        for k in range(number_repetitions):
            for i in range(1,l):
                label_image[:,:,i] = (labels == i).astype(float)
            D = km_dict.improb_to_dictprob(A, label_image[:,:,1:], number_nodes, int_patch_size) # Dictionary
            P = km_dict.dictprob_to_improb(A, D, int_patch_size) # Probability map
            labels = (np.argmax(P,axis=2) + 1)*(np.sum(P,axis=2)>0)# Segmentation
    else:
        P = np.empty((r,c,0))
        D = None
    return labels, P.transpose(2,0,1), D

pf = lambda labels: processing_function(labels)[:2]

print('Showtime')    

# showtime
app = insegtprobannotator.PyQt5.QtWidgets.QApplication(sys.argv) 
ex = insegtprobannotator.InSegtProbAnnotator(image, pf)
sys.exit(app.exec_())

