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
import feat_seg
import numpy as np

#%% EXAMPLE 1: glass fibres

## loading image
print('Loading image')
filename = '../data/glass.png'
image = skimage.io.imread(filename)

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

#%% COMMON PART

int_patch_size = 9
branching_factor = 5
number_layers = 5
number_training_patches = 200000
normalization = False

patch_size_feat = 5
n_train = 50000
n_keep = 25
order_keep = (True, True, True)

image_float = image.astype(np.float)/255

# Compute feature image
feat_im, vec, mean_patch, norm_fac = feat_seg.get_uni_pca_feat(image_float, patch_size_feat, n_train, n_keep, order_keep, sigma = -1)
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
    return labels, P.transpose(2,0,1), D

pf = lambda labels: processing_function(labels)[:2]

print('Showtime')    

# showtime
app = insegtprobannotator.PyQt5.QtWidgets.QApplication(sys.argv) 
ex = insegtprobannotator.InSegtProbAnnotator(image, pf)
sys.exit(app.exec_())






#%%
labels = ex.rgbaToLabels(ex.pixmapToArray(ex.annotationPix))
labels1 = np.argmax(ex.probabilities.transpose(1,2,0), axis = 2)
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1,2)
ax[0].imshow(labels)
ax[1].imshow(labels1)


#%%
import matplotlib.colors

labels = ex.rgbaToLabels(ex.pixmapToArray(ex.annotationPix))

segmentation, P_out, D = processing_function(labels)

colors = np.asarray(ex.colors)/255
cm = matplotlib.colors.LinearSegmentedColormap.from_list('insegt_colors', colors, N=len(ex.colors))

l = int(D.shape[1]/(int_patch_size**2))
fig,ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(9,3))
ax[0].imshow(image, cmap = 'gray')
ax[1].imshow(labels, vmin = 0, vmax = len(ex.colors), cmap = cm)
ax[2].imshow(segmentation, vmin = 0, vmax = len(ex.colors), cmap = cm)
plt.show()
fig,ax = plt.subplots(1, figsize=(5,3))
ax.imshow(D, aspect='auto', cmap = 'jet')
plt.show()









# ax.imshow(ex.probabilities.transpose(1,2,0)[:,:,0])

# def processing_function(labels):
#     r,c = labels.shape
#     l = np.max(labels)+1
#     label_image = np.zeros((r,c,l))
#     for k in range(number_repetitions):
#         for i in range(1,l):
#             label_image[:,:,i] = (labels == i).astype(float)
#         D = km_dict.improb_to_dictprob(A, label_image, number_nodes, int_patch_size) # Dictionary
#         P = km_dict.dictprob_to_improb(A, D, int_patch_size) # Probability map
#         labels = np.argmax(P,axis=2) # Segmentation
#     return labels
