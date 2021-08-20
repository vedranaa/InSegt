#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo showing how km_dict and insegtannotator may be used together for
interactive segmentation. 

@author: vand and abda
"""

import sys
import insegtannotator
import skimage.io
import skimage.data
import km_dict
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

#%% COMMON PART

patch_size = 11
branching_factor = 5
number_layers = 5
number_training_patches = 35000
normalization = False

image_float = image.astype(np.float)/255

# Build tree
T = km_dict.build_km_tree(image_float, patch_size, branching_factor, number_training_patches, number_layers, normalization)
# Search km-tree and get assignment
A, number_nodes = km_dict.search_km_tree(image_float, T, branching_factor, normalization)
# number of repetitions for updating the segmentation
number_repetitions = 2

def processing_function(labels):
    r,c = labels.shape
    l = np.max(labels)+1
    label_image = np.zeros((r,c,l))
    for k in range(number_repetitions):
        for i in range(1,l):
            label_image[:,:,i] = (labels == i).astype(float)
        D = km_dict.improb_to_dictprob(A, label_image, number_nodes, patch_size) # Dictionary
        P = km_dict.dictprob_to_improb(A, D, patch_size) # Probability map
        labels = np.argmax(P,axis=2) # Segmentation
    return labels

print('Showtime')    

# showtime
app = insegtannotator.PyQt5.QtWidgets.QApplication([]) 
ex = insegtannotator.InSegtAnnotator(image, processing_function)
app.exec()
sys.exit()
