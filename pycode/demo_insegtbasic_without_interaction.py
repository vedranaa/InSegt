#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 12:11:48 2020
@author: vand@dtu.dk
"""

import PIL.Image  # using PIL for easier treatment of indexed rgb images
import numpy as np
import matplotlib.pyplot as plt
import insegtbasic

    
# Reading image and labels.
image = np.array(PIL.Image.open('../data/glass.png'))
labels = np.array(PIL.Image.open('../data/glass_labels.png'))
nr_classes = np.max(labels)

# Settings.
patch_size = 9
nr_training_patches = 10000 # number of randomly extracted patches for clustering
nr_clusters = 1000 # number of dictionary clusters

# Pre-processing.
T1, T2 = insegtbasic.patch_clustering(image, patch_size, nr_training_patches, nr_clusters)

# Processing.
segmentation = insegtbasic.two_binarized(labels, T1, T2)

#%% Visualization
fig, ax = plt.subplots(1, 3, sharex = True, sharey = True, figsize=(15,5))
ax[0].imshow(image, cmap='gray')
ax[0].set_title('image')
ax[1].imshow(labels, vmin=0, vmax=nr_classes, cmap = insegtbasic.gray_cool(nr_classes))
ax[1].set_title('labels')
ax[2].imshow(segmentation, vmin=0, vmax=nr_classes, cmap = insegtbasic.gray_cool(nr_classes))
ax[2].set_title('segmentation')

#fig.savefig('demo_insegtbasic.png')