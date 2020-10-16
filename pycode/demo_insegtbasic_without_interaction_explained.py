#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 13:08:33 2020
@author: vand@dtu.dk
"""

import PIL.Image  # using PIL for easier treatment of indexed rgb images
import numpy as np
import matplotlib.pyplot as plt
import time
import insegtbasic

# %% Pre-processing
    
# Reading image and labels.
image = np.array(PIL.Image.open('../data/glass.png'))
labels = np.array(PIL.Image.open('../data/glass_labels.png'))
nr_classes = np.max(labels)

# Settings.
patch_size = 9
nr_training_patches = 10000 # number of randomly extracted patches for clustering
nr_clusters = 1000 # number of dictionary clusters

# Preparing for transformation.
t = time.time()
assignment = insegtbasic.image2assignment(image, patch_size, nr_clusters, nr_training_patches)
print(f'Image to assignment: {time.time()-t}')
t = time.time()
B = insegtbasic.assignment2biadjacency(assignment, image.shape, patch_size, nr_clusters)
print(f'Assignment to biadjacency: {time.time()-t}')
t = time.time()
T1, T2 = insegtbasic.biadjacency2transformations(B)
print(f'Biadjacency to transformations: {time.time()-t}')

# %% Incorporating labelings

# Transforming labels into probabilities.
t = time.time()
labcol = insegtbasic.labels2labcol(labels, nr_classes=nr_classes)
probcol = T2*(T1*labcol) # first linear diffusion
probcol = np.asarray(probcol.todense())
print(f'Matrix multiplications: {time.time()-t}')

t = time.time()
labcol = insegtbasic.probcol2labcol(probcol) # binarizing labels
probcol = T2*(T1*labcol) # second linear diffusion
probcol = np.asarray(probcol.todense())
print(f'Second round of matrix multiplications: {time.time()-t}')

t = time.time()
probabilities = insegtbasic.probcol2probabilities(probcol, image.shape)
print(f'Probcol to probabilities: {time.time()-t}')

#%% Producing a final segmentation

segmentation = insegtbasic.probcol2labcol(probcol) * \
        (np.arange(probcol.shape[1], dtype=np.uint8) + 1) # final segmentation
segmentation = segmentation.reshape(image.shape)

#%% Preparing for visualization

# Preparing for visualization of probabilities as rgb image.
P_rgb = np.zeros(probabilities.shape[0:2]+(3,))
k = min(nr_classes, 3)
P_rgb[:,:,:k] = probabilities[:,:,:k]

# Correctly padding assignment image for visuzlization.
a_im = np.zeros(image.shape)
s = (patch_size-1)//2
a_im[s:-s,s:-s] = assignment

# %% Visualization.

fig, ax = plt.subplots(2, 3, sharex = True, sharey = True, figsize=(15, 10))
ax[0,0].imshow(image, cmap='gray')
ax[0,0].set_title('image')
ax[1,0].imshow(a_im)
ax[1,0].set_title('assignment')
ax[0,1].imshow(labels, vmin=0, vmax=nr_classes, cmap = insegtbasic.gray_cool(nr_classes))
ax[0,1].set_title('labels')
ax[1,1].imshow(segmentation, vmin=0, vmax=nr_classes, cmap = insegtbasic.gray_cool(nr_classes))
ax[1,1].set_title('segmentation')
ax[0,2].imshow(probabilities[:,:,0], vmin=0, vmax=1, cmap = 'jet')
ax[0,2].set_title('probablitily label 1')
ax[1,2].imshow(probabilities[:,:,1], vmin=0, vmax=1, cmap = 'jet')
ax[1,2].set_title('probablitily label 2')

fig.savefig('demo_insegtbasic_explained.png')
