#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 13:08:33 2020
@author: vand@dtu.dk
"""

import PIL.Image  # using PIL for easier treatment of indexed rgb images
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import sklearn.cluster
import time
import scipy.sparse

def image2patches(image, patch_size, stepsize=1):
    """Rearrange image patches into columns
    Arguments:
        image: a 2D image, shape (X,Y).
        patch size: size of extracted squared paches.
        stepsize: patch step size.
    Returns:
        patches: a 2D array which in every column has a patch associated with
            one image pixel. For stepsize 1, the number of returned patches
            is (X-patch_size+1)*(Y-patch_size+1) due to bounary. The length
            of columns is patch_size**2.
    """    
    X, Y = image.shape
    s0, s1 = image.strides    
    nrows = X-patch_size+1
    ncols = Y-patch_size+1
    shp = patch_size, patch_size, nrows, ncols
    strd = s0, s1, s0, s1
    out_view = np.lib.stride_tricks.as_strided(image, shape=shp, strides=strd)
    return out_view.reshape(patch_size*patch_size,-1)[:,::stepsize]

def image2assignment(image, patch_size, nr_clusters, nr_training_patches):
    """ Extract, cluster and assign image patches using minibatch k-means."""
    X, Y = image.shape
    patches = image2patches(image, patch_size)
    patches_subset = patches[:,np.random.permutation(np.arange(patches.shape[1]))
                             [:nr_training_patches]]
    kmeans = sklearn.cluster.MiniBatchKMeans(n_clusters = nr_clusters, max_iter = 10, batch_size = 3*nr_clusters)
    kmeans.fit(patches_subset.T)
    assignment = kmeans.predict(patches.T)
    return assignment.reshape((X-patch_size+1,Y-patch_size+1))

def assignment2biadjacency(assignment, image_shape, patch_size, nr_clusters):
    """ Algorithm 1 from https://arxiv.org/pdf/1809.02226.pdf"""
    n = image_shape[0]*image_shape[1]
    m = patch_size*patch_size*nr_clusters
    s = (patch_size-1)//2
    # find displacement in i and j for within-patch positions dx and dy.
    dy, dx = np.meshgrid(np.arange(patch_size)-s,np.arange(patch_size)-s)
    di = (dy + image_shape[1]*dx).ravel()
    dj = (dy + patch_size*dx).ravel()
    # populate index list for every assignment.
    i_accumulate = np.empty((assignment.size,patch_size**2))
    j_accumulate = np.empty((assignment.size,patch_size**2))
    for x in range(assignment.shape[0]):
        for y in range(assignment.shape[1]):
            k = assignment[x,y]
            i = (y+s) + (x+s)*image_shape[1] # linear index of the image pixel
            j = s + s*patch_size + k*patch_size**2 # linear index of the patch center           
            p = y + x*assignment.shape[1]
            i_accumulate[p] = i+di
            j_accumulate[p] = j+dj          
    B = scipy.sparse.coo_matrix((np.ones(i_accumulate.size, dtype=np.bool),
                    (i_accumulate.ravel(),j_accumulate.ravel())),shape=(n,m))
    return B.tocsr()

def biadjacency2transformations(B):
    """ Eq. (6) and Eq. (7) from https://arxiv.org/pdf/1809.02226.pdf"""
    s1 = np.asarray(B.sum(axis=0)) # length m
    s2 = np.asarray(B.sum(axis=1)) # length n
    s1[s1==0] = 1 # preventing division by zero
    s2[s2==0] = 1
    s1 = 1/s1
    s2 = 1/s2
    T1 = scipy.sparse.diags(s1.ravel())*B.transpose()
    T2 = scipy.sparse.diags(s2.ravel())*B
    return (T1, T2)

def labels2labcol(labels, nr_classes):
    """ Unfold labeling image into columns for matrix multiplication."""
    labels = np.ravel(labels)
    labeled = labels>0
    labcol = scipy.sparse.coo_matrix((np.ones(np.sum(labeled), dtype=np.bool),
                    (np.where(labeled)[0], labels[labeled]-1)),
                    shape=(labels.size, nr_classes)).tocsr()
    return labcol

def probcol2probabilities(probcol, image_shape):
    """ Fold columns of probabilities into probability image. """
    p = np.sum(probcol, axis=1)
    nonempty = p>0
    probcol[nonempty] = probcol[nonempty]/(p[nonempty].reshape((-1,1)))
    return probcol.reshape(image_shape + (-1,))

def probcol2labcol(probcol):
    """ Probability to labels using max approach.  """
    p = np.sum(probcol, axis=1)
    nonempty = p>0
    l = np.argmax(probcol[nonempty],axis=1)
    labcol = scipy.sparse.coo_matrix((np.ones(np.sum(nonempty), dtype=np.bool),
                    (np.where(nonempty)[0], l)),
                    shape=probcol.shape).tocsr()
    return labcol
    
def gray_cool(nr_classes):
    """ Colormap as in original InSegt """
    colors = plt.cm.cool(np.linspace(0,1,nr_classes))
    colors = np.r_[np.array([[0.5, 0.5, 0.5, 1]]),colors]
    cmap = matplotlib.colors.ListedColormap(colors)
    return cmap 

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
assignment = image2assignment(image, patch_size, nr_clusters, nr_training_patches)
print(f'Image to assignment: {time.time()-t}')
t = time.time()
B = assignment2biadjacency(assignment, image.shape, patch_size, nr_clusters)
print(f'Assignment to biadjacency: {time.time()-t}')
t = time.time()
T1, T2 = biadjacency2transformations(B)
print(f'Biadjacency to transformations: {time.time()-t}')

# %% Incorporating labelings

# Transforming labels into probabilities.
t = time.time()
labcol = labels2labcol(labels, nr_classes=nr_classes)
probcol = T2*(T1*labcol) # first linear diffusion
probcol = np.asarray(probcol.todense())
print(f'Matrix multiplications: {time.time()-t}')

t = time.time()
labcol = probcol2labcol(probcol) # binarizing labels
probcol = T2*(T1*labcol) # second linear diffusion
probcol = np.asarray(probcol.todense())
print(f'Second round of matrix multiplications: {time.time()-t}')

t = time.time()
probabilities = probcol2probabilities(probcol, image.shape)
print(f'Probcol to probabilities: {time.time()-t}')

#%% Producing a final segmentation

segmentation = probcol2labcol(probcol)*(np.arange(probcol.shape[1])+1) # final segmentation
segmentation = segmentation.reshape(image.shape)

# Preparing for visualization of probabilities as rgb image.
P_rgb = np.zeros(probabilities.shape[0:2]+(3,))
k = min(nr_classes, 3)
P_rgb[:,:,:k] = probabilities[:,:,:k]

# Correctly padding assignment image for visuzlization.
a_im = np.zeros(image.shape)
s = (patch_size-1)//2
a_im[s:-s,s:-s] = assignment

# %% Visualization.

fig, ax = plt.subplots(2, 2, sharex = True, sharey = True)
ax[0,0].imshow(image, cmap='gray')
ax[0,0].set_title('image')
ax[0,1].imshow(a_im)
ax[0,1].set_title('assignment')
ax[1,0].imshow(labels, vmin=0, vmax=nr_classes, cmap = gray_cool(nr_classes))
ax[1,0].set_title('labels')
ax[1,1].imshow(segmentation, vmin=0, vmax=nr_classes, cmap = gray_cool(nr_classes))
ax[1,1].set_title('segmentation')

