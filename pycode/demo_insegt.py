#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo showing how insegtbasic and insegtannotator may be used together for
interactive segmentation. As examples we use CT image of glass fibres
(accessible from https://github.com/vedranaa/InSegt/tree/master/data), and
images from skimage.data.

Created on Fri Oct 16 00:08:59 2020
@author: vand
"""

import sys
import insegtannotator
import insegtbasic
import skimage.io
import skimage.data

#%% EXAMPLE 1: glass fibres

## loading image
print('Loading image')
filename = '../data/glass.png'
image = skimage.io.imread(filename) 
image_gray = image         

#%% EXAMPLE 2:  immunohistochemistry

## loading image
# print('Loading image')
# image = skimage.data.immunohistochemistry()
# image_gray = (255*skimage.color.rgb2gray(image)).astype('uint8')
# skimage.io.imshow(image)

#%% COMMON PART

# defining processing function
print('Defining processing function')
patch_size = 9
nr_training_patches = 10000
nr_clusters = 100
T1, T2 = insegtbasic.patch_clustering(image, patch_size, 
        nr_training_patches, nr_clusters)
def processing_function(labels):
    return insegtbasic.two_binarized(labels, T1, T2)

print('Showtime')    
# showtime
app = insegtannotator.PyQt5.QtWidgets.QApplication([]) 
ex = insegtannotator.InSegtAnnotator(image_gray, processing_function)
app.exec()
sys.exit()  