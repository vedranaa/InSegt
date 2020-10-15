#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 00:08:59 2020

@author: vand
"""

import sys
import insegtannotator
import insegtbasic
import skimage.io

# loading image
print('Loading image')
filename = '../data/glass.png'
image = skimage.io.imread(filename)           

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
app = insegtannotator.PyQt5.QtWidgets.QApplication(sys.argv) 
ex = insegtannotator.InSegtAnnotator(image, processing_function)
sys.exit(app.exec_())  