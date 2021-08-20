#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Interactive segmentation annotator. 

This module contains the InSegtAnnotator class, which is a subclass of the 
Annotator class from the module annotator. InSegtAnnotator extends Annotator 
with the functionality for interactive segmentation. Segmentation is computed 
from annotations using a generic processing function.  


Use:
    Run from your environmend by passing InSegtAnnotator a grayscale uint8 
    image and a processing function which given labeling returns segmentation. 
    Check also example at the bottom of this file.
    
Author: vand@dtu.dk, 2020
Created on Sun Oct 11 22:42:32 2020

Todo:
    * All from annotator.
    * Showing probability images for probability-based processing function.

GitHub:
   https://github.com/vedranaa/InSegt/tree/master/pycode

"""

import annotator
import numpy as np
import PyQt5.QtCore  
import skimage.io # this is just to get hold of example image
import sys

class InSegtAnnotator(annotator.Annotator):
    
    def __init__(self, image, processing_function):
        '''
        Initializes InSegtAnnotator given an image and a processing function.

        Parameters
        ----------
        image : An image as a 2D array of dtype uint8.
        processing_function : A processing function which given an annotation 
            returns a segmentation. Annotation is given as a 2D array of 
            dtype uint8, where 0 represents unlabeled pixels, and numbers
            1 to C represent labelings for diffeerent classes. (In current 
            impelmentation C<10.) A segmentation is also a 2D array of dytpe
            uint8.
        '''
        
        imagePix = self.grayToPixmap(image)
        self.segmentationPix = PyQt5.QtGui.QPixmap(imagePix.width(), imagePix.height())
        self.segmentationPix.fill(self.color_picker(label=0, opacity=0))
        self.showImage = True

        super().__init__(imagePix.size()) # the first drawing happens when init calls show()  
        
        self.overlays = {0:'both', 1:'annotation', 2:'segmentation'}
        self.annotationOpacity = 0.3
        self.segmentationOpacity = 0.3
        self.imagePix = imagePix
        self.processing_function = processing_function
    
    # METHODS OVERRIDING ANNOTATOR METHODS:                   
    def paintEvent(self, event):
        """ Paint event for displaying the content of the widget."""
        painter_display = PyQt5.QtGui.QPainter(self) # this is painter used for display
        painter_display.setCompositionMode(PyQt5.QtGui.QPainter.CompositionMode_SourceOver)
        if self.showImage: 
            painter_display.drawPixmap(self.target, self.imagePix, self.source)
        if self.overlay != 1: # overlay 0 or 2
            painter_display.drawPixmap(self.target, self.segmentationPix, self.source)
        if self.overlay != 2: # overlay 0 or 1
            painter_display.drawPixmap(self.target, self.annotationPix, self.source)
        if self.showImage:    
            painter_display.drawPixmap(self.target, self.cursorPix, self.source)
      
    def mouseReleaseEvent(self, event):
        """Segmentation is computed on mouse release."""
        super().mouseReleaseEvent(event)
        self.transformLabels()
        self.update()
        
    def keyPressEvent(self, event):
        """Adding events to annotator"""   
        if event.key()==PyQt5.QtCore.Qt.Key_I:
            if self.showImage:          
                self.showImage = False
                self.update()
                self.showInfo('Turned off show image')
        else:
            super().keyPressEvent(event)
    
    def keyReleaseEvent(self, event):
        """Adding events to annotator"""   
        if event.key()==PyQt5.QtCore.Qt.Key_I: # i
            self.showImage = True
            self.update()
            self.showInfo('Turned on show image')
        else:
            super().keyReleaseEvent(event)
    
    def transformLabels(self):
        """Transforming pixmap annotation to pixmap segmentation."""        
        annotations = self.pixmapToArray(self.annotationPix) # numpy RGBA: height x width x 4, values uint8      
        labels = self.rgbaToLabels(annotations) # numpy labels: height x width, values 0 to N uint8    
        segmentation = self.processing_function(labels) # numpy labels: height x width, values 0 to N uint8
        segmentation_rgba = self.labelsToRgba(segmentation, 
                                              self.segmentationOpacity) # numpy RGBA: height x width x 4, values uint8  
        self.segmentationPix = self.rgbaToPixmap(segmentation_rgba)    # final pixmap    
    
    @staticmethod
    def savePixmap(pixmap, filenamebase, gray):
        """Helping function for saving annotation and segmentation pixmaps."""
        pixmap.save(filenamebase + '_pixmap.png', 'png')
        rgba = InSegtAnnotator.pixmapToArray(pixmap) # numpy RGBA: height x width x 4, values uint8      
        skimage.io.imsave(filenamebase + '_rgb.png', rgba[:,:,:3], 
                          check_contrast=False)   
        labels = InSegtAnnotator.rgbaToLabels(rgba) # numpy labels: height x width, values 0 to N uint8    
        skimage.io.imsave(filenamebase + '_index.png', 30*labels, 
                          check_contrast=False) # 30*8 = 240<255     
        alpha = (rgba[:,:,3:].astype(np.float))/255
        overlay = gray[:,:,:3]*(1-alpha) + rgba[:,:,:3]*(alpha)
        skimage.io.imsave(filenamebase + '_overlay.png', 
                          overlay.astype(np.uint8), check_contrast=False)                 
         
    def saveOutcome(self):
        gray = self.pixmapToArray(self.imagePix) # numpy RGBA: height x width x 4, values uint8 
        skimage.io.imsave('gray.png', gray[:,:,:1], check_contrast=False)   
        self.savePixmap(self.annotationPix, 'annotations', gray)
        self.savePixmap(self.segmentationPix, 'segmentations', gray)
        self.showInfo('Saved annotations and segmentations in various data types')        
    
    helpText = (
        '<i>Help for InSegt Annotator</i> <br>' 
        '<b>KEYBOARD COMMANDS:</b> <br>' 
        '&nbsp; &nbsp; <b>1</b> to <b>9</b> changes pen label (L) <br>' 
        '&nbsp; &nbsp; <b>0</b> eraser mode <br>' 
        '&nbsp; &nbsp; <b>&uarr;</b> and <b>&darr;</b> changes pen width (W) <br>' 
        '&nbsp; &nbsp; <b>O</b> changes overlay <br>' 
        '&nbsp; &nbsp; <b>I</b> held down hides image <br>'
        '&nbsp; &nbsp; <b>Z</b> held down enables zoom <br>' 
        '&nbsp; &nbsp; <b>Z</b> pressed resets zoom <br>' 
        '&nbsp; &nbsp; <b>S</b> saves results <br>' 
        '&nbsp; &nbsp; <b>H</b> shows this help <br>' 
        '<b>MOUSE DRAG:</b> <br>' 
        '&nbsp; &nbsp; Draws annotation <br>' 
        '&nbsp; &nbsp; Zooms when zoom enabled')
    
    @classmethod
    def introText(cls, rich = True):
        if rich:
            s = "<i>Starting InSegt Annotator</i> <br> For help, hit <b>H</b>"
        else:
            s = "Starting InSegt Annotator. For help, hit 'H'."
        return s        
     
    # for INSEGT, it is IMPORTANT that background is [0,0,0], otherwise rgbToLabels return wrong labels.
    # I therefore re-define collors, such that possible changes in annotator do not destroy InSegt
    # (and also I use numpy here)
    colors = np.array([
        [0, 0, 0], 
        [255, 0, 0], # label 1
        [0, 191, 0], # label 2
        [0, 0, 255], # etc
        [255, 127, 0],
        [0, 255, 191],
        [127, 0, 255],
        [191, 255, 0],
        [0, 127, 255],
        [255, 64, 191]], dtype=np.uint8)          

    # METHODS TRANSFORMING BETWEEN NUMPY (RGBA AND LABELS) AND QT5 DATA TYPES:
    @classmethod
    def rgbaToLabels(cls,rgba):
        """RGBA image to labels from 0 to N. Uses colors. All numpy."""    
        rgb = rgba.reshape(-1,4)[:,:3] # unfolding and removing alpha channel
        dist = np.sum(abs(rgb.reshape(-1,1,3).astype(np.int16) 
                - cls.colors.reshape(1,-1,3).astype(np.int16)), axis=2) # distances to pre-defined colors
        labels = np.empty((rgb.shape[0],), dtype=np.uint8)
        np.argmin(dist, axis=1, out=labels) # label given by the smallest distances
        labels = labels.reshape(rgba.shape[:2]) # folding back
        return(labels)
    
    @classmethod
    def labelsToRgba(cls, labels, opacity=1):
        """Labels from 0 to N to RGBA. Uses colors. All numpy."""
        rgb = cls.colors[labels,:]
        a = (255*opacity*(labels>0)).astype(np.uint8) # alpha channel
        a.shape = a.shape + (1,)
        rgba = np.concatenate((rgb, a), axis=2)
        return(rgba)
    
    @staticmethod
    def pixmapToArray(qpixmap):
        """Qt pixmap to np array. Assumes an 8-bit RGBA pixmap."""
        qimage = qpixmap.toImage().convertToFormat(PyQt5.QtGui.QImage.Format_RGBA8888)
        buffer = qimage.constBits()
        buffer.setsize(qpixmap.height() * qpixmap.width() * 4) # 4 layers for RGBA
        rgba = np.frombuffer(buffer, np.uint8).reshape((qpixmap.height(), 
                qpixmap.width(), 4))
        return rgba.copy()
    
    @staticmethod
    def rgbaToPixmap(rgba):
        """Np array to Qt pixmap. Assumes an 8-bit RGBA image."""
        rgba = rgba.copy()
        qimage = PyQt5.QtGui.QImage(rgba.data, rgba.shape[1], rgba.shape[0], 
                                    PyQt5.QtGui.QImage.Format_RGBA8888)
        qpixmap = PyQt5.QtGui.QPixmap(qimage)
        return qpixmap
    
    @staticmethod
    def grayToPixmap(gray):
        """Uint8 grayscale image to Qt pixmap via RGBA image."""
        rgba = np.tile(gray,(4,1,1)).transpose(1,2,0)
        rgba[:,:,3] = 255
        qpixmap = InSegtAnnotator.rgbaToPixmap(rgba)
        return qpixmap
    
    
    
if __name__ == '__main__':    
    
    '''
    An example on the use of InSegtAnnotator. An image to be may be a rgb 
    image, but InSegtAnnotator needs to be given its grayscale version. As
    shown below, this does not prevent the processing function to use rgb 
    values of the image.
    '''
    
    # defining processing function
    def processing_function(labels):
        '''
        The simplest processing function for rgb images (computes a mean color for
            each label and assigns pixels to label with color closest to pixel color)
    
        Parameters:
            labels, 2D array with labels as uin8 (0 is background)
        Returns:
            segmentation, array of the same size and type as labels
        
        Author: vand@dtu.dk
    
        '''   
        N = labels.max()
        L = 1 if image.ndim==2 else image.shape[2]
        # mean color for every label
        lable_colors = np.array([np.mean(image[labels==n],0) 
                                      for n in range(N+1)])
        # pixel-to-color distances for all pixels and all labels
        dist = ((image.reshape((-1 ,1, L)) - lable_colors.reshape((1, 
                                        N+1, L)))**2).sum(axis=2)
        empty_labels = np.isnan(dist)
        if np.any(empty_labels): # handling unlabeled parts
            dist[empty_labels] = dist[~empty_labels].max()+1 
        # assigning to min distance
        segmentation = np.empty(labels.shape, dtype=np.uint8)
        np.argmin(dist, axis=1, out=segmentation.ravel())
        return(segmentation)
    
    
    # loading image
    print('Loading image')
    image = skimage.data.astronaut()
    image_gray = image if image.ndim==2 else (255*skimage.color.rgb2gray(image)
                                              ).astype(np.uint8)  
    
    print('Showtime')    
    # showtime
    app = PyQt5.QtWidgets.QApplication(['']) 
    ex = InSegtAnnotator(image_gray, processing_function)
    sys.exit(app.exec_())  
