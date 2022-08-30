#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 18:14:00 2021

@author: abda
"""

import insegtannotator
import numpy as np
import sys
import PyQt5.QtCore  
import PyQt5.QtWidgets 
import PyQt5.QtGui
import skimage.io 

class InSegtProbAnnotator(insegtannotator.InSegtAnnotator):
    
    def __init__(self, image, processing_function):
        self.showProbabilities = 0
        super().__init__(image, processing_function)  
        self.probabilities = np.empty((0,self.imagePix.height(),self.imagePix.width()))
        self.probabilityPix = PyQt5.QtGui.QPixmap(self.imagePix.width(), self.imagePix.height())
        self.probabilityPix.fill(self.color_picker(label=0, opacity=0))

    
    def transformLabels(self):
        """Transforming pixmap annotation to pixmap segmentation."""        
        annotations = self.pixmapToArray(self.annotationPix) # numpy RGBA: height x width x 4, values uint8      
        labels = self.rgbaToLabels(annotations) # numpy labels: height x width, values 0 to N uint8    
        segmentation, probabilities = self.processing_function(labels) # numpy labels: height x width, values 0 to N uint8
        segmentation_rgba = self.labelsToRgba(segmentation, 
                                              self.segmentationOpacity) # numpy RGBA: height x width x 4, values uint8  
        self.segmentationPix = self.rgbaToPixmap(segmentation_rgba)    # final pixmap    
        self.probabilities = probabilities
        if(self.showProbabilities>self.probabilities.shape[0]):
            self.showProbabilities = 0
            self.showProbabilityInfo()
            

    def keyPressEvent(self, event):
        """Adding events to annotator"""   
        if event.key()==PyQt5.QtCore.Qt.Key_P: 
            self.showProbabilities = (self.showProbabilities+1)%(self.probabilities.shape[0]+1)
            self.updateProbabilityPix()
            self.showProbabilityInfo()
        else:
            super().keyPressEvent(event)


    def mouseReleaseEvent(self, event):
        """Segmentation and probabilities are computed on mouse release."""
        super().mouseReleaseEvent(event)
        self.updateProbabilityPix()
        
        
    def showProbabilityInfo(self):
        if(self.showProbabilities==0):
            self.showInfo('Not showing probabilities')
        else:
            self.showInfo(f'Showing probability for label {self.showProbabilities}')


    def paintEvent(self, event):
        """ Paint event adds displaying probabilities."""
        if self.showProbabilities>0:
            painter_display = PyQt5.QtGui.QPainter(self) # this is painter used for display
            painter_display.setCompositionMode(PyQt5.QtGui.QPainter.CompositionMode_SourceOver)
            if self.showImage: 
                painter_display.drawPixmap(self.target, self.imagePix, self.source)
            painter_display.drawPixmap(self.target, self.probabilityPix, self.source)
            if self.showImage:    
                painter_display.drawPixmap(self.target, self.cursorPix, self.source)
        else:
            super().paintEvent(event)
            
    
    def updateProbabilityPix(self):
        if(self.showProbabilities>0):
            rgba = self.probabilityToRgba(self.probabilities[self.showProbabilities-1])
            self.probabilityPix = self.rgbaToPixmap(rgba)
            
    
    @staticmethod
    def probabilityToRgba(probabilityLayer):
        mask = (probabilityLayer > 0.5).astype(np.float)
        probabilityColor = np.asarray([2*(1-mask)*probabilityLayer + mask, 
                                       1-2*np.abs(probabilityLayer-0.5), 
                                       mask*(1-2*probabilityLayer) + 1,
                                       np.ones(mask.shape)*0.5]).transpose(1,2,0)
        return (255*probabilityColor).astype(np.uint8)
    
            
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
        # np.argmin(dist, axis=1, out=segmentation.ravel())
        probabilities = np.exp(-dist**2/(10e6))
        sumProbabilities = np.sum(probabilities,axis=1)
        probabilities /= sumProbabilities.reshape(sumProbabilities.shape + (1,))
        probabilities = probabilities.reshape(labels.shape+(-1,)).transpose((2,0,1))
        segmentation = np.argmax(probabilities, axis = 0)
        return(segmentation, probabilities)
    
    
    # loading image
    print('Loading image')
    image = skimage.data.astronaut()
    image_gray = image if image.ndim==2 else (255*skimage.color.rgb2gray(image)
                                              ).astype(np.uint8)  
    
    print('Showtime')    
    # showtime
    app = PyQt5.QtWidgets.QApplication(['']) 
    ex = InSegtProbAnnotator(image_gray, processing_function)
    sys.exit(app.exec_())  




