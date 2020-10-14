#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 22:42:32 2020

@author: vand@dtu.dk, 2020
"""

import sys 
import annotator
import numpy as np
import PyQt5.QtCore  
import insegtbasic
import skimage.io

class InSegtAnnotator(annotator.Annotator):
    
    def __init__(self, image, processing_function):
        
        imagePix = self.grayToPixmap(image)
        self.segmentationPix = PyQt5.QtGui.QPixmap(imagePix.width(), imagePix.height())
        self.segmentationPix.fill(self.color_picker(label=0, opacity=0))
        self.showImage = True

        super().__init__(imagePix.size()) # the first drawing happens when init calls show()  
        
        self.views = {0:'both', 1:'annotation', 2:'segmentation'}
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
        if self.view != 1: # view 0 or 2
            painter_display.drawPixmap(self.target, self.segmentationPix, self.source)
        if self.view != 2: # view 0 or 1
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
        if event.key()==73: # i
            if self.showImage:          
                self.showImage = False
                self.update()
                print(f'   Turined off show image')
        else:
            super().keyPressEvent(event)
    
    def keyReleaseEvent(self, event):
        """Adding events to annotator"""   
        if event.key()==73: # i
            self.showImage = True
            self.update()
            print(f'   Turined on show image')
        else:
            super().keyReleaseEvent(event)
    
    def transformLabels(self):
        """Transforming pixmap annotation to pixmap segmentation via numpy and processing function"""        
        annotations = self.pixmapToArray(self.annotationPix) # numpy RGBA: height x width x 4, values uint8      
        labels = self.rgbaToLabels(annotations) # numpy labels: height x width, values 0 to N uint8    
        segmentation = self.processing_function(labels) # numpy labels: height x width, values 0 to N uint8
        segmentation_rgba = self.labelsToRgba(segmentation, self.segmentationOpacity) # numpy RGBA: height x width x 4, values uint8  
        self.segmentationPix = self.arrayToPixmap(segmentation_rgba)    # final pixmap    
    
    def saveOutcome(self):
        self.annotationPix.save('annotations_pixmap.png', 'png')
        self.segmentationPix.save('segmentations_pixmap.png', 'png')    
        annotations = self.pixmapToArray(self.annotationPix) # numpy RGBA: height x width x 4, values uint8      
        skimage.io.imsave('annotations_rgb.png', annotations[:,:,:3], check_contrast=False)                 
        segmentations = self.pixmapToArray(self.segmentationPix) # numpy RGBA: height x width x 4, values uint8 
        skimage.io.imsave('segmentations_rgb.png', segmentations[:,:,:3], check_contrast=False)       
        labels = self.rgbaToLabels(annotations) # numpy labels: height x width, values 0 to N uint8    
        labels_segmentation = self.rgbaToLabels(segmentations) # numpy labels: height x width, values 0 to N uint8            
        skimage.io.imsave('annotations_gray.png', 30*labels, check_contrast=False) # 30*8 = 240<255     
        skimage.io.imsave('segmentations_gray.png', 30*labels_segmentation, check_contrast=False)              
        print('   Saved annotations and segmentations in various data types')
    
    @classmethod
    def printHelp(cls): 
        """Overriding help"""
        
        print('******* Help for annotator *******')
        print('KEYBORD COMMANDS:')
        print("   '1' to '9' changes label (pen color)")
        print("   '0' eraser mode")
        print("   'uparrow' and 'downarrow' changes pen width")
        print("   'W' changes view (annotation, segmentation or both)")
        print("   'I' held down temporarily removes shown image")
        print("   'Z' held down allows zoom")
        print("   'Z' pressed resets zoom")
        print("   'S' saves annotation and segmentation")
        print("   'H' prints this help")
        print('**********************************')  
     
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
        img = np.frombuffer(buffer, np.uint8).reshape((qpixmap.height(), 
                qpixmap.width(), 4))
        return img.copy()
    
    @staticmethod
    def arrayToPixmap(img):
        """Np array to Qt pixmap. Assumes an 8-bit RGBA image."""
        img = img.copy()
        qimage = PyQt5.QtGui.QImage(img.data, img.shape[1], img.shape[0], 
                                    PyQt5.QtGui.QImage.Format_RGBA8888)
        qpixmap = PyQt5.QtGui.QPixmap(qimage)
        return qpixmap
    
    @staticmethod
    def grayToPixmap(gray):
        """Uint8 grayscale image to Qt pixmap via RGBA image."""
        gray = np.tile(gray,(4,1,1)).transpose(1,2,0)
        gray[:,:,3] = 255
        qpixmap = InSegtAnnotator.arrayToPixmap(gray)
        return qpixmap
        

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
app = PyQt5.QtWidgets.QApplication(sys.argv) 
ex = InSegtAnnotator(image, processing_function)
sys.exit(app.exec_())  