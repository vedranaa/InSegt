#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Simple image annotator. 

This module contains the Annotator class. Annotator is a widget that allows the 
user to place annotations on top of an image. All interaction is using mouse 
clicks, draws, and keyboard input. Help is accessed by pressing 'H'. 

Use:
    Run fra command-line with
        $ python annotator image_filename
    Or run from your environmend by passing it an rgba uint8 image, as in the 
    example at the bottom of this file.
    
Author: vand@dtu.dk, 2020
Created on Sat Jun 20 12:43:29 2020

Todo:
    * Save output in a folder, and/or make a dialog when saving.

GitHub:
   https://github.com/vedranaa/InSegt/tree/master/pycode

"""

import sys 
import PyQt5.QtCore  
import PyQt5.QtWidgets 
import PyQt5.QtGui
import skimage.data # this is just to get hold of example image
import numpy as np # this is just to transform example image to rgba
  
class Annotator(PyQt5.QtWidgets.QWidget):
    
    def __init__(self, size=None):
        '''
        Initializes an Annotator without the image.

        Parameters
        ----------
        size : two-element tupple for the size of the annotator.

        '''
        
        super().__init__() 
        
        if size is None:
            size = PyQt5.QtCore.QSize(256,256)
        elif type(size) is tuple:
            size = PyQt5.QtCore.QSize(size[0],size[1])
        self.annotationsFilename = 'empty.png'
            
        # Pixmap layers
        self.imagePix = PyQt5.QtGui.QPixmap(size.width(), size.height()) 
        self.imagePix.fill(self.color_picker(label=0, opacity=0))
        self.annotationPix = PyQt5.QtGui.QPixmap(self.imagePix.width(), 
                                                 self.imagePix.height())
        self.annotationPix.fill(self.color_picker(label=0, opacity=0))
        self.cursorPix = PyQt5.QtGui.QPixmap(self.imagePix.width(), 
                                             self.imagePix.height())
        self.cursorPix.fill(self.color_picker(label=0, opacity=0))    
        
        # Atributes for drawing
        self.label = 1
        self.penWidth = 9     
        self.lastDrawPoint = PyQt5.QtCore.QPoint()
        
        # Atributes for displaying
        self.overlay = 0
        self.overlays = {0:'both', 1:'annotation', 2:'image'}
        self.annotationOpacity = 0.5
        self.cursorOpacity = 0.5
        self.zoomOpacity = 0.5
        self.setTitle()
        self.setCursor(PyQt5.QtGui.QCursor(PyQt5.QtCore.Qt.CrossCursor))
        self.lastCursorPoint = PyQt5.QtCore.QPoint()
        self.setMouseTracking(True)
        
        # Atributes relating to the transformation between widget 
        # coordinate system and image coordinate system
        self.zoomFactor = 1 # accounts for resizing of the widget and for zooming in the part of the image
        self.padding = PyQt5.QtCore.QPoint(0, 0) # padding when aspect ratio of image and widget does not match
        self.target = PyQt5.QtCore.QRect(0, 0, self.width(),self.height()) # part of the target being drawn on
        self.source = PyQt5.QtCore.QRect(0, 0, 
                self.imagePix.width(), self.imagePix.height()) # part of the image being drawn
        self.offset = PyQt5.QtCore.QPoint(0, 0) # offset between image center and area of interest center
        
        # Flags needed to keep track of different states
        self.zPressed = False # when z is pressed zooming can start
        self.activelyZooming = False 
        self.activelyDrawing = False
        self.newZoomValues = None
        
        # Label for displaying text overlay
        self.textField = PyQt5.QtWidgets.QLabel(self)
        self.textField.setStyleSheet("background-color: rgba(191,191,191,191)")
        self.textField.setTextFormat(PyQt5.QtCore.Qt.RichText)
        self.textField.resize(0,0)
        self.textField.move(10,10)     
        self.hPressed = False
        self.textField.setAttribute(PyQt5.QtCore.Qt.WA_TransparentForMouseEvents)
        #self.textField.setAttribute(PyQt5.QtCore.Qt.WA_TranslucentBackground) # considered making bacground translucent      
        
        # Timer for displaying text overlay
        self.timer = PyQt5.QtCore.QTimer()
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.hideText)

        # Playtime
        initial_zoom = min(2000/max(self.imagePix.width(), 
                4*self.imagePix.height()/3),1) # downsize if larger than (2000,1500)
        self.resize(initial_zoom*self.imagePix.width(), 
                    initial_zoom*self.imagePix.height())
        self.show()
        
        self.showInfo(self.introText(),5000)
        print(self.introText(False))
    
    @classmethod
    def fromFilename(cls, filename):
        '''
        Initializes an Annotator with an image loaded from a file.

        Parameters
        ----------
        filename : filename of an image in usuall formats (jpg, png, ...).

        '''
        imagePix = PyQt5.QtGui.QPixmap(filename)
        annotator = Annotator(imagePix.size())
        annotator.imagePix = imagePix
        annotator.annotationsFilename = filename[:-4 ]+ '_annotations.png' 
        return annotator
    
    @classmethod  
    def fromRgba(cls, rgba):
        '''
        Initializes an Annotator with an image given as an rgba array.

        Parameters
        ----------
        rgba : (..., 4) array with dtype uint8.

        '''
        rgba = rgba.copy() # check whether needed
        qimage = PyQt5.QtGui.QImage(rgba.data, rgba.shape[1], rgba.shape[0], 
                                    PyQt5.QtGui.QImage.Format_RGBA8888)
        imagePix = PyQt5.QtGui.QPixmap(qimage)
        annotator = Annotator(imagePix.size())
        annotator.imagePix = imagePix
        annotator.annotationsFilename = 'from_rgba_annotations.png'
        return annotator    
    
    @classmethod  
    def fromGrayscale(cls, gray):
        '''
        Initializes an Annotator with an image given as an grayscale array.

        Parameters
        ----------
        grat : 2D array with dtype uint8.

        '''
        gray = gray.copy() # check whether needed
        
        bytesPerLine = gray.nbytes//gray.shape[0]
        qimage = PyQt5.QtGui.QImage(gray.data, gray.shape[1], gray.shape[0],
                                    bytesPerLine,
                                    PyQt5.QtGui.QImage.Format_Grayscale8)
        imagePix = PyQt5.QtGui.QPixmap(qimage)
        annotator = Annotator(imagePix.size())
        annotator.imagePix = imagePix
        annotator.annotationsFilename = 'from_grayscale_annotations.png'
        return annotator    
     
    helpText = (
        '<i>Help for annotator</i> <br>' 
        '<b>KEYBOARD COMMANDS:</b> <br>' 
        '&nbsp; &nbsp; <b>1</b> to <b>9</b> changes pen label (L) <br>' 
        '&nbsp; &nbsp; <b>0</b> eraser mode <br>' 
        '&nbsp; &nbsp; <b>&uarr;</b> and <b>&darr;</b> changes pen width (W) <br>' 
        '&nbsp; &nbsp; <b>O</b> changes overlay <br>' 
        '&nbsp; &nbsp; <b>Z</b> held down enables zoom <br>' 
        '&nbsp; &nbsp; <b>Z</b> pressed resets zoom <br>' 
        '&nbsp; &nbsp; <b>S</b> saves annotation <br>' 
        '&nbsp; &nbsp; <b>H</b> shows this help <br>' 
        '<b>MOUSE DRAG:</b> <br>' 
        '&nbsp; &nbsp; Draws annotation <br>' 
        '&nbsp; &nbsp; Zooms when zoom enabled')
    
    @classmethod
    def introText(cls, rich = True):
        if rich:
            s = '<i>Starting annotator</i> <br> For help, hit <b>H</b>'
            #'<hr> ANNOTATOR <br> Copyright (C) 2020 <br> Vedrana A. Dahl'
        else:
            s = "Starting annotator. For help, hit 'H'."
        return s
        
    def showHelp(self):
        self.timer.stop()
        self.showText(self.helpText)
    
    def showInfo(self, text, time=1000):
        if not self.hPressed:
            self.timer.start(time)
            self.showText(text)
    
    def showText(self, text):
        self.textField.setText(text)
        #self.textField.resize(self.textField.fontMetrics().size(PyQt5.QtCore.Qt.TextExpandTabs, text))
        self.textField.adjustSize()
        self.update()
          
    def hideText(self):
        self.textField.resize(0,0)
        self.update()
        
    def setTitle(self):
        self.setWindowTitle(f'L:{self.label}, W:{self.penWidth}, '+
                            f'O:{self.overlays[self.overlay]}')
            
    def makePainter(self, pixmap, color):
        """" Returns scribble painter operating on a given pixmap. """
        painter_scribble = PyQt5.QtGui.QPainter(pixmap)       
        painter_scribble.setPen(PyQt5.QtGui.QPen(color, 
                    self.penWidth*self.zoomFactor, PyQt5.QtCore.Qt.SolidLine, 
                    PyQt5.QtCore.Qt.RoundCap, PyQt5.QtCore.Qt.RoundJoin))
        painter_scribble.translate(-self.offset)
        painter_scribble.translate(-0.25,-0.25) # a compromise between odd and even pen width
        painter_scribble.scale(1/self.zoomFactor, 1/self.zoomFactor)
        painter_scribble.translate(-self.padding)        
        painter_scribble.setCompositionMode(
                    PyQt5.QtGui.QPainter.CompositionMode_Source)
        return painter_scribble

    def paintEvent(self, event):
        """ Paint event for displaying the content of the widget."""
        painter_display = PyQt5.QtGui.QPainter(self) # this is painter used for display
        painter_display.setCompositionMode(
                    PyQt5.QtGui.QPainter.CompositionMode_SourceOver)
        if self.overlay != 1: # overlay 0 or 2
            painter_display.drawPixmap(self.target, self.imagePix, self.source)
        if self.overlay != 2: # overlay 0 or 1
            painter_display.drawPixmap(self.target, self.annotationPix, 
                                       self.source)
        painter_display.drawPixmap(self.target, self.cursorPix, self.source)
        
    def drawCursorPoint(self, point):
        """Called when cursorPix needs update due to pen change or movement"""
        self.cursorPix.fill(self.color_picker(label=0, opacity=0)) # transparent
        painter_scribble = self.makePainter(self.cursorPix, 
                    self.color_picker(self.label, self.cursorOpacity)) # the painter used for cursor
        painter_scribble.drawPoint(point)   
    
    def mousePressEvent(self, event):
        if event.button() == PyQt5.QtCore.Qt.LeftButton: 
            if self.zPressed: # initiate zooming and not drawing
                self.cursorPix.fill(self.color_picker(label=0, opacity=0)) # clear (fill with transparent)
                self.lastCursorPoint = event.pos()
                self.activelyZooming = True
                self.newZoomValues = 0 # for distinction between reset and cancel
            else: # initiate drawing
                painter_scribble = self.makePainter(self.annotationPix, 
                        self.color_picker(self.label, 
                            (self.label>0)*self.annotationOpacity)) # the painter used for drawing        
                painter_scribble.drawPoint(event.pos())
                self.lastDrawPoint = event.pos()   
                self.activelyDrawing = True
            self.update()
    
    def mouseMoveEvent(self, event):
        if self.activelyZooming: 
            self.cursorPix.fill(self.color_picker(label=0, opacity=0)) # clear (fill with transparent)
            painter_scribble = self.makePainter(self.cursorPix,
                    self.color_picker(0, self.zoomOpacity))          
            x = min(self.lastCursorPoint.x(), event.x())
            y = min(self.lastCursorPoint.y(), event.y())
            w = abs(self.lastCursorPoint.x() - event.x())
            h = abs(self.lastCursorPoint.y() - event.y())      
            painter_scribble.fillRect(x, y, w, h, 
                            self.color_picker(0, self.zoomOpacity))
        else:     
            if self.activelyDrawing: 
                painter_scribble = self.makePainter(self.annotationPix, 
                        self.color_picker(self.label, 
                                (self.label>0)*self.annotationOpacity)) # the painter used for drawing        
                painter_scribble.drawLine(self.lastDrawPoint, event.pos())
                self.lastDrawPoint = event.pos()
            if self.zPressed:
                if  self.newZoomValues is None:
                    self.newZoomValues = 0 # for distinction between reset and cancel
            else: # just moving around
                self.drawCursorPoint(event.pos())
            self.lastCursorPoint = event.pos()      
        self.update()
    
    def mouseReleaseEvent(self, event):  
        if self.activelyZooming:
            x = min(self.lastCursorPoint.x(), event.x())
            y = min(self.lastCursorPoint.y(), event.y())
            w = abs(self.lastCursorPoint.x() - event.x())
            h = abs(self.lastCursorPoint.y() - event.y())
            if w>0 and h>0:
                self.newZoomValues = PyQt5.QtCore.QRect(x,y,w,h)
            self.lastCursorPoint = event.pos()    
            self.activelyZooming = False
            if not self.zPressed:
                self.executeZoom()
        elif self.activelyDrawing:
            self.activelyDrawing = False
    
    def leaveEvent(self, event):
        """Removes curser when mouse leaves widget. """
        if not (self.activelyZooming or self.zPressed):
            self.cursorPix.fill(self.color_picker(label=0, opacity=0)) # clear (fill with transparent)
            self.update()
            
    def resizeEvent(self, event):
        """ Triggered by resizing of the widget window. """
        self.adjustTarget()
                
    def adjustTarget(self):
        """ Computes padding needed such that aspect ratio of the image is correct. """
        self.cursorPix.fill(self.color_picker(label=0, opacity=0)) # clear (fill with transparent)
        self.update()   

        zoomWidth = self.width()/self.source.width()
        zoomHeight = self.height()/self.source.height() 
        
        # depending on aspect ratios, either pad up and down, or left and rigth
        if zoomWidth > zoomHeight:
            self.zoomFactor = zoomHeight
            self.padding = PyQt5.QtCore.QPoint(int((self.width() 
                            - self.source.width()*self.zoomFactor)/2), 0)
        else:
            self.zoomFactor = zoomWidth
            self.padding = PyQt5.QtCore.QPoint(0, int((self.height()
                            - self.source.height()*self.zoomFactor)/2))
            
        self.target = PyQt5.QtCore.QRect(self.padding, 
                            self.rect().bottomRight() - self.padding)
                   
    def executeZoom(self):
        """ Zooms to rectangle given by newZoomValues. """
        self.newZoomValues.translate(-self.padding)
        self.source = PyQt5.QtCore.QRect(self.newZoomValues.topLeft()/self.zoomFactor,
                self.newZoomValues.size()/self.zoomFactor)
        self.source.translate(-self.offset)
        self.source = self.source.intersected(self.imagePix.rect()) 
        self.showInfo('Zooming to ' + self.formatQRect(self.source))     
        self.offset = self.imagePix.rect().topLeft() - self.source.topLeft()
        self.adjustTarget()
        self.newZoomValues = None
    
    def resetZoom(self):
        """ Back to original zoom """
        self.source = PyQt5.QtCore.QRect(0,0,self.imagePix.width(), 
                                         self.imagePix.height())
        self.showInfo('Reseting zoom to ' + self.formatQRect(self.source))        
        self.offset = PyQt5.QtCore.QPoint(0,0)
        self.adjustTarget()        
        self.newZoomValues = None
            
    def keyPressEvent(self, event):
        if 47<event.key()<58: #numbers 0 (48) to 9 (57)
            self.label = event.key()-48
            self.drawCursorPoint(self.lastCursorPoint)
            self.update()
            self.showInfo(f'Changed pen label to {self.label}')
        elif event.key()==PyQt5.QtCore.Qt.Key_Up: # uparrow          
            self.penWidth = min(self.penWidth+1,50) 
            self.drawCursorPoint(self.lastCursorPoint)
            self.update()
            self.showInfo(f'Changed pen width to {self.penWidth}')
        elif event.key()==PyQt5.QtCore.Qt.Key_Down: # downarrow
            self.penWidth = max(self.penWidth-1,1)
            self.drawCursorPoint(self.lastCursorPoint)
            self.update()
            self.showInfo(f'Changed pen widht to {self.penWidth}')
        elif event.key()==PyQt5.QtCore.Qt.Key_S: # s
            self.saveOutcome()
        elif event.key()==PyQt5.QtCore.Qt.Key_O: # o
            self.overlay = (self.overlay+1)%len(self.overlays)
            self.update()
            self.showInfo(f'Changed overlay to {self.overlays[self.overlay]}')
        elif event.key()==PyQt5.QtCore.Qt.Key_Z: # z
            if not self.zPressed:
                self.showInfo('Zooming enabled')
                self.zPressed = True
                self.cursorPix.fill(self.color_picker(label=0, opacity=0)) # clear (fill with transparent)
                self.update()
        elif event.key()==PyQt5.QtCore.Qt.Key_H: # h        
            if not self.hPressed:
                self.hPressed = True
                self.showHelp()
        # elif event.key()==PyQt5.QtCore.Qt.Key_Escape: # escape
        #     self.closeEvent()
        self.setTitle()
        
    def keyReleaseEvent(self, event):
        if event.key()==PyQt5.QtCore.Qt.Key_Z: # z
            if not self.activelyZooming:
                self.drawCursorPoint(self.lastCursorPoint)
                if self.newZoomValues is None:
                    self.resetZoom()
                elif self.newZoomValues==0:
                    self.showInfo('Zooming canceled')
                    self.newZoomValues = None
                else:
                    self.executeZoom()                       
                self.update()
            self.zPressed = False
        elif event.key()==PyQt5.QtCore.Qt.Key_H: # h
            self.hideText()
            self.hPressed = False
            
    # def closeEvent(self, event):
    #     self.showInfo("Bye, I'm closing")
    #     PyQt5.QtWidgets.QApplication.quit()
    #     # hint from: https://stackoverflow.com/questions/54045134/pyqt5-gui-cant-be-close-from-spyder
    #     # should also check: https://github.com/spyder-ide/spyder/wiki/How-to-run-PyQt-applications-within-Spyder
   
    def saveOutcome(self):
        self.annotationPix.save(self.annotationsFilename, 'png')
        self.showInfo(f'Saved annotations as {self.annotationsFilename}')
        
    # colors associated with different labels
    colors = [
        [0, 0, 0], # background, transparency is always drawn with black
        [255, 0, 0], # label 1
        [0, 191, 0], # label 2
        [0, 0, 255], # etc
        [255, 127, 0],
        [0, 255, 191],
        [127, 0, 255],
        [191, 255, 0],
        [0, 127, 255],
        [255, 64, 191]] 

    @classmethod
    def color_picker(cls, label, opacity):
        """ Pen colors given for a label number. """
        opacity_value = int(opacity*255)
        color = PyQt5.QtGui.QColor(cls.colors[label][0], cls.colors[label][1], 
                cls.colors[label][2], opacity_value)
        return(color)
    
    @staticmethod
    def formatQRect(rect):
        coords =  rect.getCoords()
        s = f'({coords[0]},{coords[1]})--({coords[2]},{coords[3]})'
        return(s)     
      
if __name__ == '__main__':
    
    '''
    Annotator may be used from command-line. If no image is given, a test
    image from skimage.data is used.
    '''
       
    app = PyQt5.QtWidgets.QApplication([])
    
    if len(sys.argv)>1:
         filename = sys.argv[1]
         ex = Annotator.fromFilename(filename)
    else:
         image = skimage.data.camera()
         ex = Annotator.fromGrayscale(image)
    
    # ex.show() is probably better placed here than in init
    app.exec()
    
    #app.quit(), not needed? exec starts the loop which quits when the last top widget is closed  
    #sys.exit(), not needed?  
    
    
    
    