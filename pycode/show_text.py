""" Simple image annotator. 
Author: vand@dtu.dk, 2020
"""

import sys 
import PyQt5.QtCore  
import PyQt5.QtWidgets
import PyQt5.QtGui



#class OnScreenTextView(PyQt5.QtWidgets.QGraphicsView):
#    def 
    
    
    
class ShowText(PyQt5.QtWidgets.QWidget):
        
    def __init__(self, filename):
        
        super().__init__() 
    
        # Pixmap layers
        self.imagePix = PyQt5.QtGui.QPixmap(filename)
        
        self.showHelp = True
           
        # Atributes relating to the transformation between widget 
        # coordinate system and image coordinate system
        self.target = PyQt5.QtCore.QRect(0,0,self.width(),self.height()) # part of the target being drawn on
        self.source = PyQt5.QtCore.QRect(0,0,self.imagePix.width(),self.imagePix.height()) # part of the image being drawn
        
        # Playtime
        self.resize(self.imagePix.width(), self.imagePix.height())

        print(f'{self.width()}, {self.height()}')
        # self.helpPix = self.makeHelpPixmap(self.width(),self.height())
        # print(f'{self.helpPix.width()}, {self.helpPix.height()}')
        self.label = PyQt5.QtWidgets.QLabel(self.helpText, self)
        self.label.setStyleSheet("background-color: rgba(190,190,190,200)")
        # self.label.autoFillBackground = False
        self.label.move(10,10)
            
        self.show()

        print("#####################################")    
        print("Starting show image. For help, hit 'H'")    
        
    @staticmethod   
    def makeHelpPixmap(width,height):
    
        
        scene = PyQt5.QtWidgets.QGraphicsScene()
        scene.addRect(0, 0, width, height, PyQt5.QtGui.QPen(PyQt5.QtCore.Qt.NoPen), 
                PyQt5.QtGui.QBrush(PyQt5.QtGui.QColor(191,191,191,255)))
        scene.addSimpleText(ShowText.helpText, PyQt5.QtGui.QFont('SansSerif',10))
        
        graphicsView = PyQt5.QtWidgets.QGraphicsView(scene)
        
        textPix = PyQt5.QtGui.QPixmap(width, height) 
        painter = PyQt5.QtGui.QPainter(textPix)
        
        graphicsView.setBackgroundBrush((PyQt5.QtGui.QColor("gray")))
        graphicsView.setGeometry(0, 0, width, height) 
                
        #rectf = PyQt5.QtCore.QRectF(0, 0, width, height) 
        rect = PyQt5.QtCore.QRect(0, 0, width, height) 
        
        print(f'scene rect {scene.sceneRect()}')
        #graphicsView.render(painter, target=rectf, source=rect)
        graphicsView.render(painter, source=rect)
        
        #graphicsView.render(painter)
        return textPix    
    
    def paintEvent(self, event):
        """ Paint event for displaying the content of the widget."""
        painter_display = PyQt5.QtGui.QPainter(self) # this is painter used for display
        painter_display.setCompositionMode(PyQt5.QtGui.QPainter.CompositionMode_SourceOver)
        painter_display.drawPixmap(self.target, self.imagePix, self.source)
        # if self.showHelp:        
        #     #painter_display.setCompositionMode(PyQt5.QtGui.QPainter.CompositionMode_Overlay)          
        #     painter_display.drawPixmap(self.target, self.helpPix, self.source)
                            
    def resizeEvent(self, event):
        """ Computes padding needed such that aspect ratio of the image is correct. """
        zoomWidth = self.width()/self.source.width()
        zoomHeight = self.height()/self.source.height() 
        if zoomWidth > zoomHeight:
            padding = PyQt5.QtCore.QPoint((int(self.width()-self.source.width()*zoomHeight)/2), 0)
        else:
            padding = PyQt5.QtCore.QPoint(0, int((self.height()-self.source.height()*zoomWidth)/2))    
        self.target = PyQt5.QtCore.QRect(padding, self.rect().bottomRight()-padding)
        
        #self.graphicsView.setGeometry(0.25*self.width(),0.25*self.height(),0.5*self.width(),0.5*self.height())  

            
    def keyPressEvent(self, event):
        # print(f'key {event.key()}, text {event.text()}') 
        
        if event.key()==72: #h
            print(self.helpText)
            self.showHelp = not self.showHelp
            self.label.setText(self.helpText)
            self.label.setStyleSheet("background-color: rgba(190,190,190,200)")
        
            self.update()
        elif event.key()==16777216: # escape
            self.closeEvent(event)
            self.label.clear()
            
        
    def keyReleaseEvent(self, event):
        if event.key()==72: # h
            self.label.setStyleSheet("background-color: rgba(190,190,190,0)")
            self.label.clear()
            
    def closeEvent(self, event):
        print("Bye, I'm closing")
        PyQt5.QtWidgets.QApplication.quit()
        # hint from: https://stackoverflow.com/questions/54045134/pyqt5-gui-cant-be-close-from-spyder
        # should also check: https://github.com/spyder-ide/spyder/wiki/How-to-run-PyQt-applications-within-Spyder
    

    helpText = (
        '\n'
        '   ************* Help for InSegt annotator *************   ' + '\n' +
        '    KEYBORD COMMANDS:' + '\n' +
        "    '1' to '9' changes label (pen color)" + '\n' +
        "    '0' eraser mode" + '\n' +
        "    'uparrow' and 'downarrow' changes pen width" + '\n' +
        "    'W' changes view (annotation, segmentation, both)" + '\n' +
        "    'I' held down temporarily removes shown image" + '\n' +
        "    'Z' held down allows zoom" + '\n' +
        "    'Z' pressed resets zoom" + '\n' +
        "    'S' saves annotation and segmentation" + '\n' +
        "    'H' prints this help" + '\n' +
        '   *****************************************************   ' + '\n'
        ) 

if __name__ == '__main__':
       
    app = PyQt5.QtWidgets.QApplication(sys.argv)
    ex = ShowText('../data/carbon.png')
    sys.exit(app.exec_())  
    
    
    
    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 08:35:46 2020

@author: vand
"""

