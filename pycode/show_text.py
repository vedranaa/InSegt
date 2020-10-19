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
           
        # Atributes relating to the transformation between widget 
        # coordinate system and image coordinate system
        self.target = PyQt5.QtCore.QRect(0,0,self.width(),self.height()) # part of the target being drawn on
        self.source = PyQt5.QtCore.QRect(0,0,self.imagePix.width(),self.imagePix.height()) # part of the image being drawn
        
        # Playtime
        self.resize(self.imagePix.width(), self.imagePix.height())

        # Text overlay
        self.label = PyQt5.QtWidgets.QLabel(self)   
        self.label.setStyleSheet("background-color: rgba(190,190,190,200)")
        self.label.resize(0,0)
        self.label.move(10,10)     
        self.showingHelp = False
        
        self.timer = PyQt5.QtCore.QTimer()
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.stopShow)
 
        
        self.show()

        print("#####################################")    
        print("Starting show image. For help, hit 'H'")    
        
    
        
    
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
    
    def showHelp(self):
        self.label.setText(self.helpText)
        self.label.adjustSize()
        #self.label.resize(self.label.fontMetrics().size(PyQt5.QtCore.Qt.TextExpandTabs, self.helpText))
        self.update()
          
    def hideHelp(self):
        self.label.resize(0,0)
        self.update()
        
    def startShow(self):
        #self.timer.stop()
        self.timer.setInterval(2000)
        self.timer.start()
        print('Starting again')
        
        
    def stopShow(self):
        print('...Stopping now!')
       
            
    def keyPressEvent(self, event):
        # print(f'key {event.key()}, text {event.text()}') 
        
        self.startShow()
        
        
        
        if event.key()==72: #h
            if not self.showingHelp:
                self.showingHelp = True
                self.showHelp()
            
        elif event.key()==16777216: # escape
            self.closeEvent(event)
            self.label.clear()
            
        
    def keyReleaseEvent(self, event):
        if event.key()==72: # h
            if self.showingHelp:
                self.showingHelp = False
                self.hideHelp()
            
    def closeEvent(self, event):
        print("Bye, I'm closing")
        PyQt5.QtWidgets.QApplication.quit()
        # hint from: https://stackoverflow.com/questions/54045134/pyqt5-gui-cant-be-close-from-spyder
        # should also check: https://github.com/spyder-ide/spyder/wiki/How-to-run-PyQt-applications-within-Spyder
    

    helpText = (
        '\n' +
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

