""" Simple image annotator. 
Author: vand@dtu.dk, 2020
"""

import sys 
import PyQt5.QtCore  
import PyQt5.QtWidgets 
import PyQt5.QtGui
  
class Annotator(PyQt5.QtWidgets.QWidget):
    
    def __init__(self, size=None):
        
        super().__init__() 
        
        if size is None:
            size = PyQt5.QtCore.QSize(256,256)
        elif type(size) is tuple:
            size = PyQt5.QtCore.QSize(size[0],size[1])
        
        # Pixmap layers
        self.imagePix = PyQt5.QtGui.QPixmap(size.width(), size.height()) 
        self.imagePix.fill(Annotator.color_picker(label=0, opacity=0))
        self.annotationPix = PyQt5.QtGui.QPixmap(self.imagePix.width(), self.imagePix.height())
        self.annotationPix.fill(self.color_picker(label=0, opacity=0))
        self.cursorPix = PyQt5.QtGui.QPixmap(self.imagePix.width(), self.imagePix.height())
        self.cursorPix.fill(self.color_picker(label=0, opacity=0))    
        
        # Atributes for drawing
        self.label = 1
        self.penWidth = 9     
        self.lastDrawPoint = PyQt5.QtCore.QPoint()
        
        # Atributes for displaying
        self.view = 0
        self.opacity = 0.5
        self.zoomOpacity = 0.5
        self.setTitle()
        self.setCursor(PyQt5.QtGui.QCursor(PyQt5.QtCore.Qt.CrossCursor))
        self.lastCursorPoint = PyQt5.QtCore.QPoint()
        self.setMouseTracking(True)
        
        # Atributes relating to the transformation between widget 
        # coordinate system and image coordinate system
        self.zoomFactor = 1 # accounts for resizing of the widget and for zooming in the part of the image
        self.padding = PyQt5.QtCore.QPoint(0,0) # padding when aspect ratio of image and widget does not match
        self.target = PyQt5.QtCore.QRect(0,0,self.width(),self.height()) # part of the target being drawn on
        self.source = PyQt5.QtCore.QRect(0,0,self.imagePix.width(),self.imagePix.height()) # part of the image being drawn
        self.offset = PyQt5.QtCore.QPoint(0,0) # offset between image center and area of interest center
        
        # Flags needed to keep track of different states
        self.zPressed = False # when z is pressed zooming can start
        self.activelyZooming = False 
        self.activelyDrawing = False
        self.newZoomValues = None

        # Playtime
        initial_zoom = min(2000/max(self.imagePix.width(), 4*self.imagePix.height()/3),1) # downsize if larger than (2000,1500)
        self.resize(initial_zoom*self.imagePix.width(), initial_zoom*self.imagePix.height())
        self.show()
        print("Starting annotator. For help, hit 'H'")    
    
    @classmethod
    def fromFilename(cls, filename):
        imagePix = PyQt5.QtGui.QPixmap(filename)
        annotator = Annotator(imagePix.size())
        annotator.imagePix = imagePix
        return annotator
    
    def setTitle(self):
        views = {0:'both', 1:'annotation', 2:'image'}
        self.setWindowTitle(f'L:{self.label}, P:{self.penWidth}, W:{views[self.view]}')
    
    def annotationOpacity(self):
        if self.label == 0:
            return 0
        else:
            return self.opacity
        
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
        painter_scribble.setCompositionMode(PyQt5.QtGui.QPainter.CompositionMode_Source)
        return painter_scribble

    def paintEvent(self, event):
        """ Paint event for displaying the content of the widget."""
        painter_display = PyQt5.QtGui.QPainter(self) # this is painter used for display
        painter_display.setCompositionMode(PyQt5.QtGui.QPainter.CompositionMode_SourceOver)
        if self.view != 1: # view 0 or 2
            painter_display.drawPixmap(self.target, self.imagePix, self.source)
        if self.view != 2: # view 0 or 1
            painter_display.drawPixmap(self.target, self.annotationPix, self.source)
        painter_display.drawPixmap(self.target, self.cursorPix, self.source)
        
    def drawCursorPoint(self, point):
        """Called when cursorPix needs update due to pen change or movement"""
        self.cursorPix.fill(self.color_picker(label=0, opacity=0)) # transparent
        painter_scribble = self.makePainter(self.cursorPix, 
                    self.color_picker(self.label, self.opacity)) # the painter used for cursor
        painter_scribble.drawPoint(point)   
    
    def mousePressEvent(self, event):
        #print('mouse press')
        if event.button() == PyQt5.QtCore.Qt.LeftButton: 
            if self.zPressed: # initiate zooming and not drawing
                self.cursorPix.fill(self.color_picker(label=0, opacity=0)) # clear (fill with transparent)
                self.lastCursorPoint = event.pos()
                self.activelyZooming = True
                self.newZoomValues = 0 # for distinction between reset and cancel
            else: # initiate drawing
                painter_scribble = self.makePainter(self.annotationPix, 
                        self.color_picker(self.label, self.annotationOpacity())) # the painter used for drawing        
                painter_scribble.drawPoint(event.pos())
                self.lastDrawPoint = event.pos()   
                self.activelyDrawing = True
            self.update()
    
    def mouseMoveEvent(self, event):
        #print('mouse move event')       
        if self.activelyZooming: 
            self.cursorPix.fill(self.color_picker(label=0, opacity=0)) # clear (fill with transparent)
            painter_scribble = self.makePainter(self.cursorPix,
                    self.color_picker(0, self.zoomOpacity))          
            x = min(self.lastCursorPoint.x(), event.x())
            y = min(self.lastCursorPoint.y(), event.y())
            w = abs(self.lastCursorPoint.x() - event.x())
            h = abs(self.lastCursorPoint.y() - event.y())      
            painter_scribble.fillRect(x,y,w,h,self.color_picker(0, self.zoomOpacity))
        else:          
            if self.activelyDrawing: 
                painter_scribble = self.makePainter(self.annotationPix, 
                        self.color_picker(self.label, self.annotationOpacity())) # the painter used for drawing        
                painter_scribble.drawLine(self.lastDrawPoint, event.pos())
                self.lastDrawPoint = event.pos()
            if self.zPressed:
                if  self.newZoomValues is None:
                    self.newZoomValues = 0 # for distinction between reset and cancel
            else:
                self.drawCursorPoint(event.pos())
            self.lastCursorPoint = event.pos()      
        self.update()
    
    def mouseReleaseEvent(self, event):
        #print('mouse release event')        
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
            self.padding = PyQt5.QtCore.QPoint((self.width()-self.source.width()*self.zoomFactor)/2, 0)
        else:
            self.zoomFactor = zoomWidth
            self.padding = PyQt5.QtCore.QPoint(0, (self.height()-self.source.height()*self.zoomFactor)/2)
            
        self.target = PyQt5.QtCore.QRect(self.padding, self.rect().bottomRight()-self.padding)
                   
    def executeZoom(self):
        """ Zooms to rectangle given by newZoomValues. """
        self.newZoomValues.translate(-self.padding)
        self.source = PyQt5.QtCore.QRect(self.newZoomValues.topLeft()/self.zoomFactor,
                self.newZoomValues.size()/self.zoomFactor)
        self.source.translate(-self.offset)
        self.source = self.source.intersected(self.imagePix.rect()) 
        print('   Zooming to ' + Annotator.formatQRect(self.source))     
        self.offset = self.imagePix.rect().topLeft() - self.source.topLeft()
        self.adjustTarget()
        self.newZoomValues = None
    
    def resetZoom(self):
        """ Back to original zoom """
        self.source = PyQt5.QtCore.QRect(0,0,self.imagePix.width(),self.imagePix.height())
        print(f'   Reseting zoom to ' + Annotator.formatQRect(self.source))        
        self.offset = PyQt5.QtCore.QPoint(0,0)
        self.adjustTarget()        
        self.newZoomValues = None
            
    def keyPressEvent(self, event):
        # print(f'key {event.key()}, text {event.text()}') 
        if 47<event.key()<58: #numbers 0 (48) to 9 (57)
            self.label = event.key()-48
            self.drawCursorPoint(self.lastCursorPoint)
            self.update()
            print(f'   Changed to label {self.label}')
        elif event.key()==16777235: # uparrow          
            self.penWidth = min(self.penWidth+1,50) 
            self.drawCursorPoint(self.lastCursorPoint)
            self.update()
            print(f'   Changed pen width to  {self.penWidth}')
        elif event.key()==16777237: # downarrow
            self.penWidth = max(self.penWidth-1,1)
            self.drawCursorPoint(self.lastCursorPoint)
            self.update()
            print(f'   Changed pen widht to  {self.penWidth}')
        elif event.key()==83: # s
            self.saveOutcome()
        elif event.key()==87: # w
            self.view = (self.view+1)%3
            self.update()
            print(f'   Changed view to  {self.view}')
        elif event.key()==72: #h
            self.printHelp()
        elif event.key()==90: # z
            if not self.zPressed:
                print('   Zooming enabled')
                self.zPressed = True
                self.cursorPix.fill(self.color_picker(label=0, opacity=0)) # clear (fill with transparent)
                self.update()
        elif event.key()==16777216: # escape
            self.closeEvent(event)
        self.setTitle()
        
    def keyReleaseEvent(self, event):
        if event.key()==90: # z
            if not self.activelyZooming:
                self.drawCursorPoint(self.lastCursorPoint)
                if self.newZoomValues is None:
                    self.resetZoom()
                elif self.newZoomValues==0:
                    print('   Zooming canceled')
                    self.newZoomValues = None
                else:
                    self.executeZoom()                       
                self.update()
            self.zPressed = False
            
    def closeEvent(self, event):
        print("Bye, I'm closing")
        PyQt5.QtWidgets.QApplication.quit()
        # hint from: https://stackoverflow.com/questions/54045134/pyqt5-gui-cant-be-close-from-spyder
        # should also check: https://github.com/spyder-ide/spyder/wiki/How-to-run-PyQt-applications-within-Spyder
   
    def saveOutcome(self):
        self.annotationPix.save('annotations.png', 'png')
        print('   Saved annotations')
        
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
    
    @classmethod 
    def printHelp(cls):
        """Help"""
        print('******* Help for annotator *******')
        print('KEYBORD COMMANDS:')
        print("   '1' to '9' changes label (pen color)")
        print("   '0' eraser mode")
        print("   'uparrow' and 'downarrow' changes pen width")
        print("   'W' changes view (image, annotation or both)")
        print("   'Z' hold down allows zoom")
        print("   'Z' pressed resets zoom")
        print("   'S' saves annotation")
        print("   'H' prints this help")
        print('**********************************')  
    
if __name__ == '__main__':
       
    app = PyQt5.QtWidgets.QApplication(sys.argv)
    if len(sys.argv)>1:
         filename = sys.argv[1]
         ex = Annotator.fromFilename(filename)
    else:
         ex = Annotator()
    # ex = Annotator((1024,512))    
    # ex = Annotator.fromFilename('../data/carbon.png')
    sys.exit(app.exec_())  
    
    
    
    