""" Image annotator. 
Author: vand@dtu.dk, 2020
"""

import sys
import PyQt5.QtCore  
import PyQt5.QtWidgets 
import PyQt5.QtGui

# Todo: circular cursor indicating pen width, check QCursor
# Resizing preserving aspect ratio, check QWidget.setSizePolicy
# Zooming-in, check QWidged.canvas.setFocusPolicy

# colors associated with different labels
def color_picker(label=0, opacity=0.5):
    opacity_value = int(opacity*255)
    colors = {
        0: PyQt5.QtGui.QColor(127, 127, 127,   0),
        1: PyQt5.QtGui.QColor(255,   0,   0, opacity_value),
        2: PyQt5.QtGui.QColor(  0, 191,   0, opacity_value),
        3: PyQt5.QtGui.QColor(  0,   0, 255, opacity_value),
        4: PyQt5.QtGui.QColor(255, 127,   0, opacity_value),
        5: PyQt5.QtGui.QColor(  0, 255, 191, opacity_value),
        6: PyQt5.QtGui.QColor(127,   0, 255, opacity_value),
        7: PyQt5.QtGui.QColor(191, 255,   0, opacity_value),
        8: PyQt5.QtGui.QColor(  0, 127, 255, opacity_value),
        9: PyQt5.QtGui.QColor(255,   64, 191, opacity_value)}
    return colors[label]

def printHelp():
    print('******* Help for annotator *******')
    print('KEYBORD COMMANDS:')
    print("   '1' to '9' changes label (pen color)")
    print("   '0' eraser mode")
    print("   'uparrow' and 'downarrow' changes pen width")
    print("   'W' changes view (image, annotation or both)")
    print("   'S' saves annotation")
    print("   'H' prints this help")
    print('**********************************')
  
class Annotator(PyQt5.QtWidgets.QWidget):
    
    def __init__(self,filename):
        super().__init__() 
        self.image = PyQt5.QtGui.QPixmap(filename)
        self.resize(self.image.width(), self.image.height())
        self.annotation = PyQt5.QtGui.QPixmap(self.image.width(),self.image.height())
        self.annotation.fill(color_picker(label=0, opacity=0.5))
        self.label = 1
        self.penWidth = 3     
        self.lastPoint = PyQt5.QtCore.QPoint()
        self.view = 0
        self.opacity = 0.5
        self.setTitle()
        self.setCursor(PyQt5.QtGui.QCursor(PyQt5.QtCore.Qt.CrossCursor))

    def paintEvent(self, event):
        """ Painter for displaying the content of the widget."""
        #print('paint event')
        painter_display = PyQt5.QtGui.QPainter(self) # this is painter used for display
        painter_display.setCompositionMode(PyQt5.QtGui.QPainter.CompositionMode_SourceOver)
        if self.view!=1: # view 0 or 1
            painter_display.drawPixmap(self.rect(), self.image)
        if self.view!=2: # view 0 or 2
            painter_display.drawPixmap(self.rect(), self.annotation)
    
    def setTitle(self):
        views = {0:'both', 1:'annotation', 2:'image'}
        self.setWindowTitle(f'L:{self.label}, P:{self.penWidth}, W:{views[self.view]}')

    def mousePressEvent(self, event):
        #print('mouse press')
        if event.button() == PyQt5.QtCore.Qt.LeftButton: 
            self.lastPoint = event.pos()

    def mouseMoveEvent(self, event):
        #print('mouse move event')
        if event.buttons() & PyQt5.QtCore.Qt.LeftButton: # one of buttons pressed and left button
            
            painter_scribble = PyQt5.QtGui.QPainter(self.annotation) # this is painter used for drawing        
            painter_scribble.setPen(PyQt5.QtGui.QPen(color_picker(self.label, self.opacity), 
                                            self.penWidth*2/(self.scaleWidth+self.scaleWidth), 
                                PyQt5.QtCore.Qt.SolidLine, 
                                PyQt5.QtCore.Qt.RoundCap, 
                                PyQt5.QtCore.Qt.RoundJoin))
            painter_scribble.scale(self.scaleWidth, self.scaleHeight)
            painter_scribble.setCompositionMode(PyQt5.QtGui.QPainter.CompositionMode_Source)
            painter_scribble.drawLine(self.lastPoint, event.pos())
            self.lastPoint = event.pos()
            self.update()
    
    def mouseReleaseEvent(self, event):
        #print('mouse release event')
        if PyQt5.QtCore.Qt.LeftButton:
            painter_scribble = PyQt5.QtGui.QPainter(self.annotation) # this is painter used for drawing        
            painter_scribble.setPen(PyQt5.QtGui.QPen(color_picker(self.label, self.opacity), 
                                            self.penWidth*2/(self.scaleWidth+self.scaleWidth), 
                                PyQt5.QtCore.Qt.SolidLine, 
                                PyQt5.QtCore.Qt.RoundCap, 
                                PyQt5.QtCore.Qt.RoundJoin))
            painter_scribble.scale(self.scaleWidth, self.scaleHeight)
            painter_scribble.setCompositionMode(PyQt5.QtGui.QPainter.CompositionMode_Source)
            painter_scribble.drawPoint(self.lastPoint)
            self.lastPoint = event.pos()
            self.update()            
            
    def resizeEvent(self, event):
        """ Handles resizing of the widget window. """
        self.scaleWidth = self.image.width()/self.width()
        self.scaleHeight = self.image.height()/self.height()      
        #print(f'scaling {self.scaleWidth}, sy {self.scaleHeight}')          
    
    def keyPressEvent(self, event):
        # print(f'key {event.key()}, text {event.text()}') 
        if 47<event.key()<58: #numbers 0 (48) to 9 (57)
            self.label = event.key()-48
            print(f'   Changed to label {self.label}')
        elif event.key()==16777235: # uparrow          
            self.penWidth = min(self.penWidth+1,50) 
            print(f'   Changed pen width to  {self.penWidth}')
        elif event.key()==16777237: # downarrow
            self.penWidth = max(self.penWidth-1,1)
            print(f'   Changed pen widht to  {self.penWidth}')
        elif event.key()==83: # s
            self.saveAnnotations()
            print('   Saved annotations')
        elif event.key()==87: # w
            self.view = (self.view+1)%3
            self.update()
            print(f'   Changed view to  {self.view}')
        elif event.key()==72: #h
            printHelp()
        elif event.key()==16777216: # escape
            self.closeEvent(event)
        self.setTitle()
            
    def closeEvent(self, event):
        print("Bye, I'm closing")
        PyQt5.QtWidgets.QApplication.quit()
        # hint from: https://stackoverflow.com/questions/54045134/pyqt5-gui-cant-be-close-from-spyder
        # should also check: https://github.com/spyder-ide/spyder/wiki/How-to-run-PyQt-applications-within-Spyder
   
    def saveAnnotation(self):
        self.annotation.save('annotations.png', 'png')
    
if __name__ == '__main__':
    app = PyQt5.QtWidgets.QApplication(sys.argv) 
    if len(sys.argv)>1:
       filename = sys.argv[1]    
    else:
       filename = '../data/glass.png'
    ex = Annotator(filename)
    ex.show()
    print("Starting annotator. For help, hit 'H'")
    sys.exit(app.exec_())  
    