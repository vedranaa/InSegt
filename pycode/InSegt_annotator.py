""" Image annotator. 
Author: vand@dtu.dk, 2020
"""

import sys
import PyQt5.QtCore  
import PyQt5.QtWidgets 
import PyQt5.QtGui
import InSegt
import skimage.io
import numpy as np

# colors associated with different labels
colors = np.array([
    [0, 0, 0, 0], # background, will always be given alpha=0
    [255, 0, 0, 255], # label 1
    [0, 191, 0, 255], # label 2
    [0, 0, 255, 255], # etc
    [255, 127, 0, 255],
    [0, 255, 191, 255],
    [127, 0, 255, 255],
    [191, 255, 0, 255],
    [0, 127, 255, 255],
    [255, 64, 191, 255]
    ], dtype=np.uint8) 

def color_picker(label=0, opacity=0.5):
    color = PyQt5.QtGui.QColor(colors[label,0], colors[label,1], 
            colors[label,2], int(opacity*colors[label,3]))
    return(color)

def printHelp():
    print('******* Help for annotator *******')
    print('KEYBORD COMMANDS:')
    print("   '1' to '9' changes label (pen color)")
    print("   '0' eraser mode")
    print("   'uparrow' and 'downarrow' changes pen width")
    print("   'O' changes overlay (annotation, segmentation or both)")
    print("   'I' changes background (image or emtpy)")  
    print("   'S' saves annotation and segmentation")
    print("   'H' prints this help")
    print('**********************************')    
    
def rgbaToLabels(rgba):
    """RGBA image to labels from 0 to N. Uses colors."""    
    rgb = rgba.reshape(-1,4)[:,:3] # unfolding and removing alpha channel
    dist = np.sum(abs(rgb.reshape(-1,1,3) - colors.reshape(1,-1,4)[:,:,:3]), axis=2) # distances to pre-defined colors
    labels = np.argmin(dist, axis=1) # label given by the smallest distances
    labels = labels.reshape(rgba.shape[:2]) # folding back
    return(labels)

def labelsToRgba(labels, opacity=1):
    """Labels from 0 to N to RGBA. Uses colors."""
    rgba = colors[labels,:]
    rgba[:,:,3] = opacity*rgba[:,:,3]
    return(rgba)

def QPixmapToArray(qpixmap):
    """Qt pixmap to np array. Assumes an 8-bit RGBA pixmap."""
    qimage = qpixmap.toImage().convertToFormat(PyQt5.QtGui.QImage.Format_RGBA8888)
    buffer = qimage.constBits()
    buffer.setsize(qpixmap.height() * qpixmap.width() * 4) # 4 layers for RGBA
    img = np.frombuffer(buffer, np.uint8).reshape((qpixmap.height(), qpixmap.width(), 4))
    return img

def ArrayToQPixmap(img):
    """Np array to Qt pixmap. Assumes an 8-bit RGBA image."""
    img = img.copy()
    qimage = PyQt5.QtGui.QImage(img.data, img.shape[1], img.shape[0], 
                                PyQt5.QtGui.QImage.Format_RGBA8888)
    qpixmap = PyQt5.QtGui.QPixmap(qimage)
    return qpixmap

def grayToQPixmap(gray):
    """Uint8 grayscale image to Qt pixmap via RGBA image."""
    gray = np.tile(image,(4,1,1)).transpose(1,2,0)
    gray[:,:,3] = 255
    qpixmap = ArrayToQPixmap(gray)
    return qpixmap
  
class InSegtAnnotator(PyQt5.QtWidgets.QWidget):
    
    def __init__(self, image, processing_function):
        super().__init__() 
        #self.image = PyQt5.QtGui.QPixmap(filename)
        self.image = grayToQPixmap(image)
        self.resize(self.image.width(), self.image.height())
        self.annotation = PyQt5.QtGui.QPixmap(self.image.width(),self.image.height())
        self.annotation.fill(color_picker(label=0))
        self.label = 1
        self.penWidth = 10     
        self.lastPoint = PyQt5.QtCore.QPoint()
        self.show_image = True
        self.overlay_view = 2 # initial view: both
        self.annotation_opacity = 0.5
        self.setTitle()
        self.setCursor(PyQt5.QtGui.QCursor(PyQt5.QtCore.Qt.CrossCursor))
        
        self.processing_function = processing_function
        self.segmentation = PyQt5.QtGui.QPixmap(self.image.width(),self.image.height())
        self.segmentation.fill(color_picker(label=0))
        self.segmentation_opacity = 0.2
        self.show()
        
            
    def paintEvent(self, event):
        """ Painter for displaying the content of the widget."""
        #print('paint event')
        painter_display = PyQt5.QtGui.QPainter(self) # this is painter used for display
        painter_display.setCompositionMode(PyQt5.QtGui.QPainter.CompositionMode_SourceOver)
        if self.show_image:
            painter_display.drawPixmap(self.rect(), self.image)
        if self.overlay_view!=1: 
            painter_display.drawPixmap(self.rect(), self.annotation)
        if self.overlay_view!=0:
            painter_display.drawPixmap(self.rect(), self.segmentation)
            
    def setTitle(self):
        views = {0:'annotation', 1:'segmentation', 2:'both'}
        self.setWindowTitle(f'L:{self.label}, P:{self.penWidth}, O:{views[self.overlay_view]}. For CL help, hit H')
    
    def mousePressEvent(self, event):
        #print('mouse press')
        if event.button() == PyQt5.QtCore.Qt.LeftButton: 
            self.lastPoint = event.pos()

    def mouseMoveEvent(self, event):
        #print('mouse move event')
        if event.buttons() & PyQt5.QtCore.Qt.LeftButton: # one of buttons pressed and left button
            
            painter_scribble = PyQt5.QtGui.QPainter(self.annotation) # this is painter used for drawing        
            painter_scribble.setPen(PyQt5.QtGui.QPen(color_picker(self.label, self.annotation_opacity), 
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
            painter_scribble.setPen(PyQt5.QtGui.QPen(color_picker(self.label, self.annotation_opacity), 
                                            self.penWidth*2/(self.scaleWidth+self.scaleWidth), 
                                PyQt5.QtCore.Qt.SolidLine, 
                                PyQt5.QtCore.Qt.RoundCap, 
                                PyQt5.QtCore.Qt.RoundJoin))
            painter_scribble.scale(self.scaleWidth, self.scaleHeight)
            painter_scribble.setCompositionMode(PyQt5.QtGui.QPainter.CompositionMode_Source)
            painter_scribble.drawPoint(self.lastPoint)
            self.lastPoint = event.pos()
            self.transformLabels()
            self.update()            
   
    def heightForWidth(self, w):
        return w * self.image.height()/self.image.width()
         
    def resizeEvent(self, event):
        """ Triggered by resizing of the widget window. """
        # this does not influence size, is only book-keeping
        self.scaleWidth = self.image.width()/self.width() 
        self.scaleHeight = self.image.height()/self.height()             
     
    def keyPressEvent(self, event):
        #print(f'key {event.key()}, text {event.text()}') 
        if 47<event.key()<58: #numbers 0 (48) to 9 (57)
            self.label = event.key()-48
            #print(f'   Changed to label {self.label}')
        elif event.key()==16777235: # uparrow          
            self.penWidth = min(self.penWidth+1,50) 
            #print(f'   Changed pen width to  {self.penWidth}')
        elif event.key()==16777237: # downarrow
            self.penWidth = max(self.penWidth-1,1)
            #print(f'   Changed pen widht to  {self.penWidth}')
        elif event.key()==83: # s
            self.saveResult()
            #print('   Saved annotations')
        elif event.key()==79: # w
            self.overlay_view = (self.overlay_view+1)%3
            self.update()
            #print(f'   Changed view to  {self.overlay_view}')
        elif event.key()==73: # i
            self.show_image = not self.show_image 
            self.update()
            #print(f'   Changed show_image to  {self.show_image}')
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
   
    def saveResult(self):
        self.annotation.save('annotation.png', 'png')
        self.segmentation.save('segmentation.png', 'png')
        
    def transformLabels(self):
        """Transforming pixmap annotation to pixmap segmentation via numpy"""
        
        # transformations before
        an = QPixmapToArray(self.annotation) # numpy RGBA: height x width x 4, values uint8    
        labels = rgbaToLabels(an) # numpy labels: height x width, values 0 to N 
        
        # actual work
        segmentation = self.processing_function(labels) # numpy labels: height x width, values 0 to N 
        
        # transformations after
        sg = labelsToRgba(segmentation, self.segmentation_opacity) # numpy RGBA: height x width x 4, values uint8    
        self.segmentation = ArrayToQPixmap(sg)     
    

if __name__ == '__main__':
    
    """ 
    To initializing InSegt annotator, you need:
    - A grayscale uint8 image represented as a H-by-W numpy array.
    - A processing function which given a H-by-W labeling returns a H-by-W 
        segmentation, both represented as numpy arrays. Elements of labeling 
        are integers from 0 to N, where 0 is used for unlabeled pixels.
    """
 
    # if len(sys.argv)>1:
    #    filename = sys.argv[1]    
    
    filename = '../data/glass.png'
    image = skimage.io.imread(filename)           

    patch_size = 9
    nr_training_patches = 10000
    nr_clusters = 100
    T1, T2 = InSegt.patch_clustering(image, patch_size, nr_training_patches, nr_clusters)
    processing_function = lambda labels: InSegt.two_binarized(labels, T1, T2)
    
   
    """ Now initialize InSegt annotator. """
    
    app = PyQt5.QtWidgets.QApplication(sys.argv) 
    ex = InSegtAnnotator(image, processing_function)
    print("Starting annotator. For help, hit 'H'")
    sys.exit(app.exec_())  
    