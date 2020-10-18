import sys 
import PyQt5.QtCore  
import PyQt5.QtWidgets 
import PyQt5.QtGui


def makeHelpPixmap(width,height):
    scene = PyQt5.QtWidgets.QGraphicsScene()
    helpText = ('******* Help for annotator *******' + '\n' +
        'KEYBORD COMMANDS:' + '\n' +
        "   'H' prints this help" + '\n' +
        "   'esc' closes" + '\n' +
        '**********************************') 
    text = PyQt5.QtWidgets.QGraphicsSimpleTextItem(helpText)      
    scene.addItem(text)          
    graphicsView = PyQt5.QtWidgets.QGraphicsView(scene)
    graphicsView.setGeometry(0, 0, width, height) 
    graphicsView.setBackgroundBrush((PyQt5.QtGui.QColor("bisque")))
    textPix = PyQt5.QtGui.QPixmap(width, height) 
    painter_scribble = PyQt5.QtGui.QPainter(textPix)
    graphicsView.render(painter_scribble)
    textPix.save('text_pixmap.png', 'png')
    
makeHelpPixmap(200,200)    
    
#%%


class WidgetWithText(PyQt5.QtWidgets.QWidget):
    def __init__(self):
        super().__init__()       
  
    
        self.width = 600
        self.height = 500
  
        
        self.scene = PyQt5.QtWidgets.QGraphicsScene()
        text = PyQt5.QtWidgets.QGraphicsSimpleTextItem("Lorem ipsum dolor sit amet,\nconsectetur adipiscing elit,\nsed do eiusmod tempor incididunt\nut labore et dolore magna aliqua.\nUt enim ad minim veniam,\nquis nostrud exercitation ullamco\nlaboris nisi ut aliquip\nex ea commodo consequat.")      
        self.scene.addItem(text)          
        
        graphicsView = PyQt5.QtWidgets.QGraphicsView(self.scene, self)
        graphicsView.setGeometry(0, 0, self.width, self.height) 
        graphicsView.setBackgroundBrush((PyQt5.QtGui.QColor("bisque")))
        self.textPix = PyQt5.QtGui.QPixmap(self.width, self.height) 

        painter_scribble = PyQt5.QtGui.QPainter(self.textPix)
        graphicsView.render(painter_scribble)
        
  
        self.show()
        self.textPix.save('text_pixmap.png', 'png')
       


  
app = PyQt5.QtWidgets.QApplication(sys.argv)
w = WidgetWithText()
sys.exit(app.exec_())
 
