#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 01:03:16 2020

@author: vand
"""

from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsScene, QGraphicsView, QGraphicsItem
from PyQt5.QtGui import QPen, QBrush
from PyQt5.Qt import Qt
 
import sys
 
 
 
 
class Window(QMainWindow):
    def __init__(self):
        super().__init__()
 

        self.setGeometry(500, 200, 600, 500)
 
        self.scene = QGraphicsScene()
        self.greenBrush = QBrush(Qt.green)
        self.grayBrush = QBrush(Qt.gray)
 
        self.pen = QPen(Qt.red)
 
        graphicView = QGraphicsView(self.scene, self)
        graphicView.setGeometry(0,0,600,500)
     
        self.scene.addEllipse(20,20, 200,200, self.pen, self.greenBrush)
        self.scene.addRect(-100,-100, 200,200, self.pen, self.grayBrush)
 
        self.show()
 
 
 
App = QApplication(sys.argv)
window = Window()
sys.exit(App.exec())