#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 23:25:54 2020

@author: vand
"""

import sys 
import PyQt5.QtCore  
import PyQt5.QtWidgets 
import PyQt5.QtGui


def __init__(self, model_item: Participant, parent=None):
        super().__init__(parent)

        self.model_item = model_item

        self.text = QGraphicsTextItem(self)

        self.line = QGraphicsLineItem(self)
        self.line.setPen(QPen(Qt.darkGray, 1, Qt.DashLine, Qt.RoundCap, Qt.RoundJoin))

        self.refresh() 
        
        

app = PyQt5.QtWidgets.QApplication(sys.argv) 
ex = PyQt5.QtWidgets.QWidget()

label = PyQt5.QtWidgets.QLabel()
label.setText("Hi")

ex.show()
sys.exit(app.exec_())  
