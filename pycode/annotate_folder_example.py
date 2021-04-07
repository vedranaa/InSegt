#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 17:44:36 2021

@author: vand
"""

import glob
import annotator

lenpath = len('/Users/vand/Documents/some_example_data/')
images = glob.glob(foldername)

for image in images:
    print(f'Annotating image {image}')
    app = annotator.PyQt5.QtWidgets.QApplication([])
    ex = annotator.Annotator.fromFilename(image)
    #ex.annotationsFilename = image[lenpath:-4] + '_annotations.png'
    app.exec_()
