#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 17:44:36 2021

@author: vand
"""

import glob
import annotator

foldername = '/Users/vand/Documents/PROJECTS2/Bone/Originaltdata/MJPR130202 ID8/MJPR130202 ID8_2014-02-18_1503_00/4X-50keV-10W-air-5s/recon/'
lenpath = len(foldername)
images = glob.glob(foldername +  '*444.tiff')

for image in images:
    print(f'Annotating image {image}')
    app = annotator.PyQt5.QtWidgets.QApplication([])
    ex = annotator.Annotator.fromFilename(image)
    ex.annotationsFilename = image[lenpath:-4] + '_annotations.png'
    app.exec_()
