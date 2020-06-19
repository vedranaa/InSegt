# InSegt_script.py

This is a simplistic py version of InSegt.

The differences compared to (full interactive) matlab version:

* No interaction! Instead, you load an image and a corresponding image containing the user labeling. (The plan is to make a Qt-based interaction.)
* Uses only patch-based features for clustering. In matlab we have: patch based, normalized patches, SIFT features, PCA features, Gaussian-derivative features).
* Uses minibatch k-means from sklearn for clustering (in matlab we use k-means tree implemented in C++ and compiled as mex file)
* Treatment of unlabeled pixels (in matlab I distribute the probabilities to all classes if a pixel is unlabeled, and in python I keep zeros -- need to figure out what's best, and whether it matters). 
