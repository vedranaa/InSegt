## insegt_without_interaction.py

*A simplistic py version of InSegt*.

Features:

* No interaction! Instead, you load an image and a corresponding image containing the user labeling. (The plan is to make a Qt-based interaction.)
* For processing uses insegtbasic module.


See below the segmentation computed by inSegt_script.py, given an image and labels. 
<img src="example_output.png" width = "650">

## insegtbasic

*A module providing basic InSegt functionality*.

Purely python basic version of Insegt. Features:  
* Uses only patch-based features for clustering. In matlab we have: patch based, normalized patches, SIFT features, PCA features, Gaussian-derivative features).
* Uses minibatch k-means from sklearn for clustering (in matlab we use k-means tree implemented in C++ and compiled as mex file).
* Treatment of unlabeled pixels (in matlab we distribute the probabilities to all classes if a pixel is unlabeled, and in python we keep zeros -- need to figure out what's best, and whether it matters). 
