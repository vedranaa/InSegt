## InSegtPy

*A simplistic py version of InSegt*

Contains:

* `insegtbasic.py`, a module providing basic InSegt functionality. Features (and differences compared to matlab version):
   - Purely python. (In matlab we use mex files written in C++.)
   - Uses patch-based features for clustering. (In matlab we have: patch based, normalized patches, SIFT features, PCA features, Gaussian-derivative features.)
   - Uses minibatch k-means from sklearn for clustering. (In matlab we use k-means tree.)
   - Unlabeled pixels have zeros in label images. (In matlab we distribute the probabilities to all classes if a pixel is unlabeled. We need to figure out what's best, and whether it matters.)

* `demo_insegtbasic_without_interaction.py`, a script for processing an image using functionality from `insegtbasic.py`.
   - In particular, it uses `insegtbasic.patch_clustering` function for building the dictionary and `insegtbasic.two_binarized` function for processing the label image into a segmentation image.
   - No interaction! Instead, you load an image and a corresponding image containing the user labeling.

* `demo_insegtbasic_without_interaction_explained.py`, similar to demo above, but the processing implemented in `insegtbasic.two_binarized` is here divided in steps and visualised in more detail.

<img src="example_output.png" width = "650">


* `annotator.py`, an annotator class for drawing on an image. Based on qt5.




* `insegt_annotator.py` -- WORKING ON IN!
