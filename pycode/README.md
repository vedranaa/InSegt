## InSegtPy

*A simplistic py version of InSegt*

Contains:

* `demo_insegtannotator.py`, a demo showing how `insegtannotator` together with `insegtbasic` may be used for interactive segmentation.

Input image | User labelings | Segmentation result
:---:|:---:|:---:
<img src="ExampleFigures/glass/gray.png" width = "250">  |  <img src="ExampleFigures/glass/annotations_overlay.png" width = "250"> | <img src="ExampleFigures/glass/segmentations_overlay.png" width = "250">

* `insegtbasic.py`, a module providing basic InSegt image processing functionality. Features (and differences compared to matlab version):
   - Purely python. (In matlab, we use mex files written in C++.)
   - Uses patch-based features for clustering. (In matlab, we have: patch-based, normalized patches, SIFT features, PCA features, Gaussian-derivative features.)
   - Uses minibatch k-means from sklearn for clustering. (In matlab, we use k-means tree.)
   - Unlabeled pixels have zeros in label images. (In matlab, we distribute the probabilities to all classes if a pixel is unlabeled. We need to figure out what's best, and whether it matters.)  


* `demo_insegtbasic_without_interaction.py`, a script that processes an image using functionality from `insegtbasic.py`.
   - In particular, it uses `insegtbasic.patch_clustering` function for building the dictionary and `insegtbasic.two_binarized` function for processing the label image into a segmentation image.
   - No interaction! Instead, you load an image to be segmented, and a same-size image containing the user labeling.

<div align="center"><img src="ExampleFigures/demo_insegtbasic.png" width = "750"></div>


* `demo_insegtbasic_without_interaction_explained.py`, similar to  the demo above, but the processing implemented in `insegtbasic.two_binarized` is divided into steps and visualized in more detail.
  - In particular, here you have access to assignment image and the probability images for different labels.

<div align="center"><img src="ExampleFigures/demo_insegtbasic_explained.png" width = "750"></div>

* `annotator.py`, an annotator for drawing on an image. Based on qt5.

* `insegtannotator.py`, annotator allowing for interactive segmentation. This is an extension of the annotator used for interactive segmentation. To use `InsegtAnnotator` you need a processing function that given labels returns segmentation.
