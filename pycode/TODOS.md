# Shorter-term TODOs

`Annotator`
* Better option for saving all outputs. Images should be placed in a folder and/or there should be a dialog for folder and filenames.
* Start annotator from a script
* Enable pan
* Fill function
* Adjust brightness of loaded image and transparency of label

`InSegtAnnotator`
* Better saving of outputs (inherits from Annotator).
* Possibility for turning off live update. For example **L** keyboard input for live processing on-off and **P** keyboard input for process once.
* Possibility for viewing probability images for probability-based functions. This may be just by saving probability images, such that one has to view them externally. Or by adding a layer.
* Change LICENSE to GPLv2. Add licence text to source code and README.md.
* Publish v2 betula borealis.

`InSegt`
* Add other types of processing functions! This includes SIFT features, PCA features, Gaussian-derivative features, multiscale features...
* Consider making a class `Segmentation`, with a method `Segmentation.processing_function`, which is to be passed to `InSegtAnnotator`.

# Longer-term TODOs
* Add slice-wise 3D functionality.
* Add undo (for example using 3-step image buffer)
* Add freeze label
