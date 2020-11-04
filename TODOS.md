
# Comments from Rasmus Tuxen 15/7-2020


1. Remove everything with xcorr (it is not used by the tool as i understand).

1. The main introduction scripts need explanatory text. This is especially needed for 'texture_gui_uses_script.m' and 'reusing_labels_script.m'.

1. Many scripts require the user to do a certain action (like export or save) which if not done will create an error message 
(ex. 'batch_processing_with_interaction.m').
Is this intentional/acceptable?  

1. Errors with 'process_image.m':
	1. Using PCA features to build dictionary results in error when using 'process_image' function. 
 	    See 'example_PCA_processing_error.m' for example.

	1. Should segmentation work on 3-channel images like RGB? 
	    It works for 'image_texture_gui.m' but not 'process_image.m'.

1. Move functions from 'image_texture_gui.m' such that functions like 'process_image.m' can use identical method of segmentation.
Currently, a segmentation made in the gui and one in 'process_image' can look very dissimilar using the same dictionary. 
See 'example_PCA_processing_error.m' attached in the mail from 15/7 for example (change 'build_dictionary(feat_im,dictopt)' to 'build_dictionary(im,dictopt)').

1. In 'image_texture_gui.m' [A] changes opacity (alpha), [C] changes colormap. This is not indicated anywhere in the gui.
Should it be?

1. In 'image_texture_gui.m' it is indicated that both [key] and shift+[key] perform the same action. Why is the shift+[key] option needed then?

1. Feature request: Add ctrl+z option to drawing in gui. It would make it more intuative to use. 

1. General code cleanup: There are several TODO's in code (ex image_texture_gui.m).

1. Update names for probability methods (distributed, two_max etc.) 
