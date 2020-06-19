# InSegt

Code accompanying our paper [*Content-based Propagation of User Markings for Interactive Segmentation of Patterned Images*](http://openaccess.thecvf.com/content_CVPRW_2020/papers/w57/Dahl_Content-Based_Propagation_of_User_Markings_for_Interactive_Segmentation_of_Patterned_CVPRW_2020_paper.pdf), published at [CVMI workshop](https://cvmi2020.github.io/) at CVPR 2020. An earlier version of the paper is on arxiv: [1809.02226](https://arxiv.org/pdf/1809.02226.pdf). The CVMI presentation can be seen [here](https://video.dtu.dk/media/InSegt+presentation/0_xafe3mki/319986).

Our interactive segmentation will, guided by the image content, propagate user markings to similar structures in the rest of the image. This allows easy segmentation of complex structures. For example, see how we obtain the segmentation in the image below using just a few markings. The image is a slice from &mu;CT scan of a bee eye. On the left, the input image is given a small subset of manually marked pixels. On the right, the manual labelling is being propagated to the whole image.

<img src="/images/bee_eye_segmentation.gif" width = "650">

In addition we have two examples of fibre detection in a composite material shown below.

<img src="/images/glass_example.png" width="650">\
<img src="/images/carbon_example.png" width="650">
