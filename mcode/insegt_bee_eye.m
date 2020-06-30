addpath functions
addpath functions_pca_features

%% Read and show image

im = double(rgb2gray(imread('../data/bee_eye.png')))/255;

im = imresize(im,2); % double the size to make annotation easier
figure,imagesc(im), axis image, colormap gray

%% Set feature paramters
feat_param.patch_size = 11; % patch size for computing PCA features
feat_param.n_patch = 50000; % number of patches for computing convariance matrix for PCA features
% number of principal components to keep. If the value is between 0 and 1,
% it will take the number of principal components corresponding to that
% fraction of the variance.
feat_param.n_keep = 10; 
% which types of derivatives to include. If the value is 0, the feature
% will be excluded. First element is 0th order, second is 1st order and
% thrid element is 2nd order. Ex. feat_param.feat_type = [0 1 0] will only
% compute the 1st order derivative.
feat_param.feat_type = [1, 1, 1];
% Down scale the image for faster computation. Note that the effective
% patch size changes relative to this parameter. 
feat_param.scale_factor = 1; 
[feat_im, feat_model] = get_PCA_features(im, feat_param); % Compute the features.

% Set segmentation method parameters
dictopt.method = 'euclidean';
dictopt.patch_size = 15;
dictopt.branching_factor = 5;
dictopt.number_layers = 5;
dictopt.number_training_patches = 50000;
dictionary = build_feat_dictionary(feat_im,dictopt);

%% Run the texture segmentation
% Press 'm' to choose the method called 'two_max', whic will propagate the 
% labels twice, which gives better results. You see the segmentation result
% best by pressing 'w' to make the result ovelayed on the image.

[seg_im, P, D] = image_texture_gui(im,dictionary,3);
