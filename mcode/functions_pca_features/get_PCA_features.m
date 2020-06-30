function [feat_im, feat_model] = get_PCA_features(im, feat_param)
% Comptes a PCA-based feature from an image
% function feat_im = get_PCA_features(im, feat_param)
% Input:
%   im - image
%   feat_param - struct containing parameters for computing PCA-based
%     features containing:
%     feat_param.patch_size - size of image patches on which PCA should be
%       performed. Default: 15
%     feat_param.n_patch - number of patches for building the PCA
%       covariance matrix. Default: 50000
%     feat_param.n_keep - number of PCA features to keep. If n_keep < 1 (and n_keep > 0) it will return
%       the featuers explaining the n_keep variance. Otherwise it will return
%       n_keep features (n_keep <= patch_size^2). Default: 10
%     feat_param.feat_type - optional three vector determining which derivatives are
%       used for computing the PCA feature image. At least one must be 1.
%       Default: [1, 1, 1]
%       if feat_param.feat_type(1) == 1 then the raw image is used
%       if feat_param.feat_type(2) == 1 then the first order derivative images are used
%       if feat_param.feat_type(3) == 1 then the second order derivative images are used
%     feat_param.scale_factor - downscales the input image to increase
%       speed. Default: 0.5
%   
% Output:
%   feat_im - volumetric image with PCA features in the third dimension
%   feat_model - struct containing mean vector and PCA projection matrix.
%       It must be used for computing PCA features from a new image using
%       the same model when doing batch processing.
%   
% Author: Anders Bjorholm Dahl, November 21, 2017
% 

if ( nargin == 1 )
    feat_param = struct('patch_size', 15, 'n_patch', 50000, 'n_keep', 20, ...
    'feat_type', ones(1,3), 'scale_factor', 0.5);
end

if isfield(feat_param, 'patch_size')
    patch_size = feat_param.patch_size;
end

if isfield(feat_param, 'n_patch')
    n_patch = feat_param.n_patch;
end

if isfield(feat_param, 'n_keep')
    n_keep = feat_param.n_keep;
end

if isfield(feat_param, 'feat_type')
    feat_type = feat_param.feat_type;
end

if isfield(feat_param, 'scale_factor')
    scale_factor = feat_param.scale_factor;
end


[r,c] = size(im);
im = imresize(im,scale_factor);

dim_feat = 0;
feat_id = 0;

mean_patch = {};
feat_im_cell = {};
feat_vec = {};
if ( feat_type(1) == 1 )
    feat_id = feat_id + 1;
    [feat_im_gray, vec, mean_patch{feat_id}] = getPCA_feat(im,patch_size,n_patch,n_keep);
    feat_im_cell{feat_id} = feat_im_gray;
    keep_gray = size(feat_im_gray,3);
    feat_vec{feat_id} = vec(:,end-keep_gray+1:end);
    dim_feat = dim_feat + keep_gray;
end
    

if ( feat_type(2) == 1 )
    dg = [1,0,-1];

    imX = imfilter(im,dg,'replicate');
    feat_id = feat_id + 1;
    [feat_im_x, vec, mean_patch{feat_id}] = getPCA_feat(imX,patch_size,n_patch,n_keep);
    keep_x = size(feat_im_x,3);
    feat_im_cell{feat_id} = feat_im_x;
    feat_vec{feat_id} = vec(:,end-keep_x+1:end);
    dim_feat = dim_feat + keep_x;

    imY = imfilter(im,dg','replicate');
    feat_id = feat_id + 1;
    [feat_im_y, vec, mean_patch{feat_id}] = getPCA_feat(imY,patch_size,n_patch,n_keep);
    keep_y = size(feat_im_y,3);
    feat_im_cell{feat_id} = feat_im_y;
    feat_vec{feat_id} = vec(:,end-keep_y+1:end);
    dim_feat = dim_feat + keep_y;
end

if ( feat_type(3) == 1 )
    dg = [1,0,-1];
    ddg = [1,-2,1];
    
    imXX = imfilter(im,ddg,'replicate');
    feat_id = feat_id + 1;
    [feat_im_xx, vec, mean_patch{feat_id}] = getPCA_feat(imXX,patch_size,n_patch,n_keep);
    keep_xx = size(feat_im_xx,3);
    feat_im_cell{feat_id} = feat_im_xx;
    feat_vec{feat_id} = vec(:,end-keep_xx+1:end);
    dim_feat = dim_feat + keep_xx;
    
    imYY = imfilter(im,ddg','replicate');
    feat_id = feat_id + 1;
    [feat_im_yy, vec, mean_patch{feat_id}] = getPCA_feat(imYY,patch_size,n_patch,n_keep);
    keep_yy = size(feat_im_yy,3);
    feat_im_cell{feat_id} = feat_im_yy;
    feat_vec{feat_id} = vec(:,end-keep_yy+1:end);
    dim_feat = dim_feat + keep_yy;

    imXY = imfilter(imfilter(im,dg,'replicate'),dg','replicate');
    feat_id = feat_id + 1;
    [feat_im_xy, vec, mean_patch{feat_id}] = getPCA_feat(imXY,patch_size,n_patch,n_keep);
    keep_xy = size(feat_im_xy,3);
    feat_im_cell{feat_id} = feat_im_xy;
    feat_vec{feat_id} = vec(:,end-keep_xy+1:end);
    dim_feat = dim_feat + keep_xy;
end

feat_im = ones(size(im,1), size(im,2), dim_feat);
t = 0;
for i = 1:size(feat_im_cell,2)
    f = t + 1;
    t = t + size(feat_im_cell{i},3);
    feat_im(:,:,f:t) = feat_im_cell{i};
end


feat_im = imresize(feat_im,[r,c]);

feat_model.feat_vec = feat_vec;
feat_model.mean_patch = mean_patch;
