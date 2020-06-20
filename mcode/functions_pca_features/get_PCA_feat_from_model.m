function feat_im = get_PCA_feat_from_model(im_in, feat_model, feat_param)

    feat_vec = feat_model.feat_vec;
    mean_patch = feat_model.mean_patch;
    feat_type = feat_param.feat_type;
    M = feat_param.patch_size;
    
    im = imresize(im_in, feat_param.scale_factor);

    [r,c] = size(im);

    Mh = floor(M/2);

    dim_tot = 0;
    for i = 1:size(feat_vec,2)
        dim_tot = dim_tot + size(feat_vec{i}, 2);
    end

    fIm = zeros(r,c,dim_tot);

    feat_id = 0;
    to_im = 0;

    if ( feat_type(1) == 1 )
        feat_id = feat_id + 1;

        fIm_no_border = get_feat_sub(im, M, feat_vec{feat_id}, mean_patch{feat_id});

        from_im = to_im + 1;
        to_im = to_im + size(fIm_no_border, 3);
        fIm(Mh+1:end-Mh,Mh+1:end-Mh,from_im:to_im) = fIm_no_border;
    end


    if ( feat_type(2) == 1 )
        feat_id = feat_id + 1;

        dg = [1,0,-1];
        imX = imfilter(im,dg,'replicate');

        fIm_no_border = get_feat_sub(imX, M, feat_vec{feat_id}, mean_patch{feat_id});

        from_im = to_im + 1;
        to_im = to_im + size(fIm_no_border, 3);
        fIm(Mh+1:end-Mh,Mh+1:end-Mh,from_im:to_im) = fIm_no_border;
        
        feat_id = feat_id + 1;
        imY = imfilter(im,dg','replicate');

        fIm_no_border = get_feat_sub(imY, M, feat_vec{feat_id}, mean_patch{feat_id});

        from_im = to_im + 1;
        to_im = to_im + size(fIm_no_border, 3);
        fIm(Mh+1:end-Mh,Mh+1:end-Mh,from_im:to_im) = fIm_no_border;
        
    end
    
    if ( feat_type(3) == 1 )
        feat_id = feat_id + 1;

        dg = [1,0,-1];
        ddg = [1,-2,1];
        
        imXX = imfilter(im,ddg,'replicate');

        fIm_no_border = get_feat_sub(imXX, M, feat_vec{feat_id}, mean_patch{feat_id});

        from_im = to_im + 1;
        to_im = to_im + size(fIm_no_border, 3);
        fIm(Mh+1:end-Mh,Mh+1:end-Mh,from_im:to_im) = fIm_no_border;
        
        feat_id = feat_id + 1;
        imYY = imfilter(im,ddg','replicate');

        fIm_no_border = get_feat_sub(imYY, M, feat_vec{feat_id}, mean_patch{feat_id});

        from_im = to_im + 1;
        to_im = to_im + size(fIm_no_border, 3);
        fIm(Mh+1:end-Mh,Mh+1:end-Mh,from_im:to_im) = fIm_no_border;
        
        feat_id = feat_id + 1;
        imXY = imfilter(imfilter(im,dg','replicate'),dg,'replicate');

        fIm_no_border = get_feat_sub(imXY, M, feat_vec{feat_id}, mean_patch{feat_id});

        from_im = to_im + 1;
        to_im = to_im + size(fIm_no_border, 3);
        fIm(Mh+1:end-Mh,Mh+1:end-Mh,from_im:to_im) = fIm_no_border;
        
    end
    
    feat_im = imresize(fIm, size(im_in));
end


function featIm = get_feat_sub(im, M, feat_vec, mean_patch)
    [r,c] = size(im);
    
    im_patch = im2col(im,[M,M])';
    im_patch = im_patch - ones(size(im_patch,1),1)*mean_patch;
%     lImP = sqrt(sum(im_patch.^2));
%     id = find(lImP > 0);
%     im_patch(:,id) = im_patch(:,id)./(ones(size(id,1),1)*lImP(id));
    feat_patch = im_patch*feat_vec;
    featIm = reshape(feat_patch,[r-M+1,c-M+1,size(feat_vec,2)]);
end 


% function featIm = get_feat_sub(im, M, feat_vec, mean_patch)
% 
%     [r,c] = size(im);
%     Mh = floor(M/2);
%     rM = r-M+1;
%     cM = c-M+1;
%     
%     feat_dim = size(feat_vec,2);
%     featIm = zeros(rM,cM,feat_dim);
%     iter = 0;
%     for i = -Mh:Mh
%         for j = -Mh:Mh
%             iter = iter + 1;
%             tmpIm = im((Mh+1+j):(end-Mh+j),(Mh+1+i):(end-Mh+i));
%             tmpIm = tmpIm(:)-mean_patch(iter);
%             lTmpIm = sqrt(sum(tmpIm.^2));
%             id = find(lTmpIm > 0);
%             tmpIm(:,id) = tmpIm(:,id)./lTmpIm(id);
%             featIm = featIm + reshape(tmpIm*feat_vec(iter,:),[rM, cM, feat_dim]);
%         end
%     end
% end    
% 
% 
% 






