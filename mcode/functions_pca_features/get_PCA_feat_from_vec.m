function fIm = get_PCA_feat_from_vec(im, feat_vec, mean_patch, M, feat_type)

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
end



function featIm = get_feat_sub(im, M, feat_vec, mean_patch)

    [r,c] = size(im);
    Mh = floor(M/2);
    rM = r-M+1;
    cM = c-M+1;
    
%     colIm = zeros((r-M+1)*(c-M+1),M*M*l);
%     for i = 1:l
%         f = (i-1)*M*M+1;
%         t = i*M*M;
%         colIm(:,f:t) = im2col(im(:,:,i),[M,M])';
%     end
    feat_dim = size(feat_vec,2);
    featIm = zeros(rM,cM,feat_dim);
    iter = 0;
    for i = -Mh:Mh
        for j = -Mh:Mh
            iter = iter + 1;
            tmpIm = im((Mh+1+j):(end-Mh+j),(Mh+1+i):(end-Mh+i));
            featIm = featIm + reshape((tmpIm(:)-mean_patch(iter))*feat_vec(iter,:),[rM, cM, feat_dim]);
%             featIm((Mh+1):(end-Mh),(Mh+1):end-Mh,:) = featIm((Mh+1):(end-Mh),(Mh+1):(end-Mh),:) + ...
%                 reshape(tmpIm(:)*feat_vec,[rM, cM, feat_dim]);
        end
    end
%     feat = (colIm - ones((r-M+1)*(c-M+1),1)*mean_patch)*feat_vec;
%     fIm_no_border = reshape(feat,[r-M+1,c-M+1,feat_dim]);
end    











