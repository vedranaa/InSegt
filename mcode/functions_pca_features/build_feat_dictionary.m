function dictionary = build_feat_dictionary(feat_im,dictionary_options)

dictionary.options = dictionary_options;

if isfield(dictionary_options,'method')
    method = dictionary_options.method;
else
    method = 'euclidean';    
end

if isfield(dictionary_options,'normalization')
    normalization = dictionary_options.normalization;
else
    dictionary.options.normalization = 'false';
    normalization = 'false';    
end

switch method
    case 'euclidean'
        dictionary.feat_tree = build_km_tree(feat_im,...
            1,... % patch size is always 1 for features
            dictionary_options.branching_factor,...
            dictionary_options.number_training_patches,...
            dictionary_options.number_layers,...
            normalization);
    case 'nxcorr'
        dictionary.feat_tree = build_km_tree_xcorr(feat_im,...
            1,...
            dictionary_options.branching_factor,...
            dictionary_options.number_training_patches,...
            dictionary_options.number_layers);
    otherwise
        error('Unknown dictionary method.')
        
end

% search km-tree
A = search_km_tree(feat_im, ...
                   dictionary.feat_tree, ...
                   dictionary_options.branching_factor, ...
                   normalization);

% build biadjacency matrix
B = biadjacency_matrix(A,dictionary_options.patch_size);

% compute the mapping matrices
[rc,nm] = size(B);
dictionary.T1 = sparse(1:nm,1:nm,1./(sum(B,1)+eps),nm,nm)*B';
dictionary.T2 = sparse(1:rc,1:rc,1./(sum(B,2)+eps),rc,rc)*B;














