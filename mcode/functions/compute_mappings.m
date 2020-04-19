function [mappings,A] = compute_mappings(image,dictionary)

if isfield(dictionary.options,'method')
    method = dictionary.options.method;
else
    method = 'euclidean';    
end

if isfield(dictionary.options,'normalization')
    normalization = dictionary.options.normalization;
else
    normalization = 'false';    
end

switch method
    case 'euclidean'
        A = search_km_tree(image,...
            dictionary.tree,...
            dictionary.options.branching_factor,...
            normalization);
    case 'nxcorr'
        A = search_km_tree_xcorr(image,...
            dictionary.tree,...
            dictionary.options.branching_factor);
    otherwise
        error('Unknown dictionary method.')
end
B = biadjacency_matrix(A,dictionary.options.patch_size,size(dictionary.tree,2));

[rc,nm] = size(B);
mappings.T1 = sparse(1:nm,1:nm,1./(sum(B,1)+eps),nm,nm)*B';
mappings.T2 = sparse(1:rc,1:rc,1./(sum(B,2)+eps),rc,rc)*B;

