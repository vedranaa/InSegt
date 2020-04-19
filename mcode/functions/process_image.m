function [S,P] = process_image(im,dictionary)
% segmentation and probabilities from and image and a dictionary

im = normalize_image(im);

if isfield(dictionary.options,'normalization')
    normalization = dictionary.options.normalization;
else
    normalization = 'false';    
end

P = probability_search_km_tree(im, ...
    dictionary.tree, ...
    dictionary.dictprob, ...
    dictionary.options.branching_factor, ...
    normalization);

[maxprob,S] = max(P,[],3);
uncertain = sum(P == maxprob(:,:,ones(size(P,3),1)),3)>1; 
S(uncertain) = 0;
S = uint8(S);

end

