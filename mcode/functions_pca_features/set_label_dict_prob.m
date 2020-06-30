function dictionary = set_label_dict_prob(dictionary, dict_P)

L = size(dict_P,2);
M = dictionary.options.patch_size;
n_patch = size(dict_P,1)/(M*M);
gui_dictprob = reshape(dict_P,[M^2,n_patch,L]);
gui_dictprob = permute(gui_dictprob,[1,3,2]);
gui_dictprob = reshape(gui_dictprob,[M^2*L,n_patch]);

dictionary.label_dict_prob = gui_dictprob;
