clear
close all
addpath functions

% image
im = imread('../data/slice1.png');

% dictionary
dictopt.patch_size = 9;
dictopt.branching_factor = 5;
dictopt.number_layers = 4;
dictopt.number_training_patches = 30000;
dictionary = build_dictionary(im,dictopt);

% labeling saved earlier
labeling = imread('../data/slice1_labeling.png');
nr_labels = max(labeling(:));

% processing slice1 without opening gui and updating dictionary
image_texture_gui(im,dictionary,nr_labels,labeling,'distributed')
dictionary = update_dictionary(dictionary,gui_dictprob);

% processing all images
figure
for i=1:5
    I = imread(['../data/slice',num2str(i),'.png']);
    S = process_image(I,dictionary);
    subplot(1,5,i)
    imagesc(S,[0,nr_labels]), axis image, title(i), drawnow
end
