function [I,I_rgb,I_gray] = normalize_image(I)
    % initialization: normalize image
    if isa(I,'uint8')
        I = double(I)/255;
    end
    if isa(I,'uint16')
        I = double(I)/65535;
    end
    if size(I,3)==3 % rgb image
        I_gray = repmat(rgb2gray(I),[1 1 3]);
        I_rgb = I;
    else % assuming grayscale image
        I_gray = repmat(I,[1,1,3]);
        I_rgb = I_gray;
    end
end
