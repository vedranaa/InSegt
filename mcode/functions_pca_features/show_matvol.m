function show_matvol(V,clim)
% SHOW_MATVOL   Shows volumetric data
%   show_matvol(V,clim)
%   Inputs: volume, grayscale color limits
%   Author: vand@dtu.dk

fig = figure('Units','Normalized','Position',[0.1 0.3 0.5 0.6],...
    'KeyPressFcn',@key_press);
dim = size(V);
z = round(0.5*(dim(3)+1));
%update_drawing
imagesc(V(:,:,z),clim), hold on
title(['slice ',num2str(z),'/',num2str(dim(3))]), axis image ij
colormap gray, drawnow


%%%%%%%%%% CALLBACK FUNCTIONS %%%%%%%%%%
    function key_press(~,object)
        % keyboard commands
        key = object.Key;
        switch key
            case 'uparrow'
                z = min(z+1,dim(3));
                update_drawing
            case 'downarrow'
                z = max(z-1,1);
                update_drawing
            case 'rightarrow'
                z = min(z+10,dim(3));
                update_drawing
            case 'leftarrow'
                z = max(z-10,1);
                update_drawing
            case 'pagedown'
                z = min(z+50,dim(3));
                update_drawing
            case 'pageup'
                z = max(z-50,1);
                update_drawing
        end
    end
 
%%%%%%%%%% HELPING FUNCTIONS %%%%%%%%%%
    function update_drawing
        cla, imagesc(V(:,:,z),clim)
        title(['slice ',num2str(z),'/',num2str(dim(3))])
        drawnow
    end

end

