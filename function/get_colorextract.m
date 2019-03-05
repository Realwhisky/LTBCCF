
%*****************************************************************************************************************
function [out_npca, out_pca] = get_colorextract(im, pos, window_sz, currentScaleFactor, w2c)

        if isscalar(window_sz),  %square sub-window
            window_sz = [window_sz, window_sz];
        end
        patch_sz = floor(window_sz * currentScaleFactor);

        xs = floor(pos(2)) + (1:patch_sz(2)) - floor(patch_sz(2)/2);
        ys = floor(pos(1)) + (1:patch_sz(1)) - floor(patch_sz(1)/2);

%check for out-of-bounds coordinates, and set them to the values at
%the borders
        xs(xs < 1) = 1;
        ys(ys < 1) = 1;
        xs(xs > size(im,2)) = size(im,2);
        ys(ys > size(im,1)) = size(im,1);

%extract image
        im_patch = im(ys, xs, :);

%check for out-of-bounds coordinates, and set them to the values at
%the borders
        im_patch = mexResize(im_patch, window_sz, 'auto');
        
    %    im_patch_sz = size(im_patch);
        out_pca = [];
        out_npca = [];
        
        temp_CN= get_CNfeature_map(im_patch, 'cn', w2c);    % 颜色空间 应该是10维的颜色空间
 %      out_CNfeat_R = mexResize(im_patch, out_CNfeat, 'auto');  
        %imshow(out_pca)
        out_pca= reshape(temp_CN, [prod(window_sz), size(temp_CN, 3)]); 




