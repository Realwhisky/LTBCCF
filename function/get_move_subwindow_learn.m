function [out_npca, out_pca] = get_move_subwindow_learn(im, pos, model_sz, currentScaleFactor)

if isscalar(model_sz)
    model_sz = [model_sz, model_sz];
end

patch_sz = floor(model_sz * currentScaleFactor);

%make sure the size is not to small
if patch_sz(1) < 1
    patch_sz(1) = 2;
end;
if patch_sz(2) < 1
    patch_sz(2) = 2;
end;


xs = floor(pos(2)) + (1:patch_sz(2)) - floor(patch_sz(2)/2);
ys = floor(pos(1)) + (1:patch_sz(1)) - floor(patch_sz(1)/2);

%check for out-of-bounds coordinates, and set them to the values at
%the borders
xs(xs < 1) = 1;
ys(ys < 1) = 1;
xs(xs > size(im,2)) = size(im,2);
ys(ys > size(im,1)) = size(im,1);

%extract image###############################################前面定义size，这里是根据size划取image patch
im_patch = im(ys, xs, :);

%resize image to model size
% im_patch = imresize(im_patch, model_sz, 'bilinear');
im_patch = mexResize(im_patch, model_sz, 'auto');

% compute non-pca feature map
out_npca = [];

% compute pca feature map
temp_pca = fhog(single(im_patch),4);                 % 18+9+4=31维fhog特征
temp_pca(:,:,32) = cell_grayscale(im_patch,4);       % 这里是得到一个归一化的灰度直方图

out_pca = reshape(temp_pca, [size(temp_pca, 1)*size(temp_pca, 2), size(temp_pca, 3)]);  % 就是把特征聚合到一起了 （特征聚合的指令）





end

