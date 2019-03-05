function out = get_scale_sample(im, pos, base_target_sz, scaleFactors, scale_window, scale_model_sz)

% 把每个样本resize成固定大小，分别提取31维fhog特征，每个样本的所有fhog再串联成一个特征向量构成33层金字塔特征，作为输入f
% 用于尺度相关滤波器


% out = get_scale_sample(im, pos, base_target_sz, scaleFactors, scale_window, scale_model_sz)
% 
% Extracts a sample for the scale filter at the current
% location and scale.

% 尺度相关性滤波器（SF）在获取hog特征图时，是以 当前位置 目标框 的大小为基准，以33种不同的尺度获取候选框的hog特征图！！！！！！！

nScales = length(scaleFactors);                        % 设定多少个尺度 

for s = 1:nScales                                      % 获取尺度训练样本大小 a^n * P * a^n *Q
    patch_sz = floor(base_target_sz * scaleFactors(s));
    
    xs = floor(pos(2)) + (1:patch_sz(2)) - floor(patch_sz(2)/2);   % 获取尺度训练样本的位置信息Xs
    ys = floor(pos(1)) + (1:patch_sz(1)) - floor(patch_sz(1)/2);   % 获取尺度训练样本的位置信息Ys
    
    % check for out-of-bounds coordinates, and set them to the values at
    % the borders 检查外部坐标，并将它们设置为边框的值 
    xs(xs < 1) = 1;
    ys(ys < 1) = 1;
    xs(xs > size(im,2)) = size(im,2);
    ys(ys > size(im,1)) = size(im,1);
    
    % extract image
    im_patch = im(ys, xs, :);             % 框选 提取图片
    
    % resize image to model size          % 把图片调整成统一大小
    im_patch_resized = imResample(im_patch, scale_model_sz);
    
    % extract scale features              % 提取尺度 fHOG 特征
    temp_hog = fhog(single(im_patch_resized), 4);
    temp = temp_hog(:,:,1:31);            % 提取31bins的HOG特征（360/32=11.25）
    
    if s == 1
        out = zeros(numel(temp), nScales, 'single');
    end
    
    % window
    out(:,s) = temp(:) * scale_window(s);  % 输出sz相同，但不同尺度s的窗
end