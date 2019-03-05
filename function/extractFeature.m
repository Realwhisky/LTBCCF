function feat  = extractFeature(imCNN, pos, window_sz, cos_window, indLayers)   % 注意这里的变量全不全，与上、下面的名称冲不冲突

% Get the search window from previous detection
patch = get_subwindow_CF2(imCNN, pos, window_sz);
%imshow(patch)
% Extracting hierarchical convolutional features
feat  = get_CNNfeatures(patch, cos_window, indLayers);        %  查看图片卷积特征提取__get_features.m





end