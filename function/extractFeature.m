function feat  = extractFeature(imCNN, pos, window_sz, cos_window, indLayers)   % ע������ı���ȫ��ȫ�����ϡ���������Ƴ岻��ͻ

% Get the search window from previous detection
patch = get_subwindow_CF2(imCNN, pos, window_sz);
%imshow(patch)
% Extracting hierarchical convolutional features
feat  = get_CNNfeatures(patch, cos_window, indLayers);        %  �鿴ͼƬ���������ȡ__get_features.m





end