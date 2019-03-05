function [out_npca, out_pca] = get_move_subwindow_ensemble(im, pos, window_sz, currentScaleFactor)

 
% % ����������^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
%     if ismatrix(im)                             % ȷ�������Ƿ�Ϊ����
%         im = cat(3, im, im, im);                % ����ǻҶ�ͼ���͸��Ƴ���ͨ���� &&&&&&
%     end
%     
%  indLayers = [37, 28, 19];
% %  indLayers = [19, 28, 37];
% cell_size = 4;
% cnnsz = floor(window_sz/cell_size);   % ����=CF2�е�l1_patch_num �� size;
% cos_window = hann(cnnsz(1)) * hann(cnnsz(2))';  
% feat = extractFeature(im, pos, window_sz, cos_window, indLayers);
% %feat{1}
% %nweights = [0.5,0.75,1];
%  nweights = [0.9,0.45,0.3];
% % feat_w = bsxfun(@times,feat,nweights);
%         if ismatrix(im)                      % ȷ�������Ƿ�Ϊ����
%         imCN = cat(3, im, im, im);          % ����ǻҶ�ͼ���͸��Ƴ���ͨ���� &&&&&&
%         else
%         imCN = im;                          % ����ǲ�ɫͼ������
%         end


if size(im,3) > 1, 
            im_gray = rgb2gray(im); 
else
            im_gray=im;
end



if isscalar(window_sz)
    window_sz = [window_sz, window_sz];
end

patch_sz = floor(window_sz * currentScaleFactor);

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
xs(xs > size(im_gray,2)) = size(im_gray,2);
ys(ys > size(im_gray,1)) = size(im_gray,1);

%extract image###############################################ǰ�涨��size�������Ǹ���size��ȡimage patch
im_patch = im_gray(ys, xs, :);

%resize image to model size
% im_patch = imresize(im_patch, model_sz, 'bilinear');
im_patch = mexResize(im_patch, window_sz, 'auto');

% compute non-pca feature map
out_npca = [];

% compute pca feature map
temp_pca = fhog(single(im_patch),4);                 % 18+9+4=31άfhog����
bb= fhog(single(im_patch),1); 
temp_pca(:,:,32) = cell_grayscale(im_patch,4);       % �����ǵõ�һ����һ���ĻҶ�ֱ��ͼ


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%����������ɫ��������%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  temp_pca_sz=size(temp_pca);
%  temp = load('w2crs');
%  w2c = temp.w2crs;
%  im_patch_cn = mexResize(imCN, window_sz, 'auto');  
%  out_CNfeat= get_CNfeature_map(im_patch_cn, 'cn', w2c);    % ��ɫ�ռ�˵Ӧ����10ά����ɫ�ռ�
%  out_CNfeat_R = imresize(out_CNfeat, [temp_pca_sz(1) temp_pca_sz(2)]);  
%  temp_pca(:,:,33:42) = out_CNfeat_R(:,:,1:10);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


 %im_patch = imresize(im, [sz(1) sz(2)]);
 %out_npca = get_feature_map(im_patch, 'gray', w2c);
 %out_pca = get_feature_map(im_patch, 'cn', w2c);
 
 
% 
% out(:,:,level+(1:10))= im2c(single(im_patch_cn), w2c, -2);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%feat_trans1=sum(feat{1},3);   512���������򵥵���
%feat_trans2=sum(feat{2},3);
%feat_trans3=sum(feat{3},3);
%��������������������������������������������������������������������
% conv_1=single(feat{1});
% conv_2=single(feat{2});
% conv_3=single(feat{3});
% 
% conv_1w = bsxfun(@times,conv_1,nweights(1));
% conv_2w = bsxfun(@times,conv_2,nweights(2));
% conv_3w = bsxfun(@times,conv_3,nweights(3));
% 
%  temp_pca(:,:,33:544) = (conv_1w(:,:,1:512));
%  temp_pca(:,:,545:1056) = (conv_2w(:,:,1:512));
%  temp_pca(:,:,1057:1312) = (conv_3w(:,:,1:256));
%��������������������������������������������������������������������
% temp_pca(:,:,33) = (conv_1(:,:,1));
% temp_pca(:,:,34) = (conv_1(:,:,10));
% temp_pca(:,:,35) = (conv_1(:,:,100));
% 
% temp_pca(:,:,36) = (conv_2(:,:,1));
% temp_pca(:,:,37) = (conv_2(:,:,10));


% temp_pca(:,:,545:800) = (conv_3w(:,:,1:256));
% imshow(conv_3(:,:,256))

% out_pca= reshape(temp_pca, [size(temp_pca, 1)*size(temp_pca, 2), size(temp_pca, 3)]);  % ���ǰ������ۺϵ�һ���� �������ۺϵ�ָ�
out_pca= reshape(temp_pca, [size(temp_pca, 1)*size(temp_pca, 2), size(temp_pca, 3)]); 

end


