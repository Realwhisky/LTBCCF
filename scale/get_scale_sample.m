function out = get_scale_sample(im, pos, base_target_sz, scaleFactors, scale_window, scale_model_sz)

% ��ÿ������resize�ɹ̶���С���ֱ���ȡ31άfhog������ÿ������������fhog�ٴ�����һ��������������33���������������Ϊ����f
% ���ڳ߶�����˲���


% out = get_scale_sample(im, pos, base_target_sz, scaleFactors, scale_window, scale_model_sz)
% 
% Extracts a sample for the scale filter at the current
% location and scale.

% �߶�������˲�����SF���ڻ�ȡhog����ͼʱ������ ��ǰλ�� Ŀ��� �Ĵ�СΪ��׼����33�ֲ�ͬ�ĳ߶Ȼ�ȡ��ѡ���hog����ͼ��������������

nScales = length(scaleFactors);                        % �趨���ٸ��߶� 

for s = 1:nScales                                      % ��ȡ�߶�ѵ��������С a^n * P * a^n *Q
    patch_sz = floor(base_target_sz * scaleFactors(s));
    
    xs = floor(pos(2)) + (1:patch_sz(2)) - floor(patch_sz(2)/2);   % ��ȡ�߶�ѵ��������λ����ϢXs
    ys = floor(pos(1)) + (1:patch_sz(1)) - floor(patch_sz(1)/2);   % ��ȡ�߶�ѵ��������λ����ϢYs
    
    % check for out-of-bounds coordinates, and set them to the values at
    % the borders ����ⲿ���꣬������������Ϊ�߿��ֵ 
    xs(xs < 1) = 1;
    ys(ys < 1) = 1;
    xs(xs > size(im,2)) = size(im,2);
    ys(ys > size(im,1)) = size(im,1);
    
    % extract image
    im_patch = im(ys, xs, :);             % ��ѡ ��ȡͼƬ
    
    % resize image to model size          % ��ͼƬ������ͳһ��С
    im_patch_resized = imResample(im_patch, scale_model_sz);
    
    % extract scale features              % ��ȡ�߶� fHOG ����
    temp_hog = fhog(single(im_patch_resized), 4);
    temp = temp_hog(:,:,1:31);            % ��ȡ31bins��HOG������360/32=11.25��
    
    if s == 1
        out = zeros(numel(temp), nScales, 'single');
    end
    
    % window
    out(:,s) = temp(:) * scale_window(s);  % ���sz��ͬ������ͬ�߶�s�Ĵ�
end