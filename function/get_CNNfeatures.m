
function feat = get_CNNfeatures(im, cos_window, layers)

global net               % ȫ�ֵ���
global enableGPU         % ȫ�ֵ���

if isempty(net)          % net��Ϊ�գ���
    initial_net();       % ����Ϊ��
end

sz_window = size(cos_window);

% Preprocessing*******************�ٷ����ܵ�ͼƬ��ȡ������**********************************************************

img = single(im);        % note: [0, 255] range        imgΪ���������飬��Χ0~255

img = imResample(img, net.meta.normalization.imageSize(1:2));  % �ٷ����ܵ�ͼƬ��ȡ�����裬��im = imresize();

average=net.meta.normalization.averageImage;        % ѵ�����ݵľ�ֵ

% ******************************************************************************************************************
if numel(average)==3                                % ����Ԫ����ĿΪ3
    average=reshape(average,1,1,3);                 % ��ͨ������
end

img = bsxfun(@minus, img, average);                 % ͼ���ֵ��һ����ÿ�����ص��ȥ��ͨ����ֵ

if enableGPU, img = gpuArray(img); end

% Run the CNN
res = vl_simplenn(net,img);                         

% ��ʼʹ��CNN���� ��ȡͼƬ

% Initialize feature maps                           % ��ʼ������ͼӳ��
feat = cell(length(layers), 1);                     % feat �����洢����ӳ��layers��cells = length(layers)*1 �Ĵ�С

for ii = 1:length(layers)
    
    % Resize to sz_window
    if enableGPU
        x = gather(res(layers(ii)).x); 
    else
        x = res(layers(ii)).x;
    end
    
    x = imResample(x, sz_window(1:2));       % size_window��cos����С����ĳһ����������ӳ�����resize�� cos��һ���Ĵ�С
                                             % �����ĳһ����������ֵ��cos����С
    % windowing technique
    if ~isempty(cos_window),
        x = bsxfun(@times, x, cos_window);   % Ȼ������ͼӳ�� �� cos������
    end
    
    feat{ii}=x;                              % ӳ����ֵ���� fear{ii} ��
end
   % feat{3}
end
