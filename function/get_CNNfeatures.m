
function feat = get_CNNfeatures(im, cos_window, layers)

global net               % 全局调用
global enableGPU         % 全局调用

if isempty(net)          % net若为空，则
    initial_net();       % 输入为空
end

sz_window = size(cos_window);

% Preprocessing*******************官方介绍的图片读取处理步骤**********************************************************

img = single(im);        % note: [0, 255] range     img为单精度数组，范围0~255

img = imResample(img, net.meta.normalization.imageSize(1:2));  % 官方介绍的图片读取处理步骤，是im = imresize();

average=net.meta.normalization.averageImage;        % 训练数据的均值

% ******************************************************************************************************************
if numel(average)==3                                % 数组元素数目为3
    average=reshape(average,1,1,3);                 % 三通道输入
end

img = bsxfun(@minus, img, average);                 % 图像均值归一化，每个像素点减去该通道均值

if enableGPU, img = gpuArray(img); end

% Run the CNN
res = vl_simplenn(net,img);                         

% 开始使用CNN网络 读取图片

% Initialize feature maps                           % 初始化特征图映射
feat = cell(length(layers), 1);                     % feat 用来存储特征映射layers的cells = length(layers)*1 的大小

for ii = 1:length(layers)
    
    % Resize to sz_window
    if enableGPU
        x = gather(res(layers(ii)).x); 
    else
        x = res(layers(ii)).x;
    end
    
    x = imResample(x, sz_window(1:2));       % size_window是cos窗大小，把某一层卷积的特征映射输出resize成 cos窗一样的大小
                                             % 这里把某一层卷积特征插值成cos窗大小
    % windowing technique
    if ~isempty(cos_window),
        x = bsxfun(@times, x, cos_window);   % 然后特征图映射 作 cos窗处理
    end
    
    feat{ii}=x;                              % 映射后的值返回 fear{ii} 层
end
   % feat{3}
end
