
% det using the  bb as [x y w d]

function [feat, pos_samples, labels, weights]=det_samples(im, pos, window_sz, det_config)

w_area=get_subwindow(im, pos, floor(window_sz*1.2));  % 获得1.2倍window_sz子窗口

feat=get_feature_detector(w_area, det_config.nbin);   % 从1.2倍padding窗获得feat特征

feat=imresize(feat, det_config.ratio, 'nearest');     % feat 输出为三维的特征矩阵
                                                      % 使用'最近邻插值方法‘，将长，宽，高 缩小为         即按设定的比例缩放
%config.ratio=sqrt(target_max_win/prod(target_sz));   % 设置ratio为  √￣（144/target_sz）

% w_area=imresize(w_area, det_config.ratio, 'nearest');

t_sz=det_config.t_sz;                                 % t_sz定义为config.t_sz=round(target_sz*config.ratio); 

% feat=get_feature_detector(w_area, det_config.nbin); 

sz=size(feat);                                        % 因为feat是一个三维的特征矩阵，SZ为feat特征矩阵的 长 宽 高

% step=max(floor(min(t_sz)/4),1);

step=1;                 

% *******************************************利用feat特征，k近邻的方式生成sample,pos,lables*******************************************

feat=im2colstep(feat, [t_sz(1:2), size(feat,3)], [step, step, size(feat,3)]); % 把feat特征矩阵重排，按列读取

[xx, yy]=meshgrid(1:step:sz(2)-t_sz(2)+1,1:step:sz(1)-t_sz(1)+1);        % xx是关于feat-winsz-targetsz的矩阵，yy同理

weights=fspecial('gaussian',size(xx), 25);             % 返回一个带有标准差sigma=25大小的旋转对称高斯低通滤波器，高斯权重

bb_samples=[xx(:), yy(:), ones(numel(xx),1)*t_sz(2), ones(numel(xx),1)*t_sz(1)];

bb_target=[(sz(2)-t_sz(2))/2, (sz(1)-t_sz(1))/2, t_sz(2), t_sz(1)];

labels=get_iou(bb_samples, bb_target);                       % IOU 样本与目标的重叠度，用来svm分类？

yy=(yy+t_sz(1)/2-sz(1)/2)/det_config.ratio;                  
yy=yy(:)+pos(1);

xx=(xx+t_sz(2)/2-sz(2)/2)/det_config.ratio;                  
xx=xx(:)+pos(2);

pos_samples=[yy' ; xx'];                                     % K近邻位置样本，相对位置信息

im_sz=det_config.image_sz;                   
% target_sz=det_config.target_sz;

% *********************************************************************************************************

idx=yy>im_sz(1) | yy<0 | ...                                 % idx标签
    xx>im_sz(2) | xx<0;

feat(:, idx)=[];                                             % 样本sample特征   

pos_samples(:, idx)=[];                                      % K近邻样本位置      bb_samples, bb_target

labels(idx)=[];                                              % IOU 重叠度标签

weights(idx)=[];                                             % 高斯权重

end

% *********************************************************************************************************

function iou = get_iou(r1,r2)

if size(r2,1)==1
    r2=r2(ones(1, size(r1,1)),:);
end

left = max((r1(:,1)),(r2(:,1)));
top = max((r1(:,2)),(r2(:,2)));
right = min((r1(:,1)+r1(:,3)),(r2(:,1)+r2(:,3)));
bottom = min((r1(:,2)+r1(:,4)),(r2(:,2)+r2(:,4)));
ovlp = max(right - left,0).*max(bottom - top, 0);
iou = ovlp./(r1(:,3).*r1(:,4)+r2(:,3).*r2(:,4)-ovlp);

end
