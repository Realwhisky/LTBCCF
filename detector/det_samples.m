
% det using the  bb as [x y w d]

function [feat, pos_samples, labels, weights]=det_samples(im, pos, window_sz, det_config)

w_area=get_subwindow(im, pos, floor(window_sz*1.2));  % ���1.2��window_sz�Ӵ���

feat=get_feature_detector(w_area, det_config.nbin);   % ��1.2��padding�����feat����

feat=imresize(feat, det_config.ratio, 'nearest');     % feat ���Ϊ��ά����������
                                                      % ʹ��'����ڲ�ֵ������������������ ��СΪ         �����趨�ı�������
%config.ratio=sqrt(target_max_win/prod(target_sz));   % ����ratioΪ  �̣���144/target_sz��

% w_area=imresize(w_area, det_config.ratio, 'nearest');

t_sz=det_config.t_sz;                                 % t_sz����Ϊconfig.t_sz=round(target_sz*config.ratio); 

% feat=get_feature_detector(w_area, det_config.nbin); 

sz=size(feat);                                        % ��Ϊfeat��һ����ά����������SZΪfeat��������� �� �� ��

% step=max(floor(min(t_sz)/4),1);

step=1;                 

% ********************************************����feat������k���ڵķ�ʽ����sample,pos,lables**************************************************

feat=im2colstep(feat, [t_sz(1:2), size(feat,3)], [step, step, size(feat,3)]); % ��feat�����������ţ����ж�ȡ

[xx, yy]=meshgrid(1:step:sz(2)-t_sz(2)+1,1:step:sz(1)-t_sz(1)+1);        % xx�ǹ���feat-winsz-targetsz�ľ���yyͬ��

weights=fspecial('gaussian',size(xx), 25);             % ����һ�����б�׼��sigma=25��С����ת�ԳƸ�˹��ͨ�˲�������˹Ȩ��

bb_samples=[xx(:), yy(:), ones(numel(xx),1)*t_sz(2), ones(numel(xx),1)*t_sz(1)];

bb_target=[(sz(2)-t_sz(2))/2, (sz(1)-t_sz(1))/2, t_sz(2), t_sz(1)];

labels=get_iou(bb_samples, bb_target);                       % IOU ������Ŀ����ص��ȣ�����svm���ࣿ

yy=(yy+t_sz(1)/2-sz(1)/2)/det_config.ratio;                  
yy=yy(:)+pos(1);

xx=(xx+t_sz(2)/2-sz(2)/2)/det_config.ratio;                  
xx=xx(:)+pos(2);

pos_samples=[yy' ; xx'];                                     % K����λ�����������λ����Ϣ

im_sz=det_config.image_sz;                   
% target_sz=det_config.target_sz;

% *********************************************************************************************************

idx=yy>im_sz(1) | yy<0 | ...                                 % idx��ǩ
    xx>im_sz(2) | xx<0;

feat(:, idx)=[];                                             % ����sample����   

pos_samples(:, idx)=[];                                      % K��������λ��      bb_samples, bb_target

labels(idx)=[];                                              % IOU �ص��ȱ�ǩ

weights(idx)=[];                                             % ��˹Ȩ��

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