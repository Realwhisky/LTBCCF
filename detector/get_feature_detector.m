function [ feat ] = get_feature_detector( I, nbin )
                                        %(w_area, det_config.nbin=32 )
% color image
if ~ismatrix(I) && ~isequal(I(:,:,1),I(:,:,2),I(:,:,3))
    I = uint8(255*RGB2Lab(I));                             % 将彩色图像转换为 无符号整数的 L,A,B颜色域
    nth=4;                                                 % 增加了四个颜色通道
else % gray image    
    I=I(:,:,1);
    nth=8;
end

thr=(1/(nth+1):1/(nth+1):1-1/(nth+1))*255;                  %  nth =4,thr= 51.0000  102.0000  153.0000  204.0000

ksize=4;

f_iif=255-calcIIF(I(:,:,1),[ksize ksize],nbin);  % nbin=32;

f_chn=cat(3,f_iif, I);

feat=zeros(size(f_chn,1), size(f_chn, 2), nth*size(f_chn,3)); % feat 为三维的矩阵

for ii=1:size(f_chn,3)                                        % size(A,3)返回维度A的第三维长度
    
    t_chn=f_chn(:,:,ii);
    t_chn=t_chn(:,:,ones(1,nth));                             
    t_chn=bsxfun(@gt, t_chn, reshape(thr, 1, 1, nth));        % t_chn是（:,:,4）,thr reshape成一个1*1*4的矩阵
    feat(:,:,(ii-1)*nth+1:ii*nth)=t_chn;
end

end

