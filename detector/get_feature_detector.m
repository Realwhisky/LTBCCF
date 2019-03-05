function [ feat ] = get_feature_detector( I, nbin )
                                        %(w_area, det_config.nbin=32 )
% color image
if ~ismatrix(I) && ~isequal(I(:,:,1),I(:,:,2),I(:,:,3))
    I = uint8(255*RGB2Lab(I));                             % ����ɫͼ��ת��Ϊ �޷��������� L,A,B��ɫ��
    nth=4;                                                 % �������ĸ���ɫͨ��
else % gray image    
    I=I(:,:,1);
    nth=8;
end

thr=(1/(nth+1):1/(nth+1):1-1/(nth+1))*255;                  %  nth =4,thr= 51.0000  102.0000  153.0000  204.0000

ksize=4;

f_iif=255-calcIIF(I(:,:,1),[ksize ksize],nbin);  % nbin=32;

f_chn=cat(3,f_iif, I);

feat=zeros(size(f_chn,1), size(f_chn, 2), nth*size(f_chn,3)); % feat Ϊ��ά�ľ���

for ii=1:size(f_chn,3)                                        % size(A,3)����ά��A�ĵ���ά����
    
    t_chn=f_chn(:,:,ii);
    t_chn=t_chn(:,:,ones(1,nth));                             
    t_chn=bsxfun(@gt, t_chn, reshape(thr, 1, 1, nth));        % t_chn�ǣ�:,:,4��,thr reshape��һ��1*1*4�ľ���
    feat(:,:,(ii-1)*nth+1:ii*nth)=t_chn;
end

end

