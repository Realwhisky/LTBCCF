function out = get_subwindow(im, pos, sz)
%   GET_SUBWINDOW Obtain sub-window from image, with replication-padding.
%   Returns sub-window of image IM centered at POS ([y, x] coordinates),
%   with size SZ ([height, width]). If any pixels are outside of the image,
%   they will replicate the values at the borders.

%   ������POSΪ���ĵ�ͼ��IM���Ӵ���([y��x]����)��
%   ��СΪSZ([�߶ȣ����])������κ�������ͼ��֮�⣬���ǽ��ڱ߿���ֵ��

%   Joao F. Henriques, 2014
%   http://www.isr.uc.pt/~henriques/

	if isscalar(sz),  %square sub-window       % �ж��Ƿ�Ϊ�������Ǳ�������һ������Ϊ����sz��С��
                                               % ���ε��Ӳ�������
		sz = [sz, sz];
    end
	
   
    xs = floor(pos(2)) + (1:sz(2)) - floor(sz(2)/2);       %����sz�Ĵ�С���ɲ�ͬ��Сpatch��xs,ys
    ys = floor(pos(1)) + (1:sz(1)) - floor(sz(1)/2);

    %check for out-of-bounds coordinates, and set them to the values at
    %the borders ����ⲿ���꣬������������Ϊ�߿��ֵ 
    xs(xs < 1) = 1;                           % ���ɵ�patch��С��СΪ1 
    ys(ys < 1) = 1;                           % 
    xs(xs > size(im,2)) = size(im,2);         % ����ͼ��patch��С�Ĺ�Ϊͼ��patch�߽�ֵ��
    
                                              % ����ȡĿ���������ء����������������������߽�������䣨padding�򳬳�ͼ��߽磩
    ys(ys > size(im,1)) = size(im,1);

    % extract image **************************************************************************** �����ȡimage patch����
    out=im(ys,xs,:);
    
end