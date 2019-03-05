function out = get_subwindow(im, pos, sz)
%   GET_SUBWINDOW Obtain sub-window from image, with replication-padding.
%   Returns sub-window of image IM centered at POS ([y, x] coordinates),
%   with size SZ ([height, width]). If any pixels are outside of the image,
%   they will replicate the values at the borders.

%   返回以POS为中心的图像IM的子窗口([y，x]坐标)，
%   大小为SZ([高度，宽度])。如果任何像素在图像之外，它们将在边框复制值。

%   Joao F. Henriques, 2014
%   http://www.isr.uc.pt/~henriques/

	if isscalar(sz),  %square sub-window       % 判断是否为标量，是标量生成一个长宽都为标量sz大小的
                                               % 方形的子采样窗口
		sz = [sz, sz];
    end
	
   
    xs = floor(pos(2)) + (1:sz(2)) - floor(sz(2)/2);       %根据sz的大小生成不同大小patch的xs,ys
    ys = floor(pos(1)) + (1:sz(1)) - floor(sz(1)/2);

    %check for out-of-bounds coordinates, and set them to the values at
    %the borders 检查外部坐标，并将它们设置为边框的值 
    xs(xs < 1) = 1;                           % 生成的patch大小最小为1 
    ys(ys < 1) = 1;                           % 
    xs(xs > size(im,2)) = size(im,2);         % 超出图像patch大小的归为图像patch边界值，
    
                                              % 即提取目标区域像素——————————超边界则做填充（padding框超出图像边界）
    ys(ys > size(im,1)) = size(im,1);

    % extract image **************************************************************************** 最后提取image patch窗口
    out=im(ys,xs,:);
    
end