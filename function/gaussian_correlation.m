function kf = gaussian_correlation(xf, yf, sigma)
%GAUSSIAN_CORRELATION Gaussian Kernel at all shifts, i.e. kernel correlation.
%   Evaluates a Gaussian kernel with bandwidth SIGMA for all relative
%   shifts between input images X and Y, which must both be MxN. They must 
%   also be periodic (ie., pre-processed with a cosine window). The result
%   is an MxN map of responses.
%   对——输入图像X和Y——之间的所有相对偏移进行带宽Sigma的高斯核评估，这两者都必须是MXN。
%   它们也必须是周期性的(即用余弦窗口预处理).结果是一个MXN响应图。

%   也就是说———使用带宽SIGMA计算高斯卷积核以用于所有图像X和Y之间的相对位移  
%   必须都是MxN大小。二者必须都是周期的（即，通过一个cos窗口进行预处理）  

%   Inputs and output are all in the Fourier domain.
%
%   Joao F. Henriques, 2014
%   http://www.isr.uc.pt/~henriques/
	
	N = size(xf,1) * size(xf,2);
	xx = xf(:)' * xf(:) / N;   % squared norm of x  标准化，归一化
	yy = yf(:)' * yf(:) / N;   % squared norm of y  
	
	
	xyf = xf .* conj(yf);      % 返回 yf 的元素的复共轭，作乘积
	xy = sum(real(ifft2(xyf)), 3);  % to spatial domain 使用快速傅里叶变换算法返回矩阵的二维离散 逆 傅里叶变换，取实部
                                    % 沿维度3返回每行的总和sum（A，3）
	
	%calculate gaussian response for all positions, then go back to the
	%Fourier domain
	kf = fft2(exp(-1 / sigma^2 * max(0, (xx + yy - 2 * xy) / numel(xf))));

%     kf=sum(xf.*conj(yf),3)/numel(xf);
%     n = numel(A) 返回数组 A 中的元素数目 n 等同于 prod(size(A))，这里返回数组元素的个数

end

