function kf = gaussian_correlation(xf, yf, sigma)
%GAUSSIAN_CORRELATION Gaussian Kernel at all shifts, i.e. kernel correlation.
%   Evaluates a Gaussian kernel with bandwidth SIGMA for all relative
%   shifts between input images X and Y, which must both be MxN. They must 
%   also be periodic (ie., pre-processed with a cosine window). The result
%   is an MxN map of responses.
%   �ԡ�������ͼ��X��Y����֮����������ƫ�ƽ��д���Sigma�ĸ�˹�������������߶�������MXN��
%   ����Ҳ�����������Ե�(�������Ҵ���Ԥ����).�����һ��MXN��Ӧͼ��

%   Ҳ����˵������ʹ�ô���SIGMA�����˹���������������ͼ��X��Y֮������λ��  
%   ���붼��MxN��С�����߱��붼�����ڵģ�����ͨ��һ��cos���ڽ���Ԥ����  

%   Inputs and output are all in the Fourier domain.
%
%   Joao F. Henriques, 2014
%   http://www.isr.uc.pt/~henriques/
	
	N = size(xf,1) * size(xf,2);
	xx = xf(:)' * xf(:) / N;   % squared norm of x  ��׼������һ��
	yy = yf(:)' * yf(:) / N;   % squared norm of y  
	
	
	xyf = xf .* conj(yf);      % ���� yf ��Ԫ�صĸ�������˻�
	xy = sum(real(ifft2(xyf)), 3);  % to spatial domain ʹ�ÿ��ٸ���Ҷ�任�㷨���ؾ���Ķ�ά��ɢ �� ����Ҷ�任��ȡʵ��
                                    % ��ά��3����ÿ�е��ܺ�sum��A��3��
	
	%calculate gaussian response for all positions, then go back to the
	%Fourier domain
	kf = fft2(exp(-1 / sigma^2 * max(0, (xx + yy - 2 * xy) / numel(xf))));

%     kf=sum(xf.*conj(yf),3)/numel(xf);
%     n = numel(A) �������� A �е�Ԫ����Ŀ n ��ͬ�� prod(size(A))�����ﷵ������Ԫ�صĸ���

end

