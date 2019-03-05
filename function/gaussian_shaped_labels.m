function labels = gaussian_shaped_labels(sigma, sz)
%GAUSSIAN_SHAPED_LABELS
%   Gaussian-shaped labels for all shifts of a sample.
%
%   LABELS = GAUSSIAN_SHAPED_LABELS(SIGMA, SZ)
%   Creates an array of labels (regression targets) for all shifts of a
%   sample of dimensions SZ. The output will have size SZ, representing
%   one label for each possible shift. The labels will be Gaussian-shaped,
%   with the peak at 0-shift (top-left element of the array), decaying
%   as the distance increases, and wrapping around at the borders.
%   The Gaussian function has spatial bandwidth SIGMA.
%   为维度SZ的所有移动创建标签数组(回归目标)。输出将具有SZ大小，表示每个可能的移位的一个标签。
%   标签将是高斯型，峰值在0-移位(阵列的左上角元素)，随着距离的增加而衰减，
%   并在边框周围环绕。高斯函数具有空间带宽Sigma。
%
%   Joao F. Henriques, 2014
%   http://www.isr.uc.pt/~henriques/


% 	%as a simple example, the limit sigma = 0 would be a Dirac delta
% 	sigma=0 会变成德雷克三角函数
% 	%instead of a Gaussian:
% 	labels = zeros(sz(1:2));  %labels for all shifted samples 循环位移的图片样本的标签
% 	labels(1,1) = magnitude;  %label for 0-shift (original sample)0-移位标签(原始样本)
	                          %magnitude为一个幅值

	%evaluate a Gaussian with the peak at the center element
	[rs, cs] = ndgrid((1:sz(1)) - floor(sz(1)/2), (1:sz(2)) - floor(sz(2)/2));
                                                           %rs,cs代表目标宽高的各种组合（训练patches）
	labels = exp(-0.5 / sigma^2 * (rs.^2 + cs.^2));        %标准的高斯公式
    
	%move the peak to the top-left, with wrap-around将峰值移到左上角，并绕着它转
	labels = circshift(labels, -floor(sz(1:2) / 2) + 1);

	%sanity check: make sure it's really at top-left健全检查：确保它真的在左上角
	assert(labels(1,1) == 1)

end

