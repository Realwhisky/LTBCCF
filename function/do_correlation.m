function [ pos, max_response ] = do_correlation( im, pos, window_sz, cos_window, config, model)
% *******************************计算位置滤波器Rc 与 响应得分***********************************
% if size(im,3) > 1, im = rgb2gray(im); end

cell_size=config.features.cell_size;                             % HOG cell size

patch = get_subwindow(im, pos, window_sz);                       % 训练样本集f--patch，调用get_subwindow.m生成大小不同的--训练样本集
             
zf = fft2(get_features(patch,config,cos_window));	             % 获取第二帧 预测位置 待检测图片特征	（已经经过cos窗处理）

%kzf = gaussian_correlation(zf, model.xf, config.kernel_sigma); % 高斯核K（z，x），xf是样本的patch特征，zf是待预测区域样本
                                                                 % config.kernel_sigma在run_tracker里定义的=1
 
 kzf = linear_correlation(zf, model.xf);   % *******我在这里把高斯核改成了线性核 ，为了避免 降维中 高斯核方法面临的 平滑子空间约束问题*********   

                                                                 
response =fftshift(real(ifft2(model.alphaf .* kzf)));            % 求最终响应，返回实部
                                                                 % Y = fftshift(X) 通过将零频分量移动到数组中心，重新排列傅里叶变换 X
                                                                 % 分析信号的频率分量时，将零频分量平移到中心会很有帮助
                                                      
max_response=max(response(:));                                   % 找到最大响应

[vert_delta, horiz_delta] = find(response == max_response, 1);   % 在 response 中查找第1个 response == max_response的元素
                                                                 % 得到新中心的X、Y坐标变化
                                                                                                                                                                                                           
pos = pos + cell_size * [vert_delta - floor(size(zf,1)/2)-1, horiz_delta - floor(size(zf,2)/2)-1]; 

                                                                 % 位置更新
                                                      
end

