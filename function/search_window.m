function [ window_sz, app_sz ] = search_window( target_sz, im_sz, config)

% ********* 这里实际上是搜索窗也就是padding窗大小的设置（根据target_sz的大小） *********

% 对于高度较大的对象，将搜索窗口限制为1.4倍高度。
       
    if (target_sz(1)/target_sz(2)<=2.4 && target_sz(1)/target_sz(2) >2)...
            || (target_sz(1)/target_sz(2)<=3.1 &&target_sz(1)/target_sz(2)>2.8)
        window_sz = floor(target_sz.*[1.4,1+config.padding ]);        
        % aaa=target_sz(1)/target_sz(2);   1.2*aaa
        
% target_sz = [ground_truth(1,4), ground_truth(1,3)]；config.padding=1.8（前面预设的）
% 目标高/宽>2的话，搜索框window_sz=向下取整（高*1.4，宽*（1+1.8））(高少搜一点，宽多搜一点)

% 对于高度和宽度都比较大的物体，并且至少占整个图像的10%，只搜索2倍的高度和宽度
    elseif (target_sz(1)/target_sz(2)>2.4 && target_sz(1)/target_sz(2)<=2.8)...
            || target_sz(1)/target_sz(2)>3.1
        window_sz = floor(target_sz.*[1.4,2+config.padding ]); 
        
    elseif min(target_sz)>80 && prod(target_sz)/prod(im_sz(1:2))>0.1
        window_sz=floor(target_sz*2);
    % 否则 若最小target_sz/im_sz也就是图像大小的比例超过了10%，或者最小target_sz宽或者高大于80，
    % 我们就取2倍的高和宽度的搜索窗；
    % 否则，就用自己配置的padding窗参数：

    else        
        window_sz = floor(target_sz * (1 + config.padding));
    end

    app_sz=target_sz+2*config.features.cell_size;   % 尺度窗保持大小不变，加两个cell_size是为了计算HOG特征时要去掉
                                                    % 最边上的两个（宽一个，高一个）cell，所以提前加上正好抵消
    
    % config.feature.cell_size 为HOG特征网格大小，在run_tracker里设置为4,灰度为1
    % 所以这里app_size=target_sz + 8 或者 target_sz + 2

end



