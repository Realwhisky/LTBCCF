function update_visualization_func = show_video(img_files, video_path, resize_image)
%SHOW_VIDEO
%   Visualizes a tracker in an interactive figure, given a cell array of
%   image file names, their path, and whether to resize the images to
%   half size or not.                      
%
%   This function returns an UPDATE_VISUALIZATION function handle, that
%   can be called with a frame number and a bounding box [x, y, width,
%   height], as soon as the results for a new frame have been
%   calculated.                                                    
%   This way, your results are shown in real-time, but they are also
%   remembered  so you can navigate and inspect the video          
%   afterwards.                                                    
%   Press 'Esc' to send a stop signal (returned by UPDATE_VISUALIZATION).
%                                                                  
%   Joao F. Henriques, 2014
%   http://www.isr.uc.pt/~henriques/           

%   引用Joao F. Henriques的视频读取文件，该文件特点是视频算法运行同时，每一帧运行结果同时被记录下来了，所以能回滚来查看每一帧计算结果


	%store one instance per frame
	num_frames = numel(img_files);  
    
    %定义num_frames变量为图片总帧数；
    
	boxes = cell(num_frames,1);  % 创建元胞数组；
    
    %A cell array is a collection of containers called cells in which you can store different types of data
    %cell可以存储多种数据类型，cell(a,b) 返回一个 a×b 的元胞数组
    
	%create window
	[fig_h, axes_h, unused, scroll] = videofig(num_frames, @redraw, [], [], @on_key_press);  %#ok, unused outputs
% 	set(fig_h, 'Number','off', 'Name', ['Tracker - ' video_path])
    set(fig_h, 'Name', ['Tracker - ' video_path]);
	axis off;
    
    %axis OFF关闭所用坐标轴上的标记、格栅和单位标记。但保留由text和gtext设置的对象axis ON显示坐标轴上的标记、单位和格栅
	
	%image and rectangle handles start empty, they are initialized later
    
	im_h = [];
	rect_h = [];
	
	update_visualization_func = @update_visualization; % @是Matlab中的句柄函数的标志符，即间接的函数调用方法
	stop_tracker = false;                              % 在下面定义
	

	function stop = update_visualization(frame, box)
        
		% store the tracker instance for one frame, and show it. returns true if processing should stop (user pressed 'Esc').
		% 将跟踪器实例存储在一个帧中，并显示它。如果处理应该停止，则返回true(用户按下“ESC”)
        
        boxes{frame} = box;
		scroll(frame);
		stop = stop_tracker;
	end

	function redraw(frame)
		%render main image 渲染主图像
		im = imread([video_path img_files{frame}]);
		if size(im,3) > 1,
			im = rgb2gray(im);                                     % 彩色图转灰度图
		end
		if resize_image,
			im = imresize(im, 0.5);                                % 图像大小调整为一半
        end
                                                                   % im=imresize(im,1/resize_image);
		
		if isempty(im_h),  %create image
			im_h = imshow(im, 'Border','tight', 'InitialMag',200, 'Parent',axes_h);
        else                                                       % just update it
			set(im_h, 'CData', im)
		end
		
		%render target bounding box for this frame                 渲染bounding box给本帧（画出方框）
        
		if isempty(rect_h),  %create it for the first time
			rect_h = rectangle('Position',[0,0,1,1], 'EdgeColor','g', 'Parent',axes_h);
		end
		if ~isempty(boxes{frame}),
			set(rect_h, 'Visible', 'on', 'Position', boxes{frame});
		else
			set(rect_h, 'Visible', 'off');
		end
	end

	function on_key_press(key)
		if strcmp(key, 'escape'),                                   %stop on 'Esc'
			stop_tracker = true;
		end
	end

end

