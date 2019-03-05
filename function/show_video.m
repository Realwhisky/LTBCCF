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

%   ����Joao F. Henriques����Ƶ��ȡ�ļ������ļ��ص�����Ƶ�㷨����ͬʱ��ÿһ֡���н��ͬʱ����¼�����ˣ������ܻع����鿴ÿһ֡������


	%store one instance per frame
	num_frames = numel(img_files);  
    
    %����num_frames����ΪͼƬ��֡����
    
	boxes = cell(num_frames,1);  % ����Ԫ�����飻
    
    %A cell array is a collection of containers called cells in which you can store different types of data
    %cell���Դ洢�����������ͣ�cell(a,b) ����һ�� a��b ��Ԫ������
    
	%create window
	[fig_h, axes_h, unused, scroll] = videofig(num_frames, @redraw, [], [], @on_key_press);  %#ok, unused outputs
% 	set(fig_h, 'Number','off', 'Name', ['Tracker - ' video_path])
    set(fig_h, 'Name', ['Tracker - ' video_path]);
	axis off;
    
    %axis OFF�ر������������ϵı�ǡ���դ�͵�λ��ǡ���������text��gtext���õĶ���axis ON��ʾ�������ϵı�ǡ���λ�͸�դ
	
	%image and rectangle handles start empty, they are initialized later
    
	im_h = [];
	rect_h = [];
	
	update_visualization_func = @update_visualization; % @��Matlab�еľ�������ı�־��������ӵĺ������÷���
	stop_tracker = false;                              % �����涨��
	

	function stop = update_visualization(frame, box)
        
		% store the tracker instance for one frame, and show it. returns true if processing should stop (user pressed 'Esc').
		% ��������ʵ���洢��һ��֡�У�����ʾ�����������Ӧ��ֹͣ���򷵻�true(�û����¡�ESC��)
        
        boxes{frame} = box;
		scroll(frame);
		stop = stop_tracker;
	end

	function redraw(frame)
		%render main image ��Ⱦ��ͼ��
		im = imread([video_path img_files{frame}]);
		if size(im,3) > 1,
			im = rgb2gray(im);                                     % ��ɫͼת�Ҷ�ͼ
		end
		if resize_image,
			im = imresize(im, 0.5);                                % ͼ���С����Ϊһ��
        end
                                                                   % im=imresize(im,1/resize_image);
		
		if isempty(im_h),  %create image
			im_h = imshow(im, 'Border','tight', 'InitialMag',200, 'Parent',axes_h);
        else                                                       % just update it
			set(im_h, 'CData', im)
		end
		
		%render target bounding box for this frame                 ��Ⱦbounding box����֡����������
        
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

