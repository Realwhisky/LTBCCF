function [img_files, pos, target_sz, ground_truth, video_path] = load_video_info(base_path, video)


%LOAD_VIDEO_INFO
%   Loads all the relevant information for the video in the given path:
%   the list of image files (cell array of strings), initial position
%   (1x2), target size (1x2), the ground truth information for precision
%   calculations (Nx2, for N frames), and the path where the images are
%   located. The ordering of coordinates and sizes is always [y, x].
%   在给定路径中加载视频的所有相关信息：图像文件列表(字符串单元数组)、初始
%   位置(1x2)、目标大小(1x2)、ground truth的基本信息(Nx2，用于N帧)
%   以及图像所在的路径。坐标和大小的排序总是[y，x]。


%***********************************先读路径，再读视频，再读ground_truth********************************


	% see if there's a suffix, specifying one of multiple targets, for
	% example the dot and number in 'Jogging.1' or 'Jogging.2'.
    % 查看是否有后缀，找一个有多个运动目标的数据集（如jogging）,这里引入了Jogging里面的两个目标。
    % 例如“Jogging.1”或“Jogging.2”中的点和数字.1&.2
	if numel(video) >= 2 && video(end-1) == '.' && ~isnan(str2double(video(end))),
    % 如果能依次满足 从前到后 视频数>=2 且 视频后缀有'.' 且 最后一位后缀非空 
    % ‘&&’与运算，a && b，只要a不满足条件就不会检查b***********************
    
		suffix = video(end-1:end);  %remember the suffix 记住该后缀
		video = video(1:end-2);     %remove it from the video name 从视频名字中移除后缀 （除去了最后两位'.'和'1'或'2'符号）
    else
		suffix = '';                %否则后缀为空
	end

	% full path to the video's files 视频文件的完整路径
	if base_path(end) ~= '/' && base_path(end) ~= '\',
		base_path(end+1) = '/';
    % 若路径填写不完整，给补上 '/'保证正确读取*******************************
	end
	video_path = [base_path video '/'];

	%try to load ground truth from text file (Benchmark's format)
    %载入文件名是'groundtruth_rect'，后缀是'.txt'的文件
	filename = [video_path 'groundtruth_rect' suffix '.txt'];
    %打开groundtruth,没有就显示...suffix后缀
	f = fopen(filename);
	assert(f ~= -1, ['No initial position or ground truth to load ("' filename '").'])
	% assert为断言函数，在程序中确保某些条件成立，否则调用系统 error 函数终止运行。(注意其格式)*****************
    
	%the format is [x, y, width, height] 格式是坐标x,y,宽，高
    
	try
		ground_truth = textscan(f, '%f,%f,%f,%f', 'ReturnOnError',false); 
    % 定义ground_truth为读入前面定义的‘f’,textscan函数功能——将已打开的文本文件中的
    % 数据读取到元胞数组即ground_truth，格式为双精度64位浮点数%f
    
    % 当 textscan 未能读取或转换数据时的行为，指定为由 'ReturnOnError' 和 true/false **********
    % 组成的逗号分隔对组。如果是 true，则 textscan 终止，不产生错误，***************************
    % 返回所有读取的字段。如果是 false，则 textscan 终止，产生错误，不返回输出元胞数组。*********
    
	catch  %#ok, try different format (no commas)
    % 编辑器出现了警告， 在语句后面写入 %#ok, 就可以去除编辑器的警告
		frewind(f);                                 % frewind 位置指针移至文件首部************
                                                    % 就是把位置移到文件的第一个数字开始准备读数
		ground_truth = textscan(f, '%f %f %f %f');  
	end
	ground_truth = cat(2, ground_truth{:});         % 设定ground_truth数据读取后按行联结（n*1）
	fclose(f);                                      % catenate函数cat(1,)按行联结矩阵*********
	
	%set initial position and size
	target_sz = [ground_truth(1,4), ground_truth(1,3)];                       % 设定目标的宽高
	pos = [ground_truth(1,2), ground_truth(1,1)] + floor(target_sz/2);        % 设定目标的坐标
	                                                 % floor()向负无穷大方向取整,即向下取整
	if size(ground_truth,1) == 1,                    % 返回矩阵ground_truth的行数若恒等于1
		%we have ground truth for the first frame only (initial%position) 只在第一帧时需要
		ground_truth = [];
	else
		%store positions instead of boxes            % 如果ground_truth行数不为1
		ground_truth = ground_truth(:,[2,1]) + ground_truth(:,[4,3]) / 2;
        
    end                                              % ??????????????????????????????????????????????????
	                                                 % groundtruth_rect文件里数字排列有不同的情况
	                                                 % 这里可能是因为元胞数组cell而不是array的问题
                                                     % 因为前面ground_truth = cat(2, ground_truth{:})
                                                     % ??????????????????????????????????????????????????
                                                     
	%from now on, work in the subfolder where all the images are
	%从现在开始读文件夹里的图片
	video_path = [video_path 'img/']; 
    %这里指具体化到数据集下video里的img文件里的图片文件，赋给video_path 
	
	%for these sequences, we must limit ourselves to a range of frames.
	%for all others, we just load all png/jpg files in the folder.
    
    %对于以下的几个序列，我们限制帧数的范围，其他的随意
    
	frames = {'David', 300, 770;
			  'Football1', 1, 74;
			  'Human3', 200, 1500;
			  'Freeman4', 1, 296;
              'Dudek', 5, 1145;
              };
              %'Tiger1',6,354};     
              %'Girl2',190,1500
              %为什么设置限制，我看了下数据集大概是一些画面灰度值很低
              %实在是无法跟踪，作者做了特殊处理，使一些frame精度达到了%100
	
	idx = find(strcmpi(video, frames(:,1)));  
    
              %比较video与frames的第一列，返回0或1，
              %比较，匹配为1，不同为0把非零的数据返回给idx
	
	if isempty(idx),
		%general case, just list all images 
        %一般情况(没有上述的几个特殊的video)，列举所有图片
		img_files = dir([video_path '*.png']);
		if isempty(img_files),
			img_files = dir([video_path '*.jpg']);
			assert(~isempty(img_files), 'No image files to load.')
            
               %读取图片，assert为断言函数，表示除非文件夹*******************
               %是非空的，否则显示'No image files to load.'
               
		end
		img_files = sort({img_files.name}); %把读到的图片patch以名字按升序排列
	else
		%list specified frames. try png first, then jpg.列出指定的那几个图像集
        
		if exist(sprintf('%s%04i.png', video_path, frames{idx,2}), 'file'),
			img_files = num2str((frames{idx,2} : frames{idx,3})', '%04i.png');
            
        % 这里注意‘%s%04i’的格式问题
		% b = exist( 'name', 'kind') kind 表示 name 的类型，kind可以取的
        % 值为：class（类），dir（文件夹），file（文件或文件夹），var（变量）等
        % 这里 判断video_path里有无''David''这样的文件，同时看frames矩阵idx.2位置
        % 即有没有'David', 300，第300帧图片，exist是否存在
            
		elseif exist(sprintf('%s%04i.jpg', video_path, frames{idx,2}), 'file'),           
            img_files = num2str((frames{idx,2} : frames{idx,3})', '%04i.jpg');
			% 找到图像后从第一帧（比如david是第300帧作为第一帧）开始到最后一帧
            % 赋给img_files
            % num2str函数功能：把数值转换成字符串， 转换后可以使用fprintf或disp函数进行输出
            
            % （...frames{idx,3})'注意这里的‘ ' ’,  去掉后会报错； 这里是矩阵的转置A’
		else
			error('No image files to load.')
		end
		
		img_files = cellstr(img_files);
        % cellstr()是将字符数组转换成cell类型为string的cell array的函数
        % 例：
        % X = ['string_1'; 'string_2'; 'string_3']  
        % Z = cellstr(X) ------------输出结果为：
        % X =
        % 0001.jpg
        % 0002.jpg
        % 0003.jpg
        % Z = 
        % '0001.jpg'
        % '0002.jpg'
        % '0003.jpg'
	end
	
end

