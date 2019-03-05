%
%  Long-Term object tracking method based on background constraints and convolutional features v1.0
%  


function [precision, fps] = run_tracker(video, show_visualization, show_plots)

    addpath('utility');
    addpath('detector');
    addpath('scale');
   
    base_path  = 'D:\视觉目标跟踪实验2018\目标跟踪算法2018\LTBCCF\data_sequence\';
    
    %  temp = load('w2crs');
    %  w2c = temp.w2crs;
    
    global enableGPU;
    enableGPU = true;     %  true  ||   false
    
	if nargin < 1, video = 'choose'; end
	if nargin < 2, show_visualization = ~strcmp(video, 'all'); end
	if nargin < 3, show_plots = ~strcmp(video, 'all'); end    
    
    config.s_num_compressed_dim = 'MAX';  % 尺度滤波器降维，在程序中降为 nscales == 17    
	
	config.padding =1.86;                 % Padding大小窗设置  /1.8---2.3
    config.cnnpaddingrate = 1.5;          % =1既padding=1.86, =1.35既padding=2.5, =1.5既padding=2.8
    
    config.kernel_sigma = 1;
	config.lambda = 1e-4;                 % 正则参数
    config.lambda_CA = 0.4;
	config.output_sigma_factor=0.01;       
    config.output_sigma_factorcn=1/16;
    config.scale_sigma_factor=1/16;       % 1/16
    config.sigmacn = 0.2;
    
    config.interp_factorCNN = 0.015;
    config.interp_factor=0.01;            % Best 0.01
    config.interp_factor_a=0.01;          % 专门给学习滤波器的 
    config.compressed_dimcn = 6;          % 颜色特征降维
    config.num_compressed_dim=18;         % 平移滤波器降维压缩 (融合了别的特征需要注意)
    config.num_compressed_dim_app=18;     % 学习滤波器维度降维
    
    config.num_compressed_dimnn512=32;
    config.num_compressed_dimnn512=32;    % 深度学习卷积特征降维
    config.num_compressed_dimnn256=64;
    
    config.features.hog_orientations=9;   % 梯度方向设置
    config.features.cell_size=4;          % Hog grid cell大小		
    config.features.window_size=6;           
    config.features.nbins=8;               
    
    config.scale_step=1.02;               % 尺度大小调整
    config.number_of_scales=17;           % 降维后的尺度
    config.number_of_interp_scales=45;    % 插值数设置33~65为宜                           
    
    config.motion_thresh=0.05;            % 学习滤波器激发检测环节时阈值
    config.appearance_thresh=0.66;        % 学习滤波器更新阈值
    config.CNNdect_update_thresh=0.5;     % 深度特征滤波器学习阈值
      
    switch video
	case 'choose',
		%读取视频
		video = choose_video(base_path);
		if ~isempty(video),
			[precision, fps] = run_tracker(video,show_visualization, show_plots);
            
			if nargout == 0, 
				clear precision
			end
		end
				
	case 'all',
		% 一次性跑目录下所有视频
		dirs = dir(base_path);
		videos = {dirs.name};
		videos(strcmp('.', videos) | strcmp('..', videos) | ...
			strcmp('anno', videos) | ~[dirs.isdir]) = [];

		videos(strcmpi('Jogging', videos)) = [];
		videos(end+1:end+2) = {'Jogging.1', 'Jogging.2'};
		
		all_precisions = zeros(numel(videos),1);  
		all_fps = zeros(numel(videos),1);
		
		if ~exist('parpool', 'file'),
			
			for k = 1:numel(videos),
				[all_precisions(k), all_fps(k)] = run_tracker(videos{k}, show_visualization, show_plots);
			end
		else
			
			if parpool('size') == 0,
				parpool open;
			end
			parfor k = 1:numel(videos),
				[all_precisions(k), all_fps(k)] = run_tracker(videos{k}, show_visualization, show_plots);
			end
		end
				
		mean_precision = mean(all_precisions);
		fps = mean(all_fps);
		fprintf('\nAverage precision (20px):% 1.3f, Average FPS:% 4.2f\n\n', mean_precision, fps)
		if nargout > 0,
			precision = mean_precision;
        end
		
		
	otherwise
		
		[img_files, pos, target_sz, ground_truth, video_path] = load_video_info(base_path, video);
	
		[positions, time] = tracker_LTBCCF(video, video_path, img_files, pos, target_sz, config, show_visualization);
		
		precisions = precision_plot(positions, ground_truth, video, show_plots);
		fps = numel(img_files) / time;

		fprintf('%12s - Precision (20px):% 1.3f, FPS:% 4.2f\n', video, precisions(20), fps)

		if nargout > 0,
			
			precision = precisions(20);
        end
        
	end
end
