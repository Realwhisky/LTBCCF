%
%  Long-term Correlation Tracking v3.0
%  
% 
%
%  Note that in this updated version, an online SVM classifier is used for 
%  recovering targets and the color channels are quantized as features for
%  detector learning


function [precision, fps] = run_tracker(video, show_visualization, show_plots)

    addpath('utility');
    addpath('detector');
    addpath('scale');
   % base_path  = 'D:\data_sequence\';
    base_path  = 'D:\视觉目标跟踪实验2018\目标跟踪算法2018\CFNet\cfnet-master\data\OTB_validation\';
    
    %  temp = load('w2crs');
    %  w2c = temp.w2crs;
    
    global enableGPU;
    enableGPU = true;     %  true  ||   false
    
	%  default settings
	if nargin < 1, video = 'choose'; end
	if nargin < 2, show_visualization = ~strcmp(video, 'all'); end
	if nargin < 3, show_plots = ~strcmp(video, 'all'); end    
    
    config.s_num_compressed_dim = 'MAX';  % 尺度滤波器降维，在程序中降为 nscales == 17    
	
	config.padding =1.86;                 % extra area surrounding the target  /1.8---2.3
    config.cnnpaddingrate = 1.2;         % =1既padding=1.86, =1.35既padding=2.5, =1.5既padding=2.8
    
    config.kernel_sigma=1;
	config.lambda = 1e-4;                 % regularization parameter 
    config.lambda_CA = 0.4;
	config.output_sigma_factor=0.01;      % 
    config.output_sigma_factorcn=1/16;
    config.scale_sigma_factor=1/16;       % 1/16
    config.sigmacn = 0.2;
    
    config.interp_factorCNN = 0.015;
    config.interp_factor=0.01;            % best 0.01
    config.interp_factor_a=0.01;          % 专门 给学习滤波器的 
    config.compressed_dimcn = 6;          % 颜色特征  降维  …………………………………
    config.num_compressed_dim=18;         % 平移滤波器降维压缩 (融合了别的特征需要注意)
    config.num_compressed_dim_app=18;     % 学习滤波器维度降维  ……………………………  
    
    config.num_compressed_dimnn512=32;
    config.num_compressed_dimnn512=32;    % 深度学习特征降维 *************
    config.num_compressed_dimnn256=64;
    
    config.features.hog_orientations=9;   % init 9
    config.features.cell_size=4;          % size of hog grid cell		
    config.features.window_size=6;        % size of local region for intensity historgram  
    config.features.nbins=8;              % bins of intensity historgram  /8
    
    config.scale_step=1.02;               % 尺度大小调整
    config.number_of_scales=17;           % 自己add，也就是程序里的nScales=config.number_of_scales;
    config.number_of_interp_scales=45;    % 自己add                               
    
    config.motion_thresh=0.05;            % 学习滤波器激发检测环节时阈值
    config.appearance_thresh=0.66;        % 学习滤波器更新阈值
    config.CNNdect_update_thresh=0.5;     % 分数低于0.5的时候要不要存储一个模板？Size问题？
      
    switch video
	case 'choose',
		%ask the user for the video, then call self with that video name.
		video = choose_video(base_path);
		if ~isempty(video),
			[precision, fps] = run_tracker(video,show_visualization, show_plots);
            
			if nargout == 0,  %don't output precision as an argument
				clear precision
			end
		end
				
	case 'all',
		%all videos, call self with each video name.
		
		%only keep valid directory names
		dirs = dir(base_path);
		videos = {dirs.name};
		videos(strcmp('.', videos) | strcmp('..', videos) | ...
			strcmp('anno', videos) | ~[dirs.isdir]) = [];
		
		%the 'Jogging' sequence has 2 targets, create one entry for each.
		%we could make this more general if multiple targets per video
		%becomes a common occurence.
		videos(strcmpi('Jogging', videos)) = [];
		videos(end+1:end+2) = {'Jogging.1', 'Jogging.2'};
		
		all_precisions = zeros(numel(videos),1);  %to compute averages
		all_fps = zeros(numel(videos),1);
		
		if ~exist('parpool', 'file'),
			%no parallel toolbox, use a simple 'for' to iterate
			for k = 1:numel(videos),
				[all_precisions(k), all_fps(k)] = run_tracker(videos{k}, show_visualization, show_plots);
			end
		else
			%evaluate trackers for all videos in parallel
			if parpool('size') == 0,
				parpool open;
			end
			parfor k = 1:numel(videos),
				[all_precisions(k), all_fps(k)] = run_tracker(videos{k}, show_visualization, show_plots);
			end
		end
		
		%compute average precision at 20px, and FPS
		mean_precision = mean(all_precisions);
		fps = mean(all_fps);
		fprintf('\nAverage precision (20px):% 1.3f, Average FPS:% 4.2f\n\n', mean_precision, fps)
		if nargout > 0,
			precision = mean_precision;
        end
		
		
	otherwise
		%we were given the name of a single video to process
	
		%get image file names, initial state, and ground truth for evaluation
		[img_files, pos, target_sz, ground_truth, video_path] = load_video_info(base_path, video);
		
        
		%call tracker function with all the relevant parameters
		[positions, time] = tracker_lct(video, video_path, img_files, pos, target_sz, config, show_visualization);
		
		
		%calculate and show precision plot, as well as frames-per-second
		precisions = precision_plot(positions, ground_truth, video, show_plots);
		fps = numel(img_files) / time;

		fprintf('%12s - Precision (20px):% 1.3f, FPS:% 4.2f\n', video, precisions(20), fps)

		if nargout > 0,
			%return precisions at a 20 pixels threshold
			precision = precisions(20);
        end
        
	end
end
