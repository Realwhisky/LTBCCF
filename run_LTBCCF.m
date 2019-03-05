function results = run_LTBCCF(seq, res_path, bSaveImage)

% �����������OTB�ϲ��ԵĽӿڣ�

    addpath('./utility');

    img_files = seq.s_frames;
    target_sz = [seq.init_rect(1,4), seq.init_rect(1,3)];
	pos = [seq.init_rect(1,2), seq.init_rect(1,1)] + floor(target_sz/2);
    
%***************************************************************************************************************        
    config.s_num_compressed_dim = 'MAX';  % �߶��˲�����ά���ڳ����н�Ϊ nscales == 17    
	
	config.padding =1.86;                 
    config.cnnpaddingrate = 2.5;          % =1��padding=1.86, =1.35��padding=2.5, =1.5��padding=2.8
    
    config.kernel_sigma=1;
	config.lambda = 1e-4;                 % �������  
	config.output_sigma_factor=0.1;       % 
    config.output_sigma_factorcn=1/16;
    config.scale_sigma_factor=1/16;       % 1/16
    config.sigmacn = 0.2;
    
    config.interp_factorCNN = 0.02;
    config.interp_factor=0.01;            % best 0.01
    config.interp_factor_a=0.01;          % ר�� ��ѧϰ�˲����� 
    config.compressed_dimcn = 8;          % ��ɫ����  ��ά  ��������������������������
    config.num_compressed_dim=18;         % ƽ���˲�����άѹ�� (�ں��˱��������Ҫע��)
    config.num_compressed_dim_app=18;     % ѧϰ�˲���ά�Ƚ�ά  ����������������������  
    
    config.num_compressed_dimnn512=64;
    config.num_compressed_dimnn512=64;    % ���ѧϰ������ά *************
    config.num_compressed_dimnn256=32;
    
    config.features.hog_orientations=9;   
    config.features.cell_size=4;         
    config.features.window_size=6;        
    config.features.nbins=8;             
    
    config.scale_step=1.02;               % �߶ȴ�С����
    config.number_of_scales=17;           % �Լ�add��Ҳ���ǳ������nScales=config.number_of_scales;
    config.number_of_interp_scales=45;    % �Լ�add                               
    
    config.motion_thresh=0.22;            % ѧϰ�˲���������⻷��ʱ��ֵ
    config.appearance_thresh=0.75;        % ѧϰ�˲���������ֵ
    config.CNNdect_update_thresh=0.5;     
%***************************************************************************************************************     

    show_visualization=0;
       
    video_path='';
    video = seq.name;
    [positions, time] = tracker_lct(video, video_path, img_files, pos, target_sz, config, show_visualization);
    %��benchmark,����variable���
    rects = [positions(:,2) - target_sz(2)/2, positions(:,1) - target_sz(1)/2];
    rects(:,3) = target_sz(2);
    rects(:,4) = target_sz(1);
    res.type = 'rect';
    res.res = rects;
    res.fps = numel(img_files)/time;
    
    results=res;   
   
end