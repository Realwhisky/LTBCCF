function [img_files, pos, target_sz, ground_truth, video_path] = load_video_info(base_path, video)


%LOAD_VIDEO_INFO
%   Loads all the relevant information for the video in the given path:
%   the list of image files (cell array of strings), initial position
%   (1x2), target size (1x2), the ground truth information for precision
%   calculations (Nx2, for N frames), and the path where the images are
%   located. The ordering of coordinates and sizes is always [y, x].
%   �ڸ���·���м�����Ƶ�����������Ϣ��ͼ���ļ��б�(�ַ�����Ԫ����)����ʼ
%   λ��(1x2)��Ŀ���С(1x2)��ground truth�Ļ�����Ϣ(Nx2������N֡)
%   �Լ�ͼ�����ڵ�·��������ʹ�С����������[y��x]��


%***********************************�ȶ�·�����ٶ���Ƶ���ٶ�ground_truth********************************


	% see if there's a suffix, specifying one of multiple targets, for
	% example the dot and number in 'Jogging.1' or 'Jogging.2'.
    % �鿴�Ƿ��к�׺����һ���ж���˶�Ŀ������ݼ�����jogging��,����������Jogging���������Ŀ�ꡣ
    % ���硰Jogging.1����Jogging.2���еĵ������.1&.2
	if numel(video) >= 2 && video(end-1) == '.' && ~isnan(str2double(video(end))),
    % ������������� ��ǰ���� ��Ƶ��>=2 �� ��Ƶ��׺��'.' �� ���һλ��׺�ǿ� 
    % ��&&�������㣬a && b��ֻҪa�����������Ͳ�����b***********************
    
		suffix = video(end-1:end);  %remember the suffix ��ס�ú�׺
		video = video(1:end-2);     %remove it from the video name ����Ƶ�������Ƴ���׺ ����ȥ�������λ'.'��'1'��'2'���ţ�
    else
		suffix = '';                %�����׺Ϊ��
	end

	% full path to the video's files ��Ƶ�ļ�������·��
	if base_path(end) ~= '/' && base_path(end) ~= '\',
		base_path(end+1) = '/';
    % ��·����д�������������� '/'��֤��ȷ��ȡ*******************************
	end
	video_path = [base_path video '/'];

	%try to load ground truth from text file (Benchmark's format)
    %�����ļ�����'groundtruth_rect'����׺��'.txt'���ļ�
	filename = [video_path 'groundtruth_rect' suffix '.txt'];
    %��groundtruth,û�о���ʾ...suffix��׺
	f = fopen(filename);
	assert(f ~= -1, ['No initial position or ground truth to load ("' filename '").'])
	% assertΪ���Ժ������ڳ�����ȷ��ĳЩ�����������������ϵͳ error ������ֹ���С�(ע�����ʽ)*****************
    
	%the format is [x, y, width, height] ��ʽ������x,y,����
    
	try
		ground_truth = textscan(f, '%f,%f,%f,%f', 'ReturnOnError',false); 
    % ����ground_truthΪ����ǰ�涨��ġ�f��,textscan�������ܡ������Ѵ򿪵��ı��ļ��е�
    % ���ݶ�ȡ��Ԫ�����鼴ground_truth����ʽΪ˫����64λ������%f
    
    % �� textscan δ�ܶ�ȡ��ת������ʱ����Ϊ��ָ��Ϊ�� 'ReturnOnError' �� true/false **********
    % ��ɵĶ��ŷָ����顣����� true���� textscan ��ֹ������������***************************
    % �������ж�ȡ���ֶΡ������ false���� textscan ��ֹ���������󣬲��������Ԫ�����顣*********
    
	catch  %#ok, try different format (no commas)
    % �༭�������˾��棬 ��������д�� %#ok, �Ϳ���ȥ���༭���ľ���
		frewind(f);                                 % frewind λ��ָ�������ļ��ײ�************
                                                    % ���ǰ�λ���Ƶ��ļ��ĵ�һ�����ֿ�ʼ׼������
		ground_truth = textscan(f, '%f %f %f %f');  
	end
	ground_truth = cat(2, ground_truth{:});         % �趨ground_truth���ݶ�ȡ�������ᣨn*1��
	fclose(f);                                      % catenate����cat(1,)�����������*********
	
	%set initial position and size
	target_sz = [ground_truth(1,4), ground_truth(1,3)];                       % �趨Ŀ��Ŀ��
	pos = [ground_truth(1,2), ground_truth(1,1)] + floor(target_sz/2);        % �趨Ŀ�������
	                                                 % floor()���������ȡ��,������ȡ��
	if size(ground_truth,1) == 1,                    % ���ؾ���ground_truth�������������1
		%we have ground truth for the first frame only (initial%position) ֻ�ڵ�һ֡ʱ��Ҫ
		ground_truth = [];
	else
		%store positions instead of boxes            % ���ground_truth������Ϊ1
		ground_truth = ground_truth(:,[2,1]) + ground_truth(:,[4,3]) / 2;
        
    end                                              % ??????????????????????????????????????????????????
	                                                 % groundtruth_rect�ļ������������в�ͬ�����
	                                                 % �����������ΪԪ������cell������array������
                                                     % ��Ϊǰ��ground_truth = cat(2, ground_truth{:})
                                                     % ??????????????????????????????????????????????????
                                                     
	%from now on, work in the subfolder where all the images are
	%�����ڿ�ʼ���ļ������ͼƬ
	video_path = [video_path 'img/']; 
    %����ָ���廯�����ݼ���video���img�ļ����ͼƬ�ļ�������video_path 
	
	%for these sequences, we must limit ourselves to a range of frames.
	%for all others, we just load all png/jpg files in the folder.
    
    %�������µļ������У���������֡���ķ�Χ������������
    
	frames = {'David', 300, 770;
			  'Football1', 1, 74;
			  'Human3', 200, 1500;
			  'Freeman4', 1, 296;
              'Dudek', 5, 1145;
              };
              %'Tiger1',6,354};     
              %'Girl2',190,1500
              %Ϊʲô�������ƣ��ҿ��������ݼ������һЩ����Ҷ�ֵ�ܵ�
              %ʵ�����޷����٣������������⴦��ʹһЩframe���ȴﵽ��%100
	
	idx = find(strcmpi(video, frames(:,1)));  
    
              %�Ƚ�video��frames�ĵ�һ�У�����0��1��
              %�Ƚϣ�ƥ��Ϊ1����ͬΪ0�ѷ�������ݷ��ظ�idx
	
	if isempty(idx),
		%general case, just list all images 
        %һ�����(û�������ļ��������video)���о�����ͼƬ
		img_files = dir([video_path '*.png']);
		if isempty(img_files),
			img_files = dir([video_path '*.jpg']);
			assert(~isempty(img_files), 'No image files to load.')
            
               %��ȡͼƬ��assertΪ���Ժ�������ʾ�����ļ���*******************
               %�Ƿǿյģ�������ʾ'No image files to load.'
               
		end
		img_files = sort({img_files.name}); %�Ѷ�����ͼƬpatch�����ְ���������
	else
		%list specified frames. try png first, then jpg.�г�ָ�����Ǽ���ͼ��
        
		if exist(sprintf('%s%04i.png', video_path, frames{idx,2}), 'file'),
			img_files = num2str((frames{idx,2} : frames{idx,3})', '%04i.png');
            
        % ����ע�⡮%s%04i���ĸ�ʽ����
		% b = exist( 'name', 'kind') kind ��ʾ name �����ͣ�kind����ȡ��
        % ֵΪ��class���ࣩ��dir���ļ��У���file���ļ����ļ��У���var����������
        % ���� �ж�video_path������''David''�������ļ���ͬʱ��frames����idx.2λ��
        % ����û��'David', 300����300֡ͼƬ��exist�Ƿ����
            
		elseif exist(sprintf('%s%04i.jpg', video_path, frames{idx,2}), 'file'),           
            img_files = num2str((frames{idx,2} : frames{idx,3})', '%04i.jpg');
			% �ҵ�ͼ���ӵ�һ֡������david�ǵ�300֡��Ϊ��һ֡����ʼ�����һ֡
            % ����img_files
            % num2str�������ܣ�����ֵת�����ַ����� ת�������ʹ��fprintf��disp�����������
            
            % ��...frames{idx,3})'ע������ġ� ' ��,  ȥ����ᱨ�� �����Ǿ����ת��A��
		else
			error('No image files to load.')
		end
		
		img_files = cellstr(img_files);
        % cellstr()�ǽ��ַ�����ת����cell����Ϊstring��cell array�ĺ���
        % ����
        % X = ['string_1'; 'string_2'; 'string_3']  
        % Z = cellstr(X) ------------������Ϊ��
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

