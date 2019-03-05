% *******************  detector���ģ�����ã�������SVM�㷨��ɣ�������ǰ��������fern�㷨�� *******************
% *******************          HOG+SVM �������������ã�IOU ���ص��ȡ���ֵ�趨              *******************
% **************************************** IOU�����С�������� *********************************************

function config=det_config(target_sz, image_sz)

% frame_min_width = 320;                ÿ֡��С���
% trackwin_max_dimension = 64;          ���ٴ������ά��64
% template_max_numel = 144;             ���ģ����144
% % frame_sz = size(frame);             ÿ֡��С
% ***************************************************************************************
% if max(target_sz) <= trackwin_max_dimension ||...    
%         frame_sz(2) <= frame_min_width
%     config.image_scale = 1;
%     ������Ŀ��ߴ�<=���ٴ������ά��64��ڶ�֡ͼ��ߴ�<=ͼ����С��ȣ��Ͱ� ͼ��ߴ�����Ϊ1
% ***************************************************************************************
% else
%     min_scale = frame_min_width/frame_sz(2);
%     ���򡪡�������С�߶�= ֡��С���/��֡��
%     config.image_scale = max(trackwin_max_dimension/max(target_sz),min_scale);  
%     ����ͼƬ�߶�=����������ߣ������ٴ������ά��64/Ŀ���_����߸�������һ��=&&=��С�߶ȣ�
% end
% ***************************************************************************************
% �������������������������� ������������� config.image_scale  ������������������������������
%
% t_sz = target_sz*config.image_scale;                 % t_sz=Ŀ���С*�������õ�ͼƬ�߶�
% win_area = prod(t_sz);                               % ��������--����Ϊt_szԪ�صĻ�
% config.ratio = (sqrt(template_max_numel/win_area));  % ���ģ����/��������Ȼ�󿪸���=ĳ����
% 
% template_sz = round(t_sz*config.ratio);              
                % ģ���С= target_sz*config.image_scale*����template_max_numel/ prod(t_sz)
                % round()��������ȡ��                   �������������������������˽��㷨��ʽ
% config.template_sz = template_sz([2 1]);
                % ����ģ���С=template_sz([2 1])       ��������������������������������������
target_max_win = 144;                              % ����Ŀ����󴰿ڣ�����Ϊ=144
                
config.ratio=sqrt(target_max_win/prod(target_sz)); % ����ratioΪ ����target_max_win/Ŀ���С

config.t_sz=round(target_sz*config.ratio);         % ����t_szΪtarget_sz*sqrt(target_max_win/prod(target_sz))

config.nbin=32;                                    % ����nbin��HOG����bin��Ϊ32

config.target_sz=target_sz;                        % ����target_sz��image_sz
config.image_sz=image_sz;
                                                   % ���³��ֵ� IOU��
                                                   % ��Ԥ�����bbox��ʵ�ʱ�ע��bbox�Ľ��� ���� ���ǵĲ�����
                                                   % ��Ȼ�������ֵԽ��˵��Ԥ��Ľ��Խ��
                                                   % ����IOU���ҳ���ǰ������IOU����bbox�� ���������ֵ����
                                                   % Ԥ���IOU��threshold��˵����ǰ�����������ȷ
                                                   
config.thresh_p = 0.5; % IOU threshold for positive training samples  ############# IOU��ֵ������������Ϊ0.5
config.thresh_n = 0.1; % IOU threshold for negative ones              ############# IOU��ֵ������������Ϊ0.1

% IoU(Intersection over Union)
% Intersection over Union��һ�ֲ������ض����ݼ��м����Ӧ����׼ȷ�ȵ�һ����׼�����ǿ����ںܶ���������ս�У�����PASCAL VOC challenge�п���ܶ�ʹ�øñ�׼������

% ͨ�������� HOG + Linear SVM object detectors �� Convolutional Neural Network detectors (R-CNN, Faster R-CNN, YOLO, etc.)��ʹ�ø÷������������
% ע�⣬�����������������������ʹ�õ��������㷨û�й�ϵ

% IoU��һ���򵥵Ĳ�����׼��ֻҪ��������еó�һ��Ԥ�ⷶΧ(bounding boxex)�����񶼿�����IoU�����в�����Ϊ�˿���ʹIoU���ڲ��������С��״�������⣬������Ҫ�� 
% 1�� ground-truth bounding boxes����Ϊ��ѵ����ͼ���б��Ҫ�������Ĵ�ŷ�Χ��
% 2�����ǵ��㷨�ó��Ľ����Χ

% Ҳ����˵�������׼���ڲ�����ʵ��Ԥ��֮�����ضȣ���ض�Խ�ߣ���ֵԽ��

% simplify explanation������ground_truth��+ʵ�ʸ��ٿ� �ص����ֵ����/ground_truth��+ʵ�ʸ��ٿ�������

% IOU reference �� https://blog.csdn.net/IAMoldpan/article/details/78799857
%                  https://blog.csdn.net/h_jlwg6688/article/details/76066890
                 

