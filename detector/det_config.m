% *******************  detector检测模块配置，这里由SVM算法组成（更改了前面的随机阙fern算法） *******************
% *******************          HOG+SVM 特征参数的设置，IOU “重叠度”阈值设定              *******************
% **************************************** IOU最大最小窗口设置 *********************************************

function config=det_config(target_sz, image_sz)

% frame_min_width = 320;                每帧最小宽度
% trackwin_max_dimension = 64;          跟踪窗口最大维度64
% template_max_numel = 144;             最大模版数144
% % frame_sz = size(frame);             每帧大小
% ***************************************************************************************
% if max(target_sz) <= trackwin_max_dimension ||...    
%         frame_sz(2) <= frame_min_width
%     config.image_scale = 1;
%     如果最大目标尺寸<=跟踪窗口最大维度64或第二帧图像尺寸<=图像最小宽度，就把 图像尺寸配置为1
% ***************************************************************************************
% else
%     min_scale = frame_min_width/frame_sz(2);
%     否则————最小尺度= 帧最小宽度/该帧宽
%     config.image_scale = max(trackwin_max_dimension/max(target_sz),min_scale);  
%     配置图片尺度=括号里最大者：（跟踪窗口最大维度64/目标的_宽或者高中最大的一个=&&=最小尺度）
% end
% ***************************************************************************************
% ————————————— 上面完成了配置 config.image_scale  ———————————————
%
% t_sz = target_sz*config.image_scale;                 % t_sz=目标大小*上面配置的图片尺度
% win_area = prod(t_sz);                               % 窗口区域--设置为t_sz元素的积
% config.ratio = (sqrt(template_max_numel/win_area));  % 最大模板数/窗口区域然后开根号=某比率
% 
% template_sz = round(t_sz*config.ratio);              
                % 模版大小= target_sz*config.image_scale*根下template_max_numel/ prod(t_sz)
                % round()四舍五入取整                   ？？？？？？？？？？？？了解算法公式
% config.template_sz = template_sz([2 1]);
                % 配置模版大小=template_sz([2 1])       ？？？？？？？？？？？？？？？？？？？
target_max_win = 144;                              % 定义目标最大窗口，设置为=144
                
config.ratio=sqrt(target_max_win/prod(target_sz)); % 设置ratio为 根下target_max_win/目标大小

config.t_sz=round(target_sz*config.ratio);         % 设置t_sz为target_sz*sqrt(target_max_win/prod(target_sz))

config.nbin=32;                                    % 设置nbin（HOG特征bin）为32

config.target_sz=target_sz;                        % 设置target_sz与image_sz
config.image_sz=image_sz;
                                                   % 以下出现的 IOU：
                                                   % 是预测出的bbox和实际标注的bbox的交集 除以 他们的并集。
                                                   % 显然，这个数值越大，说明预测的结果越好
                                                   % 计算IOU，找出当前类物体IOU最大的bbox， 如果这个最大值大于
                                                   % 预设的IOU的threshold，说明当前类物体分类正确
                                                   
config.thresh_p = 0.5; % IOU threshold for positive training samples  ############# IOU阈值，正样本设置为0.5
config.thresh_n = 0.1; % IOU threshold for negative ones              ############# IOU阈值，负样本设置为0.1

% IoU(Intersection over Union)
% Intersection over Union是一种测量在特定数据集中检测相应物体准确度的一个标准。我们可以在很多物体检测挑战中，例如PASCAL VOC challenge中看多很多使用该标准的做法

% 通常我们在 HOG + Linear SVM object detectors 和 Convolutional Neural Network detectors (R-CNN, Faster R-CNN, YOLO, etc.)中使用该方法检测其性能
% 注意，这个测量方法和你在任务中使用的物体检测算法没有关系

% IoU是一个简单的测量标准，只要是在输出中得出一个预测范围(bounding boxex)的任务都可以用IoU来进行测量。为了可以使IoU用于测量任意大小形状的物体检测，我们需要： 
% 1、 ground-truth bounding boxes（人为在训练集图像中标出要检测物体的大概范围）
% 2、我们的算法得出的结果范围

% 也就是说，这个标准用于测量真实和预测之间的相关度，相关度越高，该值越高

% simplify explanation：就是ground_truth框+实际跟踪框 重叠部分的面积/ground_truth框+实际跟踪框的总面积

% IOU reference ： https://blog.csdn.net/IAMoldpan/article/details/78799857
%                  https://blog.csdn.net/h_jlwg6688/article/details/76066890
                 

