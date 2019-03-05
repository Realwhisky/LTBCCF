
% 这个是运行算法前加载的工具箱以及Matconvnet的编译

run('D:\视觉目标跟踪实验2018\目标跟踪算法2018\LTBCCF\vlfeat-0.9.20\toolbox\vl_setup')
run('D:\视觉目标跟踪实验2018\目标跟踪算法2018\LTBCCF\external\matconvnet\matlab\vl_setupnn')
run('D:\视觉目标跟踪实验2018\目标跟踪算法2018\LTBCCF\external\matconvnet\matlab\vl_compilenn')

% vl_compilenn('enablegpu',true)
% vl_compilenn('enableGpu', true, 'cudaRoot', 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5')
% vl_compilenn('enableGpu', true, 'cudaRoot', 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5','cudaMethod' ,'nvcc')
% vl_compilenn('enableGpu', true, 'cudaRoot', 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5','cudaMethod' ,'nvcc','verbose', '2')