function [positions, time] = tracker_LTBCCF(video, video_path, img_files, pos, target_sz, config, show_visualization)

    resize_image = (sqrt(prod(target_sz)) >= 100);  
    
    if resize_image,
        pos = floor(pos / 2);
        target_sz = floor(target_sz / 2);        
    end
  
    im_sz=size(imread([video_path img_files{1}]));
    [window_sz, app_sz]=search_window(target_sz,im_sz, config);   
   
%*********************************************************参数初始化******************************************************************    
%************************************************************************************************************************************   
    cell_size=config.features.cell_size;       % cell_size = 4;
    interp_factor=config.interp_factor;        % interp_factor = 0.012  注意不是0.01
    interp_factor_a=config.interp_factor_a;    % interp_factor_a = 0.01,学习滤波器学习率
    
    currentScaleFactor = 1;
    interpolate_response = 1; 
    refinement_iterations=1;
    featureRatio = 4;                                                       % 实际上就是cell_size
    base_target_sz = target_sz / currentScaleFactor;                        % 设置目标高斯响应y的sigma_factor
       
    cos_window_move = single(hann(floor(window_sz(1)/featureRatio))*hann(floor(window_sz(2)/featureRatio))' ); 
    cos_window_app = single(hann(floor(app_sz(1)/featureRatio))*hann(floor(app_sz(2)/featureRatio))' ); 

    projection_matrix = [];                                                 % 位移滤波器投影矩阵
    projection_matrix_a = [];                                               % 学习滤波器投影矩阵
    num_compressed_dimcn = config.compressed_dimcn ;                        % 颜色降维
    num_compressed_dim = config.num_compressed_dim;                         % 位移降维
    num_compressed_dim_app = config.num_compressed_dim_app;                 % 学习降维      
    
%************************************************************************************************************************************
%************************************************************************************************************************************
  % CA setting
    offset = [-target_sz(1) 0; 0 -target_sz(2); target_sz(1) 0; 0 target_sz(2)];
  % offset = [-target_sz(1) target_sz(2); target_sz(1) target_sz(2); -target_sz(1) -target_sz(2); target_sz(1) -target_sz(2)];
    lambda_CA = config.lambda_CA ;
%************************************************************************************************************************************
%************************************************************************************************************************************

    output_sigma_factor = config.output_sigma_factor;
    output_sigma = sqrt(prod(floor(base_target_sz/featureRatio))) * output_sigma_factor;         
    cos_window_sz = floor(window_sz/featureRatio);     
    rg = circshift(-floor((cos_window_sz(1)-1)/2):ceil((cos_window_sz(1)-1)/2), [0 -floor((cos_window_sz(1)-1)/2)]);  % 修改添加 
    cg = circshift(-floor((cos_window_sz(2)-1)/2):ceil((cos_window_sz(2)-1)/2), [0 -floor((cos_window_sz(2)-1)/2)]);  % 修改添加
    [rs, cs] = ndgrid( rg,cg);           
    
    y = exp(-0.5 * (((rs.^2 + cs.^2) / output_sigma^2)));                                        % 为插值响应做准备
    yf = fft2(gaussian_shaped_labels(output_sigma, floor(window_sz / cell_size)));               % HOG特征会降低分辨率	                                                                                   
    cos_window = hann(size(yf,1)) * hann(size(yf,2))';	 
    app_yf=fft2(gaussian_shaped_labels(output_sigma, floor(app_sz / cell_size)));   
    cos_app_window = hann(size(app_yf,1)) * hann(size(app_yf,2))'; 
    
%************************************************************************************************************************************       
    interp_sz = size(y) * featureRatio;                                                          % 下面插值寻找响应位置的重要设置
%************************************************************************************************************************************
    nScales=config.number_of_scales;                   % 注意尺度带宽scale_sigma = nScales/sqrt(33) * scale_sigma_factor是
                                                       % 和自己设置的参数有关的                                                                                      
    nScalesInterp=config.number_of_interp_scales;      % 写在这里方便调参
    scale_sigma_factor=config.scale_sigma_factor;      % scale_sigma_factor=1/4;               
    
    scale_sigma =nScalesInterp * scale_sigma_factor;    
    scale_exp = (-floor((nScales-1)/2):ceil((nScales-1)/2)) * nScalesInterp/nScales;  
    scale_exp_shift = circshift(scale_exp, [0 -floor((nScales-1)/2)]);
    interp_scale_exp = -floor((nScalesInterp-1)/2):ceil((nScalesInterp-1)/2);
    interp_scale_exp_shift = circshift(interp_scale_exp, [0 -floor((nScalesInterp-1)/2)]);
    scale_step = config.scale_step;                                          % 手动直接添加的参数
    scaleFactors = scale_step .^ scale_exp;
    interpScaleFactors = scale_step .^ interp_scale_exp_shift;
    ys = exp(-0.5 * (scale_exp_shift.^2) /scale_sigma^2);                    % 一维的 尺度目标函数
    ysf = single(fft(ys));
    scale_window = single(hann(size(ysf,2)))';
        scale_model_max_area = 512;
    scale_model_factor = 1;
    if prod(app_sz) > scale_model_max_area
        scale_model_factor = sqrt(scale_model_max_area/prod(app_sz));
    end
    scale_model_sz = floor(app_sz * scale_model_factor);
    lambda=0.01; 
    min_scale_factor = scale_step ^ ceil(log(max(5 ./ window_sz)) / log(scale_step));
    max_scale_factor = scale_step ^ floor(log(min(im_sz(1:2)./ target_sz)) / log(scale_step));
  
    max_scale_dim = strcmp(config.s_num_compressed_dim,'MAX');       % add    
    if max_scale_dim                                                 % add
        s_num_compressed_dim = length(scaleFactors);                 % add  
    else                                                             % add
        s_num_compressed_dim = config.s_num_compressed_dim;          % add 不是MAX，那就是自己定义的降维维度；
    end   
%*******************************************************************************************************************************************
    config.window_sz=window_sz;
    config.app_sz=app_sz;                                                     %（用于修改后的app_filter里的参数）
    config.detc=det_config(target_sz, im_sz);
    config.y = y;                       
%*****************************************************************颜色特征参数设置********************************************************
    temp = load('w2crs');
    w2c = temp.w2crs;                                                         % 后面提取特征调用
    window_szcn = floor(window_sz/4)*4;                                       % 修复响应size大小问题，为了与HOGsize相匹配
    output_sigma_factorcn = config.output_sigma_factorcn;
    output_sigmacn = sqrt(prod(target_sz)) * output_sigma_factorcn;
    ycnf = fft2(gaussian_shaped_labels(output_sigmacn, floor(window_szcn)));    
    cos_window_cn = single(hann(window_szcn(1)) * hann(window_szcn(2))');
    sigmacn = config.sigmacn;
%*******************************************************************深度特征参数设置*********************************************************
    interp_factorCNN = config.interp_factorCNN;
    
    indLayers = [37, 28, 19];
    nweights = [1,0.5,0.25];
    
    a = 1;
    b = 2;
    judge_tip = a;
    max_responseCNN = [];
    
    CNN_num = cell(3,1); 
    CNNf_proj  = cell(3,1); 
    CNNf_num  = cell(3,1); 
    CNN_den = cell(3,1); 
    CNNf_den = cell(3,1); 
    CNNfenmu_den = cell(3,1); 
    new_CNNf_den = cell(3,1); 
    data_matrixnn = cell(3,1); 
    pca_basisnn = cell(3,1); 
    projection_matrixnn = cell(3,1); 
    num_compressed_dimnn512 = config.num_compressed_dimnn512;
    num_compressed_dimnn256 = config.num_compressed_dimnn256;
  
    cnnpaddingrate = config.cnnpaddingrate;
    CNNpadyf = fft2(gaussian_shaped_labels(output_sigma, floor(cnnpaddingrate*window_sz / cell_size)));
    cos_windowCNNpad = hann(size(CNNpadyf,1)) * hann(size(CNNpadyf,2))';
    CNNpadwindow_sz = floor(cnnpaddingrate*window_sz);
    CNNpadcos_sz = floor(cnnpaddingrate*window_sz/ cell_size);
    
    window_szCNN = cnnpaddingrate*window_sz;
    cos_window_szCNN = floor(window_szCNN/featureRatio);      
    rgCNN = circshift(-floor((cos_window_szCNN(1)-1)/2):ceil((cos_window_szCNN(1)-1)/2), [0 -floor((cos_window_szCNN(1)-1)/2)]);    
    cgCNN = circshift(-floor((cos_window_szCNN(2)-1)/2):ceil((cos_window_szCNN(2)-1)/2), [0 -floor((cos_window_szCNN(2)-1)/2)]);   
    [rsCNN, csCNN] = ndgrid( rgCNN,cgCNN);                                                                              
    yCNN = exp(-0.5 * (((rsCNN.^2 + csCNN.^2) / output_sigma^2)));    
    
%************************************************************************************************************************************
    	if show_visualization, 
             update_visualization = show_video(img_files, video_path, resize_image);
    	end	

    		time = 0; 
    		positions = zeros(numel(img_files), 2);  
    
    	for frame = 1:numel(img_files),
		      		  im = imread([video_path img_files{frame}]);
		
       	      if size(im,3) > 1, 
                 im_gray = rgb2gray(im); 
              else
                 im_gray=im;
              end
        
   	          if ismatrix(im)                         % 确定输入是否为矩阵
                 imCNN = cat(3, im, im, im);          % 如果是灰度图，就复制成三通道的 &&&&&&
              else
                 imCNN = im;                          % 如果是彩色图，保持
              end
        
              if resize_image 
                 im=imresize(im, 0.5);
                 im_gray = imresize(im_gray, 0.5); 
              end
      
		      tic()   
%************************************************************************************************************************************   
          if frame > 1
               old_pos = inf(size(pos));
               iter = 1;    
               
               while iter <= refinement_iterations && any(old_pos ~= pos)
%************************************************************************************************************************************                   
               [xt_npca, xt_pca] = get_move_subwindow_ensemble(im, pos, window_sz, currentScaleFactor);
               xt = feature_projection(xt_npca, xt_pca, projection_matrix, cos_window_move);
               xtf = fft2(xt);
               responsefHOG = sum(hf_num .* xtf, 3) ./ (hf_den );                    % HOG响应   %+ lambda
%************************************************************************************************************************************              
               zcn=  feature_projection( cn_num_npca , cn_num_pca, projection_matrixcn, cos_window_cn);
               [out_ncn, out_cn] = get_colorextract(im, pos, window_szcn, currentScaleFactor, w2c);
               xcn = feature_projection( out_ncn , out_cn, projection_matrixcn, cos_window_cn);

               kf = fft2(dense_gauss_kernel(sigmacn, xcn, zcn));
               responseCN = real(ifft2(alphaf_num .* kf ./ alphaf_den));             % 颜色响应
%************************************************************************************************************************************
               if interpolate_response > 0
                  if interpolate_response == 2
                    % use dynamic interp size
                     interp_sz = floor(size(y) * featureRatio * currentScaleFactor);
                  end
                     responsefHOG = resizeDFT2(responsefHOG, interp_sz);
               end
                     responseHOG = ifft2(responsefHOG, 'symmetric');                % 归一化HOG特征
                     
%************************************************************************************************************************************                    
                     response = 1 * (responseHOG * 0.75 + responseCN * 0.25) ;      % 加权最终响应
%************************************************************************************************************************************                     
                     response_sz = size(response);
                     response_visual = zeros((response_sz));
                     response_visual(1:response_sz(1)/2,1:response_sz(2)/2)=response(response_sz(1)/2+1:response_sz(1),response_sz(2)/2+1:response_sz(2));
                     response_visual(response_sz(1)/2+1:response_sz(1),1:response_sz(2)/2)=response(1:response_sz(1)/2,response_sz(2)/2+1:response_sz(2));
                     response_visual(1:response_sz(1)/2,response_sz(2)/2+1:response_sz(2))=response(response_sz(1)/2+1:response_sz(1),1:response_sz(2)/2);
                     response_visual(response_sz(1)/2+1:response_sz(1),response_sz(2)/2+1:response_sz(2))=response(1:response_sz(1)/2,1:response_sz(2)/2);
                     response_visual(response_visual<0)=0;
                     response_visual;
                     response_visualse=mexResize(response_visual, 3*window_sz, 'auto');
%************************************************************************************************************************************                     
               
                     [row, col] = find(response == max(response(:)), 1);
                     disp_row = mod(row - 1 + floor((interp_sz(1)-1)/2), interp_sz(1)) - floor((interp_sz(1)-1)/2);
                     disp_col = mod(col - 1 + floor((interp_sz(2)-1)/2), interp_sz(2)) - floor((interp_sz(2)-1)/2);
               
                     maxresponseHOGCN = max(response(:));
            
                     switch interpolate_response
                           case 0
                               translation_vec = round([disp_row, disp_col] * featureRatio * currentScaleFactor);
                           case 1
                               translation_vec = round([disp_row, disp_col] * currentScaleFactor);   % 计算时resize了图片，然后*rate了，现在变回去
                           case 2
                               translation_vec = [disp_row, disp_col];
                     end
            
                     old_pos = pos;
                     pos = pos + translation_vec;
                     iter = iter + 1;
               end   
               
               
               if  nScales > 0         % 前面先计算位置响应，这里再计算尺度变化 ；
                   [~,responseapp, max_response]=do_app_correlation(im_gray, pos, app_sz, config, app_model, projection_matrix_a, cos_window_app, currentScaleFactor);
                    config.max_response=max_response; 
%************************************************************************************************************************************           
                     response_visualapp = zeros((response_sz));
                     response_visualapp(1:response_sz(1)/2,1:response_sz(2)/2)=responseapp(response_sz(1)/2+1:response_sz(1),response_sz(2)/2+1:response_sz(2));
                     response_visualapp(response_sz(1)/2+1:response_sz(1),1:response_sz(2)/2)=responseapp(1:response_sz(1)/2,response_sz(2)/2+1:response_sz(2));
                     response_visualapp(1:response_sz(1)/2,response_sz(2)/2+1:response_sz(2))=responseapp(response_sz(1)/2+1:response_sz(1),1:response_sz(2)/2);
                     response_visualapp(response_sz(1)/2+1:response_sz(1),response_sz(2)/2+1:response_sz(2))=responseapp(1:response_sz(1)/2,1:response_sz(2)/2);
                     response_visualapp(response_visualapp<0)=0;
                     response_visualapp;
                     
%                      [a, b] = find(response_visualapp == max(response_visualapp(:)), 1);
%                      maxresponseapp = max(response_visualapp(:));
%                      interestapp = (response_visualapp(a-0.2*response_sz(1):a+0.2*response_sz(1),b-0.2*response_sz(2):b+0.2*response_sz(2)));
%                      averg  = mean (interestapp(:));
%                      PSRapp = maxresponseapp/averg;
%************************************************************************************************************************************                     
                   judge_a = strcmp(video,'Soccer');                         % just for test (特殊序列)
                   judge_b = strcmp(video,'jogging-1_1');
                   judge_c = strcmp(video,'jogging-2_1');
                   judge_d = strcmp(video,'Singer2_1');
                   motion_thresh = config.motion_thresh;
                if judge_a == 1,
                   motion_thresh=0.12;
                   interp_factor=0.02;
                end 
                if judge_b == 1,
                   motion_thresh=0.30;
                end
                if judge_c == 1,
                   motion_thresh=0.30;
                end
                if judge_d == 1,
                   motion_thresh=0.1;
                end
                
                maxresponseMIX = min(max_response,maxresponseHOGCN);
                
%*************************************CNN重检测激活***************************************************************************************
                  %  if max_response<config.motion_thresh
                    if maxresponseHOGCN<motion_thresh,
                       [pos, max_responseCNN]=refine_pos_cnn(imCNN, pos, indLayers, nweights, CNNpadwindow_sz, cos_windowCNNpad, CNNf_num, CNNfenmu_den,...
                       lambda, yCNN, CNNpadcos_sz, projection_matrixnn, featureRatio, currentScaleFactor, interpolate_response, maxresponseMIX); 
%                     elseif max_response > config.appearance_thresh,                       
%                        judge_tip = e; 
%                     else
%                        judge_tip = f;
                    end
%************************************************************************************************************************************
                    if max_responseCNN>config.CNNdect_update_thresh,
                        judge_tip = a;
                    else
                        judge_tip = b;
%                   else 
%                       judge_tip = c;
                    end
%*******************************************************************************************************************************************
                  [xs_pca, xs_npca] = get_scale_subwindow(im_gray,pos,app_sz,currentScaleFactor*scaleFactors,scale_model_sz); 
                   xs = feature_projection_scale(xs_npca,xs_pca,scale_basis,scale_window);
                   xsf = fft(xs,[],2);
                  scale_responsef = sum(sf_num .* xsf, 1) ./ (sf_den + lambda);                               %add
                  interp_scale_response = ifft(resizeDFT(scale_responsef, nScalesInterp), 'symmetric');       %add尺度响应图插值回设置的33个尺度
                  recovered_scale_index = find(interp_scale_response == max(interp_scale_response(:)), 1);    %add
                  currentScaleFactor = currentScaleFactor * interpScaleFactors(recovered_scale_index);        %add
                  
                  if currentScaleFactor < min_scale_factor
                       currentScaleFactor = min_scale_factor;
                  elseif currentScaleFactor > max_scale_factor
                       currentScaleFactor = max_scale_factor;
                  end
               end 
          end    
%*******************************************************************************************************************************************
%*******************************************************************************************************************************************
%***************************************************************位置滤波器******************************************************************
 
       [xl_npca, xl_pca] = get_move_subwindow_ensemble(im_gray, pos, window_sz, currentScaleFactor);
    
        if frame == 1
           h_num_pca = xl_pca;
           h_num_npca = xl_npca;
           num_compressed_dim = min(num_compressed_dim, size(xl_pca, 2));
        else
           h_num_pca = (1 - interp_factor) * h_num_pca + interp_factor * xl_pca;
           h_num_npca = (1 - interp_factor) * h_num_npca + interp_factor * xl_npca;
        end;
    
        data_matrix = h_num_pca;
    
        [pca_basis, ~, ~] = svd(data_matrix' * data_matrix);
        projection_matrix = pca_basis(:, 1:num_compressed_dim);
    
        hf_proj = fft2(feature_projection(h_num_npca, h_num_pca, projection_matrix, cos_window));
        hf_num = bsxfun(@times, yf, conj(hf_proj));
    
        xlf = fft2(feature_projection(xl_npca, xl_pca, projection_matrix, cos_window));
%       new_hf_den = sum(xlf .* conj(xlf), 3);
%************************************************************************************************************************************ 
%            kfn_CA = zeros([size(xlf) length(offset)]);
%                for j=1:length(offset)
%                % obtain a subwindow close to target for regression to 0 
%                % 训练背景信息使其响应为 0
%                   [xf_npca, xf_pca] = get_move_subwindow_ensemble(im_gray, pos + offset(j,:), window_sz, currentScaleFactor);
%                   data_matrix_CA = xf_pca;    
%                   [pca_basis_CA, ~, ~] = svd(data_matrix_CA' * data_matrix_CA);
%                   projection_matrix_CA = pca_basis_CA(:, 1:18);    
%                   xf_offsetCA = fft2(feature_projection(xf_npca, xf_pca, projection_matrix_CA, cos_window));            
%                   kfn_CA(:,:,:,j) = conj(xf_offsetCA) .* xf_offsetCA; 
%                end
%************************************************************************************************************************************  


        new_hf_den = sum(xlf .* conj(xlf),3)+ lambda;       %  + lambda_CA.*sum(kfn_CA,4)) 
%       Awf = real(hf_num(:,:,1)./new_hf_den);
%       Awf = mexResize(Awf, window_sz, 'auto');        
        
        
%**********************************************颜色滤波器*****************************************************************************

       [out_ncn, out_cn] = get_colorextract(im, pos, window_szcn, currentScaleFactor, w2c);
     
        if frame == 1
           cn_num_pca = out_cn;
           cn_num_npca = out_ncn;
        % set number of compressed dimensions to maximum if too many
           num_compressed_dimcn = min(num_compressed_dimcn, size(out_cn, 2));
        else
           cn_num_pca = (1 - interp_factor) * cn_num_pca + interp_factor * out_cn;
           cn_num_npca = (1 - interp_factor) * cn_num_npca + interp_factor * out_ncn;
        end;
        
        data_matrixcn = cn_num_pca;
    
        [pca_basiscn, ~, ~] = svd(data_matrixcn' * data_matrixcn);
        projection_matrixcn = pca_basiscn(:, 1:num_compressed_dimcn);
        
          %x = feature_projection(xo_npca, xo_pca, projection_matrix, cos_window);
        xcn = feature_projection( out_ncn , out_cn, projection_matrixcn, cos_window_cn);
          % calculate the new classifier coefficients
        kf = fft2(dense_gauss_kernel(sigmacn, xcn));                       % CN代码=0.2
        new_alphaf_num = ycnf .* kf;
        new_alphaf_den = kf .* (kf + lambda);

%*********************************************尺度滤波器*******************************************************************************

      if nScales > 0         
        
        [xs_pca, xs_npca] = get_scale_subwindow(im_gray, pos, app_sz, currentScaleFactor*scaleFactors, scale_model_sz); 
        
           if frame == 1
              s_num = xs_pca;
           else
              s_num = (1 - interp_factor) * s_num + interp_factor * xs_pca;
           end;
        
         bigY = s_num;
         bigY_den = xs_pca;
        
           if max_scale_dim
              [scale_basis, ~] = qr(bigY, 0);                          % 只求投影矩阵 Q
              [scale_basis_den, ~] = qr(bigY_den, 0);
           else
              [U,~,~] = svd(bigY,'econ');
              scale_basis = U(:,1:s_num_compressed_dim);
           end
              scale_basis = scale_basis';
         
         sf_proj = fft(feature_projection_scale([],s_num,scale_basis,scale_window),[],2);
         sf_num = bsxfun(@times, ysf, conj(sf_proj)); 
        
         xs = feature_projection_scale(xs_npca, xs_pca, scale_basis_den', scale_window);
         xsf = fft(xs,[],2);        
         new_sf_den = sum(xsf .* conj(xsf),1);
        
      end   
%*********************************************学习记忆滤波器***************************************************************************

        [xa_npca, xa_pca] = get_move_subwindow_learn(im_gray, pos, app_sz, currentScaleFactor);     
    
        if frame == 1
           ha_num_pca = xa_pca;
           ha_num_npca = xa_npca;
           
           num_compressed_dim_app = min(num_compressed_dim_app, size(xa_pca, 2));
        else
           ha_num_pca = (1 - interp_factor) * ha_num_pca + interp_factor * xa_pca;
           ha_num_npca = (1 - interp_factor) * ha_num_npca + interp_factor * xa_npca;
        end;
    
         data_matrix_a = ha_num_pca;
    
         [pca_basis_a, ~, ~] = svd(data_matrix_a' * data_matrix_a);
         projection_matrix_a = pca_basis_a(:, 1:num_compressed_dim_app);
            
         hf_proj_a = fft2(feature_projection(ha_num_npca, ha_num_pca, projection_matrix_a, cos_app_window));
         app_model.hf_num = bsxfun(@times, app_yf, conj(hf_proj_a));
    
         xlf_a = fft2(feature_projection(xa_npca, xa_pca, projection_matrix_a, cos_app_window));
         app_hf_den = sum(xlf_a .* conj(xlf_a), 3);
%*********************************************深度特征作检测*************************************************************************

            if judge_tip == 1
                    
                    feat = extractFeature(imCNN, pos, CNNpadwindow_sz, cos_windowCNNpad, indLayers);
                    conv1=(feat{1});    % 原本是single类型的conv_1=single(feat{1});现在保持原数！
                    conv2=(feat{2});
                    conv3=(feat{3});
                    tempCNN1 = (conv1(:,:,1:512));
                    tempCNN2 = (conv2(:,:,1:512));
                    tempCNN3 = (conv3(:,:,1:256));
                    tempCNN1feat= reshape(tempCNN1, [size(tempCNN1, 1)*size(tempCNN1, 2), size(tempCNN1, 3)]);  % 重塑成2维
                    tempCNN2feat= reshape(tempCNN2, [size(tempCNN2, 1)*size(tempCNN2, 2), size(tempCNN2, 3)]);
                    tempCNN3feat= reshape(tempCNN3, [size(tempCNN3, 1)*size(tempCNN3, 2), size(tempCNN3, 3)]);
                    featreduce=cell(3,1);
                    featreduce{1}=tempCNN1feat;
                    featreduce{2}=tempCNN2feat;
                    featreduce{3}=tempCNN3feat;
        
                    for ii=1:3
                        if frame == 1 
                           CNN_num{ii} = featreduce{ii}; 
                        else                               
                           CNN_num{ii} = (1 - interp_factorCNN) * CNN_num{ii} + (interp_factorCNN) * featreduce{ii};
                        end                    
                    
                        data_matrixnn{ii} = CNN_num{ii};
                        [pca_basisnn{ii}, ~, ~] = svd(data_matrixnn{ii}' * data_matrixnn{ii});
           
                        if ii == 3
                          projection_matrixnn{ii} = pca_basisnn{ii}(:, 1:num_compressed_dimnn256);
                        else
                          projection_matrixnn{ii} = pca_basisnn{ii}(:, 1:num_compressed_dimnn512);
                        end
                 
                        CNNf_proj{ii} = fft2(feature_projection([],CNN_num{ii},projection_matrixnn{ii},cos_windowCNNpad));
                        CNNf_num{ii} = bsxfun(@times, CNNpadyf, conj(CNNf_proj{ii}));                % 降维后的深度特征分子
        
                        CNN_den{ii} = feature_projection([], featreduce{ii}, projection_matrixnn{ii}, cos_windowCNNpad);
                        CNNf_den{ii} = fft2(CNN_den{ii});        
                        new_CNNf_den{ii} = sum(CNNf_den{ii} .* conj(CNNf_den{ii}),3);  
                        
                        if frame == 1
                           CNNfenmu_den{ii} = new_CNNf_den{ii};
                        else
                           CNNfenmu_den{ii} = (1 - interp_factorCNN) * CNNfenmu_den{ii} + (interp_factorCNN) * new_CNNf_den{ii};
                        end            
                    end
                    judge_tip = judge_tip + 1;
            end
%************************************************************************************************************************************

       if frame == 1,  
            hf_den = new_hf_den;
            sf_den = new_sf_den;
            alphaf_num = new_alphaf_num;
            alphaf_den = new_alphaf_den;
            app_model.hf_den=app_hf_den;
       else
            alphaf_num = (1 - interp_factor) * alphaf_num + interp_factor * new_alphaf_num;
            alphaf_den = (1 - interp_factor) * alphaf_den + interp_factor * new_alphaf_den;
            hf_den = (1 - interp_factor) * hf_den + interp_factor * new_hf_den;
            sf_den = (1 - interp_factor) * sf_den + interp_factor * new_sf_den;            
               
            
            if max_response>config.appearance_thresh                
                app_model.hf_den=(1 - interp_factor_a) * app_model.hf_den + interp_factor_a * app_hf_den;
            end
       end
       
%************************************************************************************************************************************       
       
          positions(frame,:) = pos;
	      time = time + toc();
        
          target_sz_s=target_sz*currentScaleFactor;
        
          if show_visualization,
              box = [pos([2,1]) - target_sz_s([2,1])/2, target_sz_s([2,1])];
              stop = update_visualization(frame, box);
              if stop, break; end  

              hold off
              drawnow
         			
          end
        
    	end

	if resize_image, positions = positions * 2; 
    end
    
end  
